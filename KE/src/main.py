import logging
import os
import random
from datetime import datetime
from itertools import cycle


import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler

from torch.utils.data import Dataset, DataLoader, BatchSampler,ConcatDataset,RandomSampler

from transformers import AutoTokenizer


try:
    import wandb
except ImportError:
    wandb = None

from tensorboardX import SummaryWriter
try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from model import CLP_clinical,MedCPT_clinical
from dataload import UMLS_Dataset,UMLS_ICD_Dataset,ICD10_Dataset

from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr
from training.train import train_one_epoch



def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

# # 自定义 BatchSampler
# class CustomBatchSampler(BatchSampler):
#     def __iter__(self):
#         batch = []
#         dataset_probs = [0.01, 0.01, 0.1, 0.08, 0.8]
#         for _ in range(self.batch_size):
#             selected_dataset = torch.multinomial(torch.tensor(dataset_probs), 1).item()
#             indices = list(self.sampler)
#             batch.extend(indices)
#         yield batch

def main():
    args = parse_args()
    # discover initial world args early so we can log properly
    args.distributed = True
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    args.log_path = None
    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.output_dir,args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out2.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        # if os.path.exists(args.log_path):
        #     print(
        #         "Error. Experiment already exists. Use --name {} to specify a new experiment."
        #     )
        #     return -1

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # fully initialize distributed device environment
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    # device = init_distributed_device(args)
    args.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    if is_master(args):
        args.tensorboard_path = os.path.join(args.output_dir,args.logs, args.name, "tensorboard") if args.tensorboard else ''
        args.checkpoint_path = os.path.join(args.aws_output_dir,args.logs, args.name, "checkpoints")
        for dirname in [args.tensorboard_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
        for dirname in [args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''
        args.checkpoint_path = ''
    
    if args.copy_codebase:
        copy_codebase(args)

    logging.info(f'Running with Device {args.device}.')
    random_seed(args.seed, 0)
    if args.select_model == 'bert':
        model = CLP_clinical(bert_model_name = args.pretrained)
    else:
        model = MedCPT_clinical(bert_model_name = args.pretrained)
    model = torch.nn.DataParallel(model)
    model.to(device)

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.output_dir,args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")
    

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained)#,local_files_only=True

    optimizer = None
    scaler = None
    if args.mrdef_csv_file:
        assert not args.trace, 'Cannot train with traced model'
        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )
        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            if 'epoch' in checkpoint:
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                model.load_state_dict(sd)
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if scaler is not None and 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
                logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                model.load_state_dict(checkpoint)
                logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))
    
    # # initialize datasets  
    logging.info("LOADING Dataset")
    dataset1 = ICD10_Dataset(args.icd_json_file,args.icd_level_json_file,train_level=1)
    dataset2 = ICD10_Dataset(args.icd_json_file,args.icd_level_json_file,train_level=2)
    dataset3 = ICD10_Dataset(args.icd_json_file,args.icd_level_json_file,train_level=3)
    dataset4 = ICD10_Dataset(args.icd_json_file,args.icd_level_json_file,train_level=4)
    dataset5 = ICD10_Dataset(args.icd_json_file,args.icd_level_json_file,train_level=5)
    dataset6 = UMLS_Dataset(args.mrdef_csv_file, args.umls_kg_file, args.umls_cui_file,args.rparticle_data)
    num_samples = len(dataset1) + len(dataset2) + len(dataset3) + len(dataset4) + len(dataset5)+ len(dataset6)
    
    dataloader1 = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True)
    dataloader2 = DataLoader(dataset2, batch_size=args.batch_size, shuffle=True)
    dataloader3 = DataLoader(dataset3, batch_size=args.batch_size, shuffle=True)
    dataloader4 = DataLoader(dataset4, batch_size=args.batch_size, shuffle=True)
    dataloader5 = DataLoader(dataset5, batch_size=args.batch_size, shuffle=True)
    dataloader6 = DataLoader(dataset6, batch_size=args.batch_size, shuffle=True)


    # concat_dataset = ConcatDataset([dataset1, dataset2, dataset3, dataset4, dataset5])
    logging.info("FINISH DataLoader")
    dataloader_list = [dataloader1,dataloader2,dataloader3,dataloader4,dataloader5,dataloader6]
    num_batches = len(dataloader1) + len(dataloader2) + len(dataloader3) + len(dataloader4) + len(dataloader5) + len(dataloader6)
    total_steps = num_batches * args.epochs
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
    
    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = os.path.join(args.output_dir,args.logs) and os.path.join(args.output_dir,args.logs).lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        writer = SummaryWriter(args.tensorboard_path)

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')
        train_one_epoch(model, tokenizer, dataloader_list,num_batches,num_samples, epoch, optimizer, scaler, scheduler, args, writer)
        completed_epoch = epoch + 1

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(checkpoint_dict, os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"))

            if args.save_most_recent:
                torch.save(checkpoint_dict, os.path.join(args.checkpoint_path, f"epoch_state.pt"))

    if args.wandb and is_master(args):
        wandb.finish()


def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.output_dir,args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main()
