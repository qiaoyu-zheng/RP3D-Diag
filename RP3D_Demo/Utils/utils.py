import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import random
import datetime

def visual_augment(B, images, out_folder):
    image_datas = images.cpu().numpy()
    size = image_datas.shape[2]
    new_slice_size = (size, size)
    b = math.ceil(math.sqrt(B))
    big_image = np.zeros((new_slice_size[0] * b, new_slice_size[1] * b))
    for i in tqdm(range(B), desc="processing"):
        image_data = image_datas[i,:,:,:]
        randpick = random.randrange(0,image_data.shape[-1])
        image_data = image_data[:,:,randpick]
        # image_data = image_data[:,:,0].numpy()
        # image_data = cv2.resize(image_data, new_slice_size)
        x_start = i // b * new_slice_size[0]
        x_end = x_start + new_slice_size[0]
        y_start = i % b * new_slice_size[1]
        y_end = y_start + new_slice_size[1]
        # print(x_start,x_end,y_start,y_end)
        big_image[x_start:x_end, y_start:y_end] = image_data
        # except:
            # print(nifty_list[i])
            # continue

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S-") + str(current_time.microsecond // 1000)
    # # 显示大图像
    plt.imshow(big_image, cmap='gray')
    plt.axis('off')
    plt.savefig(f"{out_folder}/aug_{formatted_time}.png", bbox_inches='tight')