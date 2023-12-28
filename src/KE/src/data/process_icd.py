import pandas as pd
import json

# 读取CSV文件
df = pd.read_csv('icd10_description.csv')

# 初始化一个空字典
icd_dict = {}

# 遍历数据框的每一行
for index, row in df.iterrows():
    icd_code = row['icd_code'][:3]  # 获取icd_code的前三位字符
    if icd_code not in icd_dict:
        icd_dict[icd_code] = []  # 初始化一个新的列表
    icd_dict[icd_code].append({'icd_code': row['icd_code'], 'long_title': row['long_title']})

# 将字典保存为JSON文件
with open('icd10_description.json', 'w') as json_file:
    json.dump(icd_dict, json_file, indent=4)

print("JSON文件已保存成功。")

icd_json_file = 'crawl_ICD_final1.json'
with open(icd_json_file, 'r') as file:
    id_json_data = json.load(file)
    
level1_keys = id_json_data.keys()
save_level_dict = {}
save_data_dict = {}
save_list = ['description','Includes','Applicable_To','Clinical_Information','Approximate_Synonyms','Note','Code_Also']
for level1_key in level1_keys:
    level1_info = id_json_data[level1_key]
    save_data_dict[level1_key] = {}
    save_level_dict[level1_key] = {}
    if 'description' in level1_info:
        save_data_dict[level1_key]['description'] = level1_info['description']
    if 'url' in level1_info:
        save_data_dict[level1_key]['url'] = level1_info['url']
    if "class" in level1_info:
        if 'Includes' in level1_info['class']:
            save_data_dict[level1_key]['Includes'] = level1_info['class']['Includes']
        if 'Applicable_To' in level1_info['class']:
            save_data_dict[level1_key]['Applicable_To'] = level1_info['class']['Applicable_To']
        if 'Clinical_Information' in level1_info['class']:
            save_data_dict[level1_key]['Clinical_Information'] = level1_info['class']['Clinical_Information']
        if 'Approximate_Synonyms' in level1_info['class']:
            save_data_dict[level1_key]['Approximate_Synonyms'] = level1_info['class']['Approximate_Synonyms']
        for level2_key in level1_info['class'].keys():
            if level2_key not in save_list:
                save_level_dict[level1_key][level2_key] = {}
                save_data_dict[level2_key] = {}
                level2_info = level1_info['class'][level2_key]
                if 'description' in level2_info:
                    save_data_dict[level2_key]['description'] = level2_info['description']
                if 'url' in level2_info:
                    save_data_dict[level2_key]['url'] = level2_info['url']
                if "class" in level2_info:
                    if 'Includes' in level2_info['class']:
                        save_data_dict[level2_key]['Includes'] = level2_info['class']['Includes']
                    if 'Applicable_To' in level2_info['class']:
                        save_data_dict[level2_key]['Applicable_To'] = level2_info['class']['Applicable_To']
                    if 'Clinical_Information' in level2_info['class']:
                        save_data_dict[level2_key]['Clinical_Information'] = level2_info['class']['Clinical_Information']
                    if 'Approximate_Synonyms' in level2_info['class']:
                        save_data_dict[level2_key]['Approximate_Synonyms'] = level2_info['class']['Approximate_Synonyms']
                    for level3_key in level2_info['class'].keys():
                        if level3_key not in save_list:
                            save_level_dict[level1_key][level2_key][level3_key] = {}
                            save_data_dict[level3_key] = {}
                            level3_info = level2_info['class'][level3_key]
                            if 'description' in level3_info:
                                save_data_dict[level3_key]['description'] = level3_info['description']
                            if 'url' in level3_info:
                                save_data_dict[level3_key]['url'] = level3_info['url']
                            if "class" in level3_info:
                                if 'Includes' in level3_info['class']:
                                    save_data_dict[level3_key]['Includes'] = level3_info['class']['Includes']
                                if 'Applicable_To' in level3_info['class']:
                                    save_data_dict[level3_key]['Applicable_To'] = level3_info['class']['Applicable_To']
                                if 'Clinical_Information' in level3_info['class']:
                                    save_data_dict[level3_key]['Clinical_Information'] = level3_info['class']['Clinical_Information']
                                if 'Approximate_Synonyms' in level3_info['class']:
                                    save_data_dict[level3_key]['Approximate_Synonyms'] = level3_info['class']['Approximate_Synonyms']
                                for level4_key in level3_info['class'].keys():
                                    if level4_key not in save_list:
                                        save_data_dict[level4_key] = {}
                                        save_level_dict[level1_key][level2_key][level3_key][level4_key] = {}
                                        level4_info = level3_info['class'][level4_key]
                                        if 'description' in level4_info:
                                            save_data_dict[level4_key]['description'] = level4_info['description']
                                        if 'url' in level4_info:
                                            save_data_dict[level4_key]['url'] = level4_info['url']
                                        if "class" in level4_info:
                                            if 'Includes' in level4_info['class']:
                                                save_data_dict[level4_key]['Includes'] = level4_info['class']['Includes']
                                            if 'Applicable_To' in level4_info['class']:
                                                save_data_dict[level4_key]['Applicable_To'] = level4_info['class']['Applicable_To']
                                            if 'Clinical_Information' in level4_info['class']:
                                                save_data_dict[level4_key]['Clinical_Information'] = level4_info['class']['Clinical_Information']
                                            if 'Approximate_Synonyms' in level4_info['class']:
                                                save_data_dict[level4_key]['Approximate_Synonyms'] = level4_info['class']['Approximate_Synonyms']
                                            for level5_key in level4_info['class'].keys():
                                                if level5_key not in save_list:
                                                    print(level5_key)
                                        
with open('train_icd.json', 'w') as json_file:
    json.dump(save_data_dict, json_file, indent=4)

with open('train_icd_level.json', 'w') as json_file:
    json.dump(save_level_dict, json_file, indent=4)
    
print("JSON文件已保存成功。")



with open('train_icd_level.json', 'r') as file:
    id_level_data = json.load(file)
level_dict = {'level1': [], 'level2': [], 'level3': [], 'level4': []}

for level1_key in id_level_data.keys():
    level_dict['level1'].append(level1_key)
    for level2_key in id_level_data[level1_key].keys():
        level_dict['level2'].append(level2_key)
        for level3_key in id_level_data[level1_key][level2_key].keys():
            level_dict['level3'].append(level3_key)
            for level4_key in id_level_data[level1_key][level2_key][level3_key].keys():
                level_dict['level4'].append(level4_key)

with open('train_icd_level_list.json', 'w') as json_file:
    json.dump(level_dict, json_file, indent=4)

with open('train_icd.json', 'r') as json_file:
    icd_data = json.load(json_file)

with open('train_icd_level.json', 'r') as file:
    id_level_data = json.load(file)
    
level_dict = {'level1': [], 'level2': [], 'level3': [], 'level4': []}

for level1_key in id_level_data.keys():
    if len(icd_data[level1_key].keys()) >2:
        level_dict['level1'].append(level1_key)
    for level2_key in id_level_data[level1_key].keys():
        if len(icd_data[level2_key].keys()) >2:
            level_dict['level2'].append(level2_key)
        for level3_key in id_level_data[level1_key][level2_key].keys():
            # level_dict['level3'].append(level3_key)
            # if 'Clinical_Information' in icd_data[level3_key].keys():
            # if len(icd_data[level3_key].keys()) >2:
            level_dict['level3'].append(level3_key)
            # else:
            #     print(level3_key)
            for level4_key in id_level_data[level1_key][level2_key][level3_key].keys():
                if len(icd_data[level4_key].keys()) >2:
                    level_dict['level4'].append(level4_key)

with open('train_icd_level_list_clean.json', 'w') as json_file:
    json.dump(level_dict, json_file, indent=4)