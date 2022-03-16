from math import ceil
from random import sample
from datetime import datetime
import os
import logging
import argparse
import gzip
import json
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Arguments for sampling tasks from environment episodes.')
    parser.add_argument(
        "-frac",
        default=0.5,
        type=float,
        help="Percent of tasks from train and val to sample from entire set",
    )

    return parser.parse_args()

def read_gz(frac, gz_path, save_folder):
    with gzip.open(gz_path, 'r') as file:
        json_bytes = file.read()
    
    file_name = os.path.basename(os.path.normpath(file.name))
    logger.info(f"Beginning on: {file_name}")
    json_str = json_bytes.decode('utf-8')
    data = json.loads(json_str)
    # Dictionaries representing the counts and id of each task type
    # The key is defined as (difficulty, target)
    count_dict = dict()
    id_dict = dict()

    for idx, floor_plan_meta in enumerate(data):
        difficulty = floor_plan_meta['difficulty']
        target = floor_plan_meta['object_type']
        key = (difficulty, target)
        
        if key in count_dict.keys():
            count_dict[key] += 1
        else:
            count_dict[key] = 1

        if key in id_dict.keys():
            id_dict[key].append(idx)
        else:
            id_dict[key] = [idx]

    assert count_dict.keys() == id_dict.keys()
    sub_sample_list = list()
    data_np = np.array(data)

    for key, count in count_dict.items():
        id_list = id_dict[key]
        sub_sample_count = ceil(count * frac)

        if sub_sample_count == count:
            logger.warning(f"File name: {file_name}, key: {key} has too few tasks: {count}, sub-sample equal to {count}")

        sampled_ids_list = sample(id_list, sub_sample_count)
        sub_sample_list = sub_sample_list + list(data_np[sampled_ids_list])
        logger.info(f"File name: {file_name}, key: {key}, original count: {count}, sub-sample count: {sub_sample_count}, sampled ids: {sampled_ids_list}")
    
    with gzip.open(f'{save_folder}/{file_name}', 'w') as fout:
        json_str = json.dumps(sampled_ids_list)
        json_bytes = json_str.encode('utf-8')
        fout.write(json_bytes)
    
    logger.info(f"Completed: {file_name}")

def iterate_over_ep_folder(folder_path, frac, save_folder):
    for file in os.listdir(folder_path):
        if file.endswith(".gz"):
            file_path_abs = os.path.join(folder_path, file)
            read_gz(frac=frac, gz_path=file_path_abs, save_folder=save_folder)

if __name__ == "__main__":
    dt = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    logging.basicConfig(filename=f'dylan_scripts/subsample_task_logs/{dt}.txt')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    cmd_args = get_args()
    train_folder_path = "/home/dyung6/robustnav/datasets/robothor-objectnav/train/episodes"
    val_folder_path = "/home/dyung6/robustnav/datasets/robothor-objectnav/val/episodes"
    subsample_task_folder = f"/home/dyung6/robustnav/dylan_scripts/subsample_task/{dt}"
    train_save_folder = save_folder=subsample_task_folder + '/train'
    val_save_folder = subsample_task_folder + '/val'

    os.mkdir(subsample_task_folder)
    os.mkdir(train_save_folder)
    os.mkdir(val_save_folder)

    logger.info(f'Running: {train_folder_path}')
    iterate_over_ep_folder(folder_path=train_folder_path, frac = cmd_args.frac, save_folder=train_save_folder)
    logger.info(f'Finished: {train_folder_path}')

    logger.info(f'Running: {val_folder_path}')
    iterate_over_ep_folder(folder_path=val_folder_path, frac = cmd_args.frac, save_folder=val_save_folder)
    logger.info(f'Finished: {val_folder_path}')    
