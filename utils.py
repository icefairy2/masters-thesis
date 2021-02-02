# Copyright (c) Facebook, Inc. and its affiliates.

import json
# file to store some often use functions
import os
import os.path as osp
import pickle
import shutil

import numpy as np


def build_dir(target_dir):
    if not osp.exists(target_dir):
        os.makedirs(target_dir)


def get_subdir(in_path):
    subdir_path = os.path.sep.join(in_path.split(os.path.sep)[:-1])
    return subdir_path


def make_subdir(in_path):
    subdir_path = get_subdir(in_path)
    build_dir(subdir_path)


def get_all_files(in_dir, extension, path_type='full', keywords=''):
    assert path_type in ['full', 'relative', 'name_only']
    assert isinstance(extension, str) or isinstance(extension, tuple)
    assert isinstance(keywords, str)

    all_files = list()
    for subdir, dirs, files in os.walk(in_dir):
        for file in files:
            if len(keywords) > 0:
                if file.find(keywords) < 0:
                    continue
            if file.endswith(extension):
                if path_type == 'full':
                    file_path = osp.join(subdir, file)
                elif path_type == 'relative':
                    file_path = osp.join(subdir, file).replace(in_dir, '')
                    if file_path.startswith('/'):
                        file_path = file_path[1:]
                else:
                    file_path = file
                all_files.append(file_path)
    return sorted(all_files)


# save data to pkl
def save_pkl(res_file, data_list, protocol=-1):
    assert res_file.endswith(".pkl")
    res_file_dir = '/'.join(res_file.split('/')[:-1])
    if len(res_file_dir) > 0:
        if not osp.exists(res_file_dir):
            os.makedirs(res_file_dir)
    with open(res_file, 'wb') as out_f:
        if protocol == 2:
            pickle.dump(data_list, out_f, protocol=2)
        else:
            pickle.dump(data_list, out_f)


def load_pkl(pkl_file):
    assert pkl_file.endswith(".pkl")
    with open(pkl_file, 'rb') as in_f:
        try:
            data = pickle.load(in_f)
        except UnicodeDecodeError:
            in_f.seek(0)
            data = pickle.load(in_f, encoding='latin1')
    return data


def load_json(in_file):
    assert in_file.endswith(".json")
    with open(in_file, 'r') as in_f:
        all_data = json.load(in_f)
        return all_data


def save_json(out_file, data):
    assert out_file.endswith(".json")
    with open(out_file, "w") as out_f:
        json.dump(data, out_f)
