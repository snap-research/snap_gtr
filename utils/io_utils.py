import json
from pathlib import Path
import bz2
import pickle
import _pickle as cPickle
import numpy as np

import torch
from typing import Any

import yaml


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def get_list_from_file(in_file):
    with open(in_file, 'r') as fin:
        lines = fin.readlines()
    return [item.strip() for item in lines]


def save_list_to_file(out_file, data, verbose=False):
    if verbose:
        print(f"Save to {out_file}")
    with open(out_file, 'w') as fout:
        for item in data:
            fout.write(f'{item}\n')


def dump_pickle(file, data, verbose=False):
    if verbose:
        print(f"save to {file}")
    pikd = open(file, "wb")
    pickle.dump(data, pikd)
    pikd.close()


# loads and returns a pickled objects
def load_pickle(file):
    pikd = open(file, "rb")
    data = pickle.load(pikd)
    pikd.close()
    return data


# Pickle a file and then compress it into a file with extension
def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', "w") as f:
        cPickle.dump(data, f)


# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, "rb")
    data = cPickle.load(data)
    return data


def get_image_file(image_dir, item):
    img_file = (
        f"{image_dir}/{item}.png"
        if Path(f"{image_dir}/{item}.png").is_file()
        else f"{image_dir}/{item}.jpg"
    )
    return img_file


def print_dict(data):
    for key, value in data.items():
        if type(value) is torch.Tensor:
            print(key, 'tensor', value.shape, f'value range: {value.min().item():.3f}, {value.max().item():.3f}')
        elif type(value) is np.ndarray:
            print(key, 'np.ndarray', value.shape, f'value range: {value.min():.3f}, {value.max():.3f}')
        elif type(value) is str:
            print(key, value)
        else:
            print(f'{key} {type(value)}')


def compare_dicts(dict1, dict2):
    """
    Recursively compare if two dictionaries are equal by evaluating their values.
    """
    if len(dict1) != len(dict2):
        return False

    for key in dict1:
        if key not in dict2:
            return False

        value1 = dict1[key]
        value2 = dict2[key]

        if isinstance(value1, dict) and isinstance(value2, dict):
            if not compare_dicts(value1, value2):
                return False
        if isinstance(value1, list) and isinstance(value2, list):
            if not np.allclose(np.array(value1), np.array(value2), atol=1e-2):
                print(np.array(value1), np.array(value2))
                return False
        else:
            print(f"Need add comparison for {type(value1)}, {type(value2)}")
    return True


def read_yaml(fpath: str):
    with open(fpath, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def write_yaml(fpath, data):
    with open(fpath, 'w') as yaml_file:
        yaml.dump(data, yaml_file)


def read_json(fpath: str):
    assert Path(fpath).is_file, f"Cannot find {fpath}"
    with open(fpath, "r") as input_file:
        data = json.load(input_file)
    return data


def write_json(fpath: str, data):
    with open(fpath, "w") as json_file:
        json.dump(data, json_file)
