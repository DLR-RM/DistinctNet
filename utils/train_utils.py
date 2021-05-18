"""
Various training utilities.
"""

import os
import argparse
import yaml
from yacs.config import CfgNode
import numpy as np
import h5py
from pathlib import PosixPath
import json
import torch


def get_default_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="", metavar="FILE", help="path to config file(s)")
    parser.add_argument("OPTS", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()

    assert os.path.isfile(args.config_file)
    with open(args.config_file, 'r') as f:
        cfg = CfgNode(yaml.load(f))

    cfg.merge_from_list(args.OPTS)

    return cfg


def load_hdf5(path, keys=None):
    if type(path) is PosixPath:
        path = path.as_posix()
    assert os.path.isfile(path), f"File {path} does not exist"
    assert '.hdf5' in path, f"File {path} is not a hdf5 file"

    with h5py.File(path, 'r') as data:
        data_keys = [key for key in data.keys()]
        if keys is None:
            keys = data_keys
        else:
            assert [key in data_keys for key in keys], f"Invalid keys {keys} for data keys {data_keys}"

        hdf5_data = {}
        for key in keys:
            if data[key].dtype.char == 'S':
                try:
                    hdf5_data[key] = json.loads(bytes(np.array(data[key])))[0]
                except:
                    hdf5_data[key] = data[key]
            else:
                hdf5_data[key] = np.array(data[key])

    return hdf5_data


def str_to_tens(string):
    return torch.tensor([ord(c) for c in string])


def tens_to_str(tens):
    return [chr(c) for c in tens]
