"""
Dummy script to check generated training data.
"""

from utils.train_utils import load_hdf5
import cv2
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('hdf5_file_path', type=str, required=True)
    args = parser.parse_args()

    data = load_hdf5(args.hdf5_file_path)
    cv2.imshow('colors', data['colors'])
    cv2.imshow('gt', data['segmap'] * 255)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
