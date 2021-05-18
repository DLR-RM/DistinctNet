"""
Dummy script to check generated training data.
"""

import os
import cv2
import argparse
import glob
import numpy as np


def overlay_im_with_mask(im, ma, alpha=0.5):
    im_col = im.copy()
    a, b = np.where(ma == 1)
    if a != []:
        im_col[a, b, :] = np.array([0, 165, 255])
    a, b = np.where(ma == 2)
    if a != []:
        im_col[a, b, :] = np.array([0, 255, 0])
    a, b = np.where(ma >= 3)
    if a != []:
        im_col[a, b, :] = np.array([0, 0, 255])
    im_overlay = im.copy()
    im_overlay = cv2.addWeighted(im_overlay, alpha, im_col, 1 - alpha, 0.0)

    return im_overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    args = parser.parse_args()

    assert os.path.exists(args.root), f"Invalid path: {args.root}"
    dirs = os.listdir(args.root)
    rgb_dirs = sorted([d for d in dirs if 'rgb' in d])
    gt_dirs = sorted([d for d in dirs if 'gt' in d])
    assert len(rgb_dirs) == len(gt_dirs), f"There are not equally many rgb ({len(rgb_dirs)}) and gt ({len(gt_dirs)}) folders"
    print(f"Found {len(rgb_dirs)} rgb folders and {len(gt_dirs)} gt folders")

    for rgb_dir, gt_dir in zip(rgb_dirs, gt_dirs):
        suffs = sorted(os.listdir(os.path.join(args.root, rgb_dir)))

    for d in rgb_dirs:
        assert len(os.listdir(os.path.join(args.root, d))) == len(suffs), f"Found different number of files in rgb directories"
    for d in gt_dirs:
        assert len(os.listdir(os.path.join(args.root, d))) == len(suffs), f"Found different number of files in gt directories"

    print(f"Found {len(suffs)} sequences of length {len(rgb_dirs)}")

    for suff in suffs:
        for i, (rgb_dir, gt_dir) in enumerate(zip(rgb_dirs, gt_dirs)):
            im = cv2.imread(os.path.join(args.root, rgb_dir, suff))
            ma = cv2.imread(os.path.join(args.root, gt_dir, suff), cv2.IMREAD_GRAYSCALE)
            masked = overlay_im_with_mask(im, ma)
            cv2.imshow(f"overlay_{i}", masked)
            cv2.imshow(rgb_dir, im)
            cv2.imshow(gt_dir, ma / ma.max())
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
