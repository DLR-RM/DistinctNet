"""
Default prediction script.
Predicts segmentation masks on a sequence of images.
"""

import os
import glob
import argparse
from PIL import Image
import cv2
from tqdm import trange
import numpy as np

from predictor import Predictor
from utils.pred_utils import overlay_im_with_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dict', type=str, required=True)
    parser.add_argument('--image-root', default='./recorded_images', type=str)
    args = parser.parse_args()

    assert os.path.exists(args.image_root), f"Invalid path: {args.image_root}"
    assert os.path.exists(os.path.join(args.image_root, 'rgb')), f"No 'rgb' folder in {args.image_root}"
    image_paths = sorted(glob.glob(os.path.join(args.image_root, 'rgb/*.png')))
    print(f"Found {len(image_paths)} images in {args.image_root}")

    os.makedirs(os.path.join(args.image_root, 'pred'), exist_ok=True)
    os.makedirs(os.path.join(args.image_root, 'pred_overlay'), exist_ok=True)
    predictor = Predictor(state_dict_path=args.state_dict)

    print(f"Starting predictions ...")
    im1 = np.array(Image.open(image_paths[0]))

    for i in trange(1, len(image_paths)):
        im2 = np.array(Image.open(image_paths[i]))

        pred = predictor.predict(im1, im2)
        pred_overlay = overlay_im_with_mask(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB), pred)

        cv2.imwrite(os.path.join(args.image_root, 'pred', str(i-1).zfill(6) + '.png'), pred)
        cv2.imwrite(os.path.join(args.image_root, 'pred_overlay', str(i-1).zfill(6) + '.png'), pred_overlay)

        im1 = im2


if __name__ == '__main__':
    main()
