"""
This script implements a small demo which segments moving objects in front of a camera.
Make sure you have downloaded the pretrained model before, or you have trained a model yourself.
"""

import cv2
import argparse

from predictor import Predictor
from utils.pred_utils import overlay_im_with_mask


def main():
    print("Running demo on webcam\n")
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dict', type=str, required=True)
    args = parser.parse_args()

    predictor = Predictor(state_dict_path=args.state_dict)

    cap = cv2.VideoCapture(0)

    _, im1 = cap.read()

    while True:
        _, im2 = cap.read()

        pred = predictor.predict(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB), cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))

        cv2.imshow('pred', overlay_im_with_mask(cv2.resize(im1, (736, 414), interpolation=cv2.INTER_LINEAR), pred))
        cv2.waitKey(1)

        im1 = im2


if __name__ == '__main__':
    main()
