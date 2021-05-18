"""
Recording script that lets you record images from a webcam.
Usually better than using `demo.py`, as its frame rate might be suboptimal.
"""

import os
import argparse
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-root', default='./recorded_images', type=str)
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)
    ctr = 0

    os.makedirs(args.image_root, exist_ok=True)
    os.makedirs(os.path.join(args.image_root, 'rgb'), exist_ok=True)

    print(f"Trying to record images and save to {args.image_root}")

    while True:
        _, im = cap.read()
        cv2.imshow('im', im)
        cv2.waitKey(1)
        cv2.imwrite(os.path.join(args.image_root, 'rgb', str(ctr).zfill(6) + '.png'), cv2.resize(im, (736, 414), interpolation=cv2.INTER_LINEAR))
        ctr += 1


if __name__ == '__main__':
    main()
