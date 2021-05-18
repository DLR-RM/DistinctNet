"""
Creates self-supervised training images by pasting manipulator and occluded object on random backgrounds.

Expects the following folder structure for manipulator images:
root
|--rgb
|  |--000000.png
|  |--000001.png
|  |--...
|--pred
|  |--000000.png
|  |--000001.png
|  |--...

Expects the following folder structure for occluding objects:
root
|--<urandom-path>
|  |--0.hdf5
|  |--1.hdf5
|  |--...
|--<urandom-path>
|  |--0.hdf5
|  |--1.hdf5
|  |--...
"""

import os
import argparse
from tqdm import tqdm, trange
import numpy as np
import cv2
import glob
import random
from utils.train_utils import load_hdf5


def scale_ims(im, ma, lower_scale_ratio=0, upper_scale_ratio=0):
    # ratio is in correspondence to final image size
    h, w = ma.shape
    min_h = int(lower_scale_ratio * h)
    max_h = int(upper_scale_ratio * h)

    im_h, im_w = im.shape[:2]

    im_ratio = h / w
    new_h = np.random.randint(min_h, max_h)
    new_w = int(new_h / im_h * im_w)

    im = cv2.resize(im, dsize=(new_w, new_h), interpolation=cv2.INTER_LINEAR)
    ma = cv2.resize(ma, dsize=(new_w, new_h), interpolation=cv2.INTER_NEAREST)

    return im, ma


def calc_bbox(mask):
    h, w = mask.shape
    inds = np.where(mask > 0)

    x_min = max(inds[1].min(), 0)
    y_min = max(inds[0].min(), 0)
    x_max = min(inds[1].max(), w)
    y_max = min(inds[0].max(), h)

    return x_min, y_min, x_max - x_min, y_max - y_min


def crop_ims_by_mask(mask, ims, h=960, w=1280):
    inds = np.where(mask > 0)

    x_min = max(inds[1].min(), 0)
    y_min = max(inds[0].min(), 0)
    x_max = min(inds[1].max(), w - 1)
    y_max = min(inds[0].max(), h - 1)

    for i in range(len(ims)):
        if len(ims[i].shape) == 3:
            ims[i] = ims[i][y_min:y_max, x_min:x_max, :]
        else:
            ims[i] = ims[i][y_min:y_max, x_min:x_max]

    return ims


def rotate_ims(ims, angle=None):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    if angle is None:
        angle = np.random.rand() * 360
    rotated_ims = []
    for im in ims:
        height, width = im.shape[:2]  # image shape has 3 dimensions
        image_center = (width/2, height/2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0,0])
        abs_sin = abs(rotation_mat[0,1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w/2 - image_center[0]
        rotation_mat[1, 2] += bound_h/2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(im, rotation_mat, (bound_w, bound_h))
        rotated_ims.append(rotated_mat)

    return rotated_ims


def load_random_manipulator():
    manipulator_id = np.random.randint(len(manipulator_ims))
    manipulator_im = cv2.imread(manipulator_ims[manipulator_id])
    manipulator_im = cv2.resize(manipulator_im, (736, 414), interpolation=cv2.INTER_LINEAR)
    manipulator_ma = cv2.imread(manipulator_mas[manipulator_id], cv2.IMREAD_GRAYSCALE)

    # small processing with erosion
    kernel = np.ones((5, 5), np.uint8)
    manipulator_ma = cv2.erode(manipulator_ma, kernel, iterations=1)

    return manipulator_im, manipulator_ma


def get_shapenet_obj(obj_id=None):
    if obj_id is None:
        while obj_id is None:
            tm = random.choice(occluding_paths)
            if len(glob.glob(tm + '/*.hdf5')) != 0:
                obj_id = tm
    hdf5s = glob.glob(obj_id + '/*.hdf5')
    path = random.choice(hdf5s)

    data = load_hdf5(path, keys=['colors', 'segmap'])
    rgb = data['colors']
    uniques, counts = np.unique(rgb, return_counts=True)
    if len(counts) < 25:
        print('RGB image contains too little color, skipping ...')
        return None, None, None

    rgb = cv2.cvtColor(data['colors'], cv2.COLOR_BGR2RGB)
    return rgb, data['segmap'], obj_id


def load_occluding_object(obj_id=None):
    obj_im, obj_ma, obj_id = get_shapenet_obj(obj_id)
    
    if obj_im is None:
        return None
    lsr = args.lower_scale_ratio_obj
    usr = args.upper_scale_ratio_obj

    obj_im, obj_ma = rotate_ims([obj_im, obj_ma])

    if obj_ma.max() == 0:
        print('Segmentation mask is 0, skipping ...')
        return None
    obj_im, obj_ma = crop_ims_by_mask(mask=obj_ma, ims=[obj_im, obj_ma], h=414, w=736)
    tmp_h, tmp_w = obj_ma.shape[:2]

    if obj_ma.max() == 0 or tmp_h < 25:
        print('Segmentation mask is either 0 or too small, skipping ...')
        return None
    obj_im, obj_ma = scale_ims(im=obj_im, ma=obj_ma, lower_scale_ratio=lsr, upper_scale_ratio=usr)
    kernel = np.ones((5, 5), np.uint8)
    obj_ma = cv2.erode(obj_ma, kernel, iterations=1)
    obj_im[:, :][obj_ma == 0] = [0, 0, 0]
    return obj_im, obj_ma, obj_id


def get_random_distractor_objects(num_objs=5, obj_ids=None, orig_ma=None):
    distractor_ids = []
    if obj_ids is None:
        obj_ids = [None for _ in range(num_objs)]

    fnl_overlay_im = np.zeros((414, 736, 3), dtype=np.uint8)
    fnl_overlay_ma = np.zeros((414, 736), dtype=np.uint8)

    for i in range(num_objs):
        # load data
        rets = None
        while rets is None:
            rets = load_occluding_object(obj_id=obj_ids[i])
        obj_im, obj_ma, obj_id = rets
        distractor_ids.append(obj_id)

        h, w = obj_ma.shape
        size_reduction = np.random.uniform(1., 3.)
        h = int(h / size_reduction)
        w = int(w / size_reduction)
        obj_im = cv2.resize(obj_im, (w, h), interpolation=cv2.INTER_LINEAR)
        obj_ma = cv2.resize(obj_ma, (w, h), interpolation=cv2.INTER_NEAREST)

        # crop
        bbox = calc_bbox(obj_ma)
        x1, y1, x2, y2 = bbox
        x2 += x1
        y2 += y1
        obj_im = obj_im[y1:y2, x1:x2, :]
        obj_ma = obj_ma[y1:y2, x1:x2]

        # paste with space
        im, ma = paste_with_space(obj_im, obj_ma, orig_ma)
        ma[ma != 0] = i + 3
        fnl_overlay_im[ma != 0] = im[ma != 0]
        fnl_overlay_ma[ma != 0] = ma[ma != 0]

    return fnl_overlay_im, fnl_overlay_ma, distractor_ids


def paste_with_space(im, ma, tar_ma, max_overlap=0.75):
    """
    Very simple pasting of a mask outside a bounding box with some max iou overlap.
    """

    max_h, max_w = tar_ma.shape
    bbox = calc_bbox(tar_ma)
    bbox = list(bbox)

    # pad results with overlap ratio
    h, w = ma.shape
    dummy_im = np.zeros((max_h + int(2 * h * max_overlap), max_w + int(2 * w * max_overlap), 3), dtype=np.uint8)
    dummy_ma = np.zeros((max_h + int(2 * h * max_overlap), max_w + int(2 * w * max_overlap)), dtype=np.uint8)
    bbox[0] += int(h * max_overlap)
    bbox[2] += int(h * max_overlap)
    bbox[1] += int(w * max_overlap)
    bbox[3] += int(w * max_overlap)

    # paste somewhere
    x_st, y_st = np.random.randint(0, max_w - w + int(2 * w * max_overlap) - 1), \
                 np.random.randint(0, max_h - h + int(2 * h * max_overlap))
    while not _is_invalid(bbox, (x_st, y_st, x_st + h, x_st + w), h=int(max_h + 2 * h * max_overlap),
                          w=int(max_w + 2 * w * max_overlap)):
        x_st, y_st = np.random.randint(0, max_w - w + int(2 * w * max_overlap) - 1), \
                     np.random.randint(0, max_h - h + int(2 * h * max_overlap))

    dummy_im[y_st:y_st + h, x_st:x_st + w, :] = im
    dummy_ma[y_st:y_st + h, x_st:x_st + w] = ma

    dummy_im = dummy_im[int(h * max_overlap):int(h * max_overlap) + max_h,
               int(w * max_overlap):int(w * max_overlap) + max_w, :]
    dummy_ma = dummy_ma[int(h * max_overlap):int(h * max_overlap) + max_h,
               int(w * max_overlap):int(w * max_overlap) + max_w]

    # cv2.imshow('im', im)
    # cv2.imshow('ma', ma / ma.max())
    # cv2.imshow('dummy im', dummy_im)
    # cv2.imshow('dummy ma', dummy_ma / dummy_ma.max())
    # cv2.waitKey(0)

    return dummy_im, dummy_ma


def _is_invalid(bbox1, bbox2, max_iou_overlap=0.2, h=414, w=736):
    x1, y1, xs1, ys1 = bbox1
    x2, y2, xs2, ys2 = bbox2

    arr1 = np.zeros((h, w), dtype=np.uint8)
    arr1[y1:y1 + ys1, x1:x1 + xs1] = 1

    arr2 = np.zeros_like(arr1)
    arr2[y2:y2 + ys2, x2:x2 + xs2] = 1

    sm = arr1 + arr2

    iou = (sm == 2).sum() / (sm > 0).sum()
    if iou > max_iou_overlap:
        return False

    return True


def generate_first_image():
    # load manipulator and occluding object
    manipulator_im, manipulator_ma = load_random_manipulator()
    
    rets = None
    while rets is None:
        rets = load_occluding_object(obj_id=None)
    obj_im, obj_ma, obj_id = rets

    # crop
    obj_im, obj_ma = crop_ims_by_mask(mask=obj_ma, ims=[obj_im, obj_ma], h=414, w=736)

    # final object image + mask
    fnl_obj_im = np.zeros_like(manipulator_im)
    fnl_obj_ma = np.zeros_like(manipulator_ma)

    # left-most hand part
    x, _, _, _ = calc_bbox(manipulator_ma)
    ys = np.where(manipulator_ma[:, x] != 0)[0]
    x += np.random.randint(0, 50)
    y = ys[int(len(ys) / 2)]

    # subtract half of obj bb to get start pos
    objh, objw = obj_ma.shape
    x -= int(objw / 2)
    x = max(0, x)
    y -= int(objh / 2)
    y = max(0, y)
    xend = x + objw
    yend = y + objh

    # cut obj mask in case it is too long
    if xend >= 736:
        obj_ma = obj_ma[:, :736-x]
        obj_im = obj_im[:, :736-x]
        xend = 736
    if yend >= 414:
        obj_ma = obj_ma[:414-y, :]
        obj_im = obj_im[:414-y, :]
        yend = 414

    # paste
    fnl_obj_im[y:yend, x:xend, :] = obj_im
    fnl_obj_ma[y:yend, x:xend] = obj_ma
    manipulator_im[fnl_obj_ma != 0] = fnl_obj_im[fnl_obj_ma != 0]
    manipulator_ma[manipulator_ma != 0] = 1
    manipulator_ma[fnl_obj_ma != 0] = 2
    #manipulator_im[manipulator_ma == 0] = [0, 0, 0]

    # crop, resize and rotate
    hand_im, hand_ma = crop_ims_by_mask(mask=manipulator_im, ims=[manipulator_im, manipulator_ma], h=414, w=736)
    hand_im, hand_ma = rotate_ims([hand_im, hand_ma])

    curr_h, curr_w = hand_ma.shape[:2]
    scl = np.random.uniform(0.5, 0.7)
    new_h = int(curr_h * scl)
    new_w = int(curr_w * scl)
    hand_ma = cv2.resize(hand_ma, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    hand_im = cv2.resize(hand_im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # random flip
    if np.random.rand() > 0.5:
        hand_im = cv2.flip(hand_im, 1)
        hand_ma = cv2.flip(hand_ma, 1)

    h, w = hand_ma.shape
    h2, w2 = int(h/2), int(w/2)
    if h > 414 or w > 736:
        hand_im = hand_im[:414, :736, :]
        hand_ma = hand_ma[:414, :736]
        h, w = hand_ma.shape

    x_st = np.random.randint(64, 736-64)
    y_st = np.random.randint(36, 414-36)

    fnl_train_im = np.zeros((414+h, 736+w, 3), dtype=np.uint8)
    fnl_train_ma = np.zeros((414+h, 736+w), dtype=np.uint8)

    fnl_train_im[y_st:y_st+h, x_st:x_st+w, :] = hand_im
    fnl_train_ma[y_st:y_st+h, x_st:x_st+w] = hand_ma
    fnl_train_im = fnl_train_im[h2:h2+414, w2:w2+736, :]
    fnl_train_ma = fnl_train_ma[h2:h2+414, w2:w2+736]

    idx = np.random.randint(len(background_ims))
    im = cv2.imread(background_ims[idx])
    im = cv2.resize(im, (736, 414), interpolation=cv2.INTER_LINEAR)
    background_image = im.copy()

    # paste
    im[fnl_train_ma != 0] = fnl_train_im[fnl_train_ma != 0]

    # paste distractor objects, if any
    if args.num_distractor_objects > 0:
        distractor_im, distractor_ma, distractor_ids = get_random_distractor_objects(num_objs=np.random.randint(args.num_distractor_objects + 1), obj_ids=None, orig_ma=fnl_train_ma)

        # overlay
        im[distractor_ma != 0] = distractor_im[distractor_ma != 0]
        fnl_train_ma[distractor_ma != 0] = distractor_ma[distractor_ma != 0]

        return im, fnl_train_ma, background_image, obj_id, distractor_ids

    return im, fnl_train_ma, background_image, obj_id, None


def generate_other_images(bg_im, obj_id, distractor_ids=None):

    # load robot, obj
    manipulator_im, manipulator_ma = load_random_manipulator()

    rets = None
    while rets is None:
        rets = load_occluding_object(obj_id=obj_id)
    obj_im, obj_ma, obj_id = rets

    # crop
    obj_im, obj_ma = crop_ims_by_mask(mask=obj_ma, ims=[obj_im, obj_ma], h=414, w=736)

    # final object image + mask
    fnl_obj_im = np.zeros_like(manipulator_im)
    fnl_obj_ma = np.zeros_like(manipulator_ma)

    # left-most hand part
    x, _, _, _ = calc_bbox(manipulator_ma)
    ys = np.where(manipulator_ma[:, x] != 0)[0]
    x += np.random.randint(0, 50)
    y = ys[int(len(ys) / 2)]

    # subtract half of obj bb to get start pos
    objh, objw = obj_ma.shape
    x -= int(objw / 2)
    x = max(0, x)
    y -= int(objh / 2)
    y = max(0, y)
    xend = x + objw
    yend = y + objh

    # cut obj mask in case it is too long
    if xend >= 736:
        obj_ma = obj_ma[:, :736-x]
        obj_im = obj_im[:, :736-x]
        xend = 736
    if yend >= 414:
        obj_ma = obj_ma[:414-y, :]
        obj_im = obj_im[:414-y, :]
        yend = 414

    # paste
    fnl_obj_im[y:yend, x:xend, :] = obj_im
    fnl_obj_ma[y:yend, x:xend] = obj_ma
    manipulator_im[fnl_obj_ma != 0] = fnl_obj_im[fnl_obj_ma != 0]
    manipulator_ma[manipulator_ma != 0] = 1
    manipulator_ma[fnl_obj_ma != 0] = 2

    # crop, resize and rotate
    hand_im, hand_ma = crop_ims_by_mask(mask=manipulator_ma, ims=[manipulator_im, manipulator_ma], h=414, w=736)
    hand_im, hand_ma = rotate_ims([hand_im, hand_ma])

    curr_h, curr_w = hand_ma.shape[:2]
    scl = np.random.uniform(0.5, 0.7)
    new_h = int(curr_h * scl)
    new_w = int(curr_w * scl)
    hand_ma = cv2.resize(hand_ma, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    hand_im = cv2.resize(hand_im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # random flip
    if np.random.rand() > 0.5:
        hand_im = cv2.flip(hand_im, 1)
        hand_ma = cv2.flip(hand_ma, 1)

    h, w = hand_ma.shape
    h2, w2 = int(h / 2), int(w / 2)
    if h > 414 or w > 736:
        hand_im = hand_im[:414, :736, :]
        hand_ma = hand_ma[:414, :736]
        h, w = hand_ma.shape

    x_st = np.random.randint(64, 736-64)
    y_st = np.random.randint(36, 414-36)

    fnl_train_im = np.zeros((414 + h, 736 + w, 3), dtype=np.uint8)
    fnl_train_ma = np.zeros((414 + h, 736 + w), dtype=np.uint8)

    fnl_train_im[y_st:y_st + h, x_st:x_st + w, :] = hand_im
    fnl_train_ma[y_st:y_st + h, x_st:x_st + w] = hand_ma
    fnl_train_im = fnl_train_im[h2:h2 + 414, w2:w2 + 736, :]
    fnl_train_ma = fnl_train_ma[h2:h2 + 414, w2:w2 + 736]

    # paste
    im = bg_im.copy()
    im[fnl_train_ma != 0] = fnl_train_im[fnl_train_ma != 0]

    # paste distractor objects, if any
    if args.num_distractor_objects > 0:
        distractor_im, distractor_ma, distractor_ids = get_random_distractor_objects(
            num_objs=len(distractor_ids), obj_ids=distractor_ids, orig_ma=fnl_train_ma)

        # overlay
        im[distractor_ma != 0] = distractor_im[distractor_ma != 0]
        fnl_train_ma[distractor_ma != 0] = distractor_ma[distractor_ma != 0]

    return im, fnl_train_ma


def check_manipulator_files(im_paths, ma_paths):
    print(f"Checking manipulator files ...")
    checked_im_paths, checked_ma_paths = [], []
    for i in trange(len(im_paths)):
        ma = cv2.imread(ma_paths[i], cv2.IMREAD_GRAYSCALE)
        kernel = np.ones((5, 5), np.uint8)
        ma = cv2.erode(ma, kernel, iterations=1)
        if ma.max() == 0:
            continue
        else:
            checked_im_paths.append(im_paths[i])
            checked_ma_paths.append(ma_paths[i])

    return checked_im_paths, checked_ma_paths


def generate(ctr):
    np.random.seed(ctr)

    # dummy loop to try 10 times
    for i in range(10):
        try:
            # generate first image
            im, ma, bg_im, obj_id, distractor_ids = generate_first_image()

            # save image, mask and gripper mask
            cv2.imwrite(os.path.join(args.output_path, f"rgb1/{str(ctr).zfill(6)}.png"), im)
            cv2.imwrite(os.path.join(args.output_path, f"gt1/{str(ctr).zfill(6)}.png"), ma)

            # generate sequence images (default just one second image)
            for j in range(args.sequence_length - 1):
                rets = generate_other_images(bg_im=bg_im, obj_id=obj_id, distractor_ids=distractor_ids)
                im, ma = rets
                cv2.imwrite(os.path.join(args.output_path, f"rgb{j+2}/{str(ctr).zfill(6)}.png"), im)
                cv2.imwrite(os.path.join(args.output_path, f"gt{j+2}/{str(ctr).zfill(6)}.png"), ma)
            break
        except Exception as e:
            print(i, e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--min-obj-size', type=int, default=1000)
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--lower-scale-ratio-obj', type=float, default=0.5)
    parser.add_argument('--upper-scale-ratio-obj', type=float, default=1.)
    parser.add_argument('--num-ims', default=100, type=int)
    parser.add_argument('--background-root', type=str, required=True)
    parser.add_argument('--background-suffix', type=str, default='png')
    parser.add_argument('--manipulator-root', type=str, required=True)
    parser.add_argument('--occluding-root', type=str, required=True)
    parser.add_argument('--num-distractor-objects', type=int, default=0)
    parser.add_argument('--sequence-length', type=int, default=2)
    parser.add_argument('--pools', type=int, default=8)
    args = parser.parse_args()

    # load background
    background_ims = glob.glob(os.path.join(args.background_root, f'*.{args.background_suffix}'))
    assert len(background_ims) != 0, f"No background images found in {args.background_root}"
    print(f"Found {len(background_ims)} background images in {args.background_root}")

    # load manipulator images
    manipulator_ims = sorted(glob.glob(os.path.join(args.manipulator_root, 'rgb/*.png')))
    manipulator_mas = sorted(glob.glob(os.path.join(args.manipulator_root, 'pred/*.png')))
    if len(manipulator_ims) == len(manipulator_mas) + 1:
        manipulator_ims = manipulator_ims[:-1]
        print(f"Assuming that the number of rgb ims is one larger than the number of annotations, so removing last rgb image")
    assert len(manipulator_ims) != 0, f"No manipulator images found in {args.manipulator_root}"
    assert len(manipulator_ims) == len(manipulator_mas), f"Unequal length of manipulator images and masks: {len(manipulator_ims)} != {len(manipulator_mas)}"
    # check manipulator images - skip all that have 0 prediction
    manipulator_ims, manipulator_mas = check_manipulator_files(manipulator_ims, manipulator_mas)
    print(f"Found {len(manipulator_ims)} valid ims and masks in {args.manipulator_root}")

    # load occluding images
    occluding_paths = glob.glob(os.path.join(args.occluding_root, '*'))
    assert len(occluding_paths) != 0, f"No occluding paths found in {args.occluding_root}"
    print(f"Found {len(occluding_paths)} different occluding objects in {args.occluding_root}")

    # create output folder
    os.makedirs(args.output_path, exist_ok=True)
    for i in range(args.sequence_length):
        os.makedirs(os.path.join(args.output_path, f'rgb{i+1}'), exist_ok=True)
        os.makedirs(os.path.join(args.output_path, f'gt{i+1}'), exist_ok=True)

    #for i in trange(args.num_ims):
    #    generate(i)
    #exit(0)

    counters = list(range(args.num_ims))
    from multiprocessing import Pool
    print('Running with {} pools'.format(args.pools))
    pool = Pool(args.pools)
    for _ in tqdm(pool.imap(generate, counters), total=len(counters)):
        pass
    
    pool.close()
    pool.join()

