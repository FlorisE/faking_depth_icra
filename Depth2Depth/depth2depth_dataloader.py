import tensorflow as tf
import numpy as np
import cv2
import glob
import os
import pickle
import random
import sys

sys.path.append('../')
import helper_functions

BATCH_SIZE = 1
MIN_DEPTH = 450
MAX_DEPTH = 2000
IMG_WIDTH = 1280
IMG_HEIGHT = 720
IMG_WIDTH_NEW = 512
IMG_HEIGHT_NEW = 512

def generator_(paths, height, width, min_depth, max_depth, min_clip=None):
    for path in paths:
        yield helper_functions.parse_depth_bin(path, height, width, min_depth, max_depth, False, min_clip=min_clip)


def ds_mapper(depth):
    flip = tf.random.uniform([], 0, 1)
    start_idx = tf.experimental.numpy.random.randint(0,IMG_WIDTH-IMG_HEIGHT)
    
    depth_flip = tf.cond(flip <= 0.5, lambda: depth, lambda: tf.reverse(depth, [1]))
    
    depth_window = depth_flip[:,start_idx:start_idx+IMG_HEIGHT,:]
    
    return tf.image.resize(depth_window, [IMG_WIDTH_NEW, IMG_WIDTH_NEW], method=tf.image.ResizeMethod.BILINEAR)


def get_transparent_dataset(data_root_train, batch_size=BATCH_SIZE, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None):
    transparent_depth_paths = sorted(glob.glob(data_root_train + 'transparent/depth/*.bin'))
    random.shuffle(transparent_depth_paths)
    print(f"Transparent: {len(transparent_depth_paths)}")
    
    generator = lambda path: generator_(path, IMG_HEIGHT, IMG_WIDTH, min_depth, max_depth, min_clip=min_clip)
    
    transparent_dataset = tf.data.Dataset.from_generator(generator,
                                                    output_signature=(
                                                        tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32)),
                                                    args=[transparent_depth_paths])
    
    return transparent_dataset.map(
        ds_mapper
    ).batch(
        batch_size
    )
    

def get_opaque_dataset(data_root_train, batch_size=BATCH_SIZE, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None):
    opaque_depth_paths = sorted(glob.glob(data_root_train + 'opaque/depth/*.bin'))
    random.shuffle(opaque_depth_paths)
    print(f"Opaque: {len(opaque_depth_paths)}")
    
    generator = lambda path: generator_(path, IMG_HEIGHT, IMG_WIDTH, min_depth, max_depth, min_clip=min_clip)

    opaque_dataset = tf.data.Dataset.from_generator(generator,
                                                output_signature=(
                                                    tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32)),
                                                args=[opaque_depth_paths])
    
    return opaque_dataset.map(
        ds_mapper
    ).batch(
        batch_size
    )
    
def get_train_dataset(data_root_train='../data/unpaired/', min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None):
    if not os.path.exists(data_root_train):
        raise ValueError(f'Dataset not found at {data_root_train}')
    transparent_dataset_rz = get_transparent_dataset(data_root_train, min_depth=min_depth, max_depth=max_depth, min_clip=min_clip)
    opaque_dataset_rz = get_opaque_dataset(data_root_train, min_depth=min_depth, max_depth=max_depth, min_clip=min_clip)
    
    return transparent_dataset_rz, opaque_dataset_rz


def crop_depth(depth):
    m, n, _ = depth.shape
    if m > n:
        margin = int((m - n) / 2)
        return depth[margin:margin+n,:,:]
    elif n > m:
        margin = int((n - m) / 2)
        return depth[:,margin:margin+m,:]
    return depth


def depth_generator(path, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None):
    depth = helper_functions.parse_depth_bin(path, IMG_HEIGHT, IMG_WIDTH, min_depth, max_depth, masks=False, min_clip=min_clip)
    return depth


def resize_depth(depth):
    return helper_functions.resize_depth(depth, IMG_HEIGHT_NEW, IMG_WIDTH_NEW)


def depth_and_img_generator(path_pairs, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None):
    for path_pair in path_pairs:
        depth = depth_generator(path_pair[0], min_depth, max_depth, min_clip=min_clip)
        depth = crop_depth(depth)
        depth = resize_depth(depth)
        color = cv2.resize(helper_functions.normalize(helper_functions.parse_rgb_img(path_pair[1], IMG_HEIGHT, IMG_WIDTH, mono=False, crop=True)), dsize=(IMG_HEIGHT_NEW, IMG_WIDTH_NEW), interpolation=cv2.INTER_NEAREST)
        yield depth, color
        

def multiple_depth_generator(paths, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None):
    for path in paths:
        depth = depth_generator(path, min_depth, max_depth, min_clip=min_clip)
        depth = crop_depth(depth)
        depth = resize_depth(depth)
        yield depth
        
        
def get_transparent_test_valid(data_root_val, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None):
    transparent_depth_matching_files = sorted(glob.glob(data_root_val + '/transparent/depth/*.bin'))
    print(f"Transparent depth test size: {len(transparent_depth_matching_files)}")
    
    generator = lambda paths: multiple_depth_generator(paths, min_depth, max_depth, min_clip=min_clip)
    
    return tf.data.Dataset.from_generator(generator,
                                          args=[transparent_depth_matching_files],
                                          output_signature=(
                                            tf.TensorSpec(shape=(IMG_HEIGHT_NEW, IMG_WIDTH_NEW, 1), dtype=tf.float32)))

    
def get_opaque_test_valid(data_root_val, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None):
    opaque_depth_matching_files = sorted(glob.glob(data_root_val + '/opaque/depth/*.bin'))
    print(f"Opaque depth test size: {len(opaque_depth_matching_files)}")
    
    generator = lambda paths: multiple_depth_generator(paths, min_depth, max_depth, min_clip=min_clip)
    
    return tf.data.Dataset.from_generator(generator,
                                          args=[opaque_depth_matching_files],
                                          output_signature=(
                                            tf.TensorSpec(shape=(IMG_HEIGHT_NEW, IMG_WIDTH_NEW, 1), dtype=tf.float32)))

def get_masks(data_root_val):
    masks_matching_files = sorted(glob.glob(data_root_val + '/masks/*.bin'))
    masks_matching = []
    for path in masks_matching_files:
        masks_matching.append(np.fromfile(path).reshape(512,512))
    return masks_matching


def get_test_valid_dataset(data_root, min_depth, max_depth, min_clip):
    transparent_matching = get_transparent_test_valid(data_root, min_depth, max_depth, min_clip=min_clip)
    opaque_matching = get_opaque_test_valid(data_root, min_depth, max_depth, min_clip=min_clip)
    masks_matching = get_masks(data_root)
    return list(transparent_matching), list(opaque_matching), masks_matching


def get_test_dataset(data_root='../data/test', min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None):
    if not os.path.exists(data_root):
        raise ValueError(f'Dataset not found at {data_root}')
    transparent_matching_list, opaque_matching_list, masks_matching = \
        get_test_valid_dataset(data_root, min_depth, max_depth, min_clip)
    return transparent_matching_list, opaque_matching_list, masks_matching


def get_valid_dataset(data_root='../data/val', min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None):
    if not os.path.exists(data_root):
        raise ValueError(f'Dataset not found at {data_root}')
    transparent_matching_list, opaque_matching_list, masks_matching = \
        get_test_valid_dataset(data_root, min_depth, max_depth, min_clip)
    return transparent_matching_list, opaque_matching_list, masks_matching
