import tensorflow as tf
import numpy as np
import cv2
import glob
import pickle
import random
import sys

from skimage.transform import resize
from skimage import img_as_bool

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

sys.path.append('../')
import helper_functions

BATCH_SIZE = 1000
MIN_DEPTH = 450
MAX_DEPTH = 2000
IMG_WIDTH = 1280
IMG_HEIGHT = 720
IMG_WIDTH_NEW = 512
IMG_HEIGHT_NEW = 512

def generator_(path_pairs, height, width, min_depth, max_depth, min_clip=None):
    for path_pair in path_pairs:
        depth = helper_functions.parse_depth_bin(path_pair[0], height, width, min_depth, max_depth, False, min_clip=min_clip)
        yield (depth, helper_functions.normalize(helper_functions.parse_rgb_img(path_pair[1], height, width, False)))

def resize_rgb(a, width=IMG_WIDTH_NEW, height=IMG_HEIGHT_NEW):
    return helper_functions.resize_rgb(a, height, width)

def ds_mapper_body(depth_path, color_path, height, width, min_depth, max_depth, min_clip=None):
    depth = helper_functions.parse_depth_bin(depth_path.numpy(), height, width, min_depth, max_depth, False, min_clip=min_clip).astype(np.float32)
    color = helper_functions.normalize(helper_functions.parse_rgb_img(color_path.numpy(), height, width, False)).astype(np.float32)
    return depth, color

def ds_mapper(path_pair, height, width, min_depth, max_depth, min_clip=None):
    depth, color = tf.py_function(ds_mapper_body, [path_pair[0], path_pair[1], height, width, min_depth, max_depth, min_clip], [tf.float32, tf.float32])
    depth.set_shape((IMG_HEIGHT, IMG_WIDTH, 1))
    color.set_shape((IMG_HEIGHT, IMG_WIDTH, 3))
    return depth, color

def randomizer(depth, color, resize_width, resize_height):
    flip = tf.random.uniform([], 0, 1)
    start_idx = tf.experimental.numpy.random.randint(0,IMG_WIDTH-IMG_HEIGHT)
    
    depth_flip = tf.cond(flip <= 0.5, lambda: depth, lambda: tf.reverse(depth, [1]))
    color_flip = tf.cond(flip <= 0.5, lambda: color, lambda: tf.reverse(color, [1]))
    
    depth_window = depth_flip[:,start_idx:start_idx+IMG_HEIGHT,:]
    color_window = color_flip[:,start_idx:start_idx+IMG_HEIGHT,:]
    
    return (tf.image.resize(depth_window, [resize_width, resize_height], method=tf.image.ResizeMethod.BILINEAR), 
            resize_rgb(color_window, resize_width, resize_height))

def ds_mapper_old(depth, color):
    flip = tf.random.uniform([], 0, 1)
    start_idx = tf.experimental.numpy.random.randint(0,IMG_WIDTH-IMG_HEIGHT)
    
    depth_flip = tf.cond(flip <= 0.5, lambda: depth, lambda: tf.reverse(depth, [1]))
    color_flip = tf.cond(flip <= 0.5, lambda: color, lambda: tf.reverse(color, [1]))
    
    depth_window = depth_flip[:,start_idx:start_idx+IMG_HEIGHT,:]
    color_window = color_flip[:,start_idx:start_idx+IMG_HEIGHT,:]
    
    return (tf.image.resize(depth_window, [IMG_WIDTH_NEW, IMG_WIDTH_NEW], method=tf.image.ResizeMethod.BILINEAR), 
            resize_rgb(color_window))


def get_transparent_dataset(data_root_train, batch_size=BATCH_SIZE, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None, resize_width=512, resize_height=512, take=1):
    transparent_depth_paths = sorted(glob.glob(data_root_train + 'transparent/depth/*.bin'))
    transparent_color_paths = sorted(glob.glob(data_root_train + 'transparent/color/*.png'))
    transparent_pair_paths = list(zip(transparent_depth_paths, transparent_color_paths))
    print(f"Transparent: {len(transparent_pair_paths)}")
    
    dataset = tf.data.Dataset.from_tensor_slices(transparent_pair_paths)
    
    ds_mapper_wrapper = lambda path_pair: ds_mapper(path_pair, IMG_HEIGHT, IMG_WIDTH, min_depth, max_depth, min_clip=min_clip)
    randomizer_wrapper = lambda depth, color: randomizer(depth, color, resize_width, resize_height)
    
    return dataset.shuffle(len(transparent_pair_paths)).take(batch_size).map(ds_mapper_wrapper).map(randomizer_wrapper).batch(take)


def get_transparent_dataset_old(data_root_train, batch_size=BATCH_SIZE, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None):
    transparent_depth_paths = sorted(glob.glob(data_root_train + 'transparent/depth/*.bin'))
    transparent_color_paths = sorted(glob.glob(data_root_train + 'transparent/color/*.png'))
    transparent_pair_paths = list(zip(transparent_depth_paths, transparent_color_paths))
    random.shuffle(transparent_pair_paths)
    print(f"Transparent: {len(transparent_pair_paths)}")
    
    generator = lambda pair: generator_(pair, IMG_HEIGHT, IMG_WIDTH, min_depth, max_depth, min_clip=min_clip)
    
    transparent_dataset = tf.data.Dataset.from_generator(generator,
                                                    output_signature=(
                                                        tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
                                                        tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32)),
                                                    args=[transparent_pair_paths])
    
    return transparent_dataset.map(
        ds_mapper
    ).batch(
        batch_size
    )
    

def get_opaque_dataset(data_root_train, batch_size=BATCH_SIZE, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None, resize_width=512, resize_height=512, take=1):
    opaque_depth_paths = sorted(glob.glob(data_root_train + 'opaque/depth/*.bin'))
    opaque_color_paths = sorted(glob.glob(data_root_train + 'opaque/color/*.png'))
    opaque_pair_paths = list(zip(opaque_depth_paths, opaque_color_paths))
    print(f"Opaque: {len(opaque_pair_paths)}")
    
    dataset = tf.data.Dataset.from_tensor_slices(opaque_pair_paths)
    
    ds_mapper_wrapper = lambda path_pair: ds_mapper(path_pair, IMG_HEIGHT, IMG_WIDTH, min_depth, max_depth, min_clip=min_clip)
    randomizer_wrapper = lambda depth, color: randomizer(depth, color, resize_width, resize_height)
    
    return dataset.shuffle(len(opaque_pair_paths)).take(batch_size).map(ds_mapper_wrapper).map(randomizer_wrapper).batch(take)


def get_opaque_dataset_old(data_root_train, batch_size=BATCH_SIZE, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None):
    opaque_depth_paths = sorted(glob.glob(data_root_train + 'opaque/depth/*.bin'))
    opaque_color_paths = sorted(glob.glob(data_root_train + 'opaque/color/*.png'))
    opaque_pair_paths = list(zip(opaque_depth_paths, opaque_color_paths))
    random.shuffle(opaque_pair_paths)
    print(f"Opaque: {len(opaque_pair_paths)}")
    
    generator = lambda pair: generator_(pair, IMG_HEIGHT, IMG_WIDTH, min_depth, max_depth, min_clip=min_clip)

    opaque_dataset = tf.data.Dataset.from_generator(generator,
                                                output_signature=(
                                                    tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
                                                    tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32)),
                                                args=[opaque_pair_paths])
    
    return opaque_dataset.map(
        ds_mapper
    ).batch(
        batch_size
    )
    
def get_train_dataset(data_root_train='../data/dishwasher2/unpaired/', min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None, resize_width=512, resize_height=512, take=1):
    transparent_dataset_rz = get_transparent_dataset(data_root_train, min_depth=min_depth, max_depth=max_depth, min_clip=min_clip, resize_width=resize_width, resize_height=resize_height, take=take)
    opaque_dataset_rz = get_opaque_dataset(data_root_train, min_depth=min_depth, max_depth=max_depth, min_clip=min_clip, resize_width=resize_width, resize_height=resize_height, take=take)
    
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


def resize_depth(depth, resize_width=512, resize_height=512):
    return helper_functions.resize_depth(depth, resize_height, resize_width)


def depth_and_img_generator(path_pairs, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None, resize_width=512, resize_height=512):
    for path_pair in path_pairs:
        depth = depth_generator(path_pair[0], min_depth, max_depth, min_clip=min_clip)
        depth = crop_depth(depth)
        depth = resize_depth(depth, resize_width, resize_height)
        color = cv2.resize(helper_functions.normalize(helper_functions.parse_rgb_img(path_pair[1], IMG_HEIGHT, IMG_WIDTH, mono=False, crop=True)), dsize=(resize_height, resize_width), interpolation=cv2.INTER_NEAREST)
        yield depth, color
        

def get_transparent_test_valid(data_root, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None, resize_width=512, resize_height=512):
    transparent_depth_matching_files = sorted(glob.glob(data_root + '/transparent/depth/*.bin'))
    transparent_rgb_matching_files = sorted(glob.glob(data_root + '/transparent/color/*.png'))
    transparent_matching_files = list(zip(transparent_depth_matching_files, transparent_rgb_matching_files))
    print(f"Transparent depth test size: {len(transparent_depth_matching_files)}")
    
    generator = lambda path_pairs: depth_and_img_generator(path_pairs, min_depth, max_depth, min_clip=min_clip, resize_width=resize_width, resize_height=resize_height)
    
    return tf.data.Dataset.from_generator(generator,
                                          args=[transparent_matching_files],
                                          output_signature=(
                                            tf.TensorSpec(shape=(resize_height, resize_width, 1), dtype=tf.float32),
                                            tf.TensorSpec(shape=(resize_height, resize_width, 3), dtype=tf.float32)))

    
def get_opaque_test_valid(data_root, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None, resize_width=512, resize_height=512):
    opaque_depth_matching_files = sorted(glob.glob(data_root + '/opaque/depth/*.bin'))
    opaque_rgb_matching_files = sorted(glob.glob(data_root + '/opaque/color/*.png'))
    opaque_matching_files = list(zip(opaque_depth_matching_files, opaque_rgb_matching_files))
    print(f"Opaque depth test size: {len(opaque_depth_matching_files)}")
    
    generator = lambda path_pairs: depth_and_img_generator(path_pairs, min_depth, max_depth, min_clip=min_clip, resize_width=resize_width, resize_height=resize_height)
    
    return tf.data.Dataset.from_generator(generator,
                                          args=[opaque_matching_files],
                                          output_signature=(
                                            tf.TensorSpec(shape=(resize_height, resize_width, 1), dtype=tf.float32),
                                            tf.TensorSpec(shape=(resize_height, resize_width, 3), dtype=tf.float32)))

def get_masks(data_root):
    masks_matching_files = sorted(glob.glob(data_root + '/masks/*.bin'))
    masks_matching = []
    for path in masks_matching_files:
        masks_matching.append(np.fromfile(path).reshape(512, 512))
    return masks_matching

def get_test_valid_dataset(data_root, min_depth, max_depth, min_clip, resize_width=512, resize_height=512):
    transparent_matching = get_transparent_test_valid(data_root, min_depth, max_depth, min_clip=min_clip, resize_width=resize_width, resize_height=resize_height)
    opaque_matching = get_opaque_test_valid(data_root, min_depth, max_depth, min_clip=min_clip, resize_width=resize_width, resize_height=resize_height)
    masks_matching = get_masks(data_root)
    if resize_width != 512 or resize_height != 512:
        masks_matching = [img_as_bool(resize(mask, (resize_width, resize_height))) for mask in masks_matching]
    return list(transparent_matching), list(opaque_matching), masks_matching


def get_test_dataset(data_root='../data/dishwasher2/test', min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None, resize_width=512, resize_height=512):
    transparent_matching_list, opaque_matching_list, masks_matching = \
        get_test_valid_dataset(data_root, min_depth, max_depth, min_clip, resize_width=resize_width, resize_height=resize_height)
    return transparent_matching_list, opaque_matching_list, masks_matching


def get_valid_dataset(data_root='../data/dishwasher2/val', min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None, resize_width=512, resize_height=512):
    transparent_matching_list, opaque_matching_list, masks_matching = \
        get_test_valid_dataset(data_root, min_depth, max_depth, min_clip, resize_width=resize_width, resize_height=resize_height)
    return transparent_matching_list, opaque_matching_list, masks_matching


def get_novel_dataset(data_root='../data/dishwasher2/novel', min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_clip=None, resize_width=512, resize_height=512):
    transparent_matching_list, opaque_matching_list, masks_matching = \
        get_test_valid_dataset(data_root, min_depth, max_depth, min_clip, resize_width=resize_width, resize_height=resize_height)
    return transparent_matching_list, opaque_matching_list, masks_matching