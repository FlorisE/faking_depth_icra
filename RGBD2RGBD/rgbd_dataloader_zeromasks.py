import tensorflow as tf
import numpy as np
import cv2
import glob
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

def generator_with_masks(path_pairs, height, width, min_depth, max_depth):
    for path_pair in path_pairs:
        depth, mask = helper_functions.parse_depth_bin(path_pair[0], height, width, min_depth, max_depth, True)
        yield (depth, mask, helper_functions.normalize(helper_functions.parse_rgb_img(path_pair[1], height, width, False)))

        
def generator(pair):
    return generator_with_masks(pair, IMG_HEIGHT, IMG_WIDTH, MIN_DEPTH, MAX_DEPTH)


def resize_rgb(a):
    return helper_functions.resize_rgb(a, IMG_HEIGHT_NEW, IMG_WIDTH_NEW)


def ds_mapper(depth, mask, color):
    flip = tf.random.uniform([], 0, 1)
    start_idx = tf.experimental.numpy.random.randint(0,IMG_WIDTH-IMG_HEIGHT)
    
    depth_flip = tf.cond(flip <= 0.5, lambda: depth, lambda: tf.reverse(depth, [1]))
    mask_flip = tf.cond(flip <= 0.5, lambda: mask, lambda: tf.reverse(mask, [1]))
    color_flip = tf.cond(flip <= 0.5, lambda: color, lambda: tf.reverse(color, [1]))
    
    depth_window = depth_flip[:,start_idx:start_idx+IMG_HEIGHT,:]
    mask_window = mask_flip[:,start_idx:start_idx+IMG_HEIGHT,:]
    color_window = color_flip[:,start_idx:start_idx+IMG_HEIGHT,:]
    
    return (tf.image.resize(depth_window, [IMG_WIDTH_NEW, IMG_WIDTH_NEW], method=tf.image.ResizeMethod.BILINEAR), 
            tf.image.resize(mask_window, [IMG_WIDTH_NEW, IMG_WIDTH_NEW], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
            resize_rgb(color_window))


def get_transparent_dataset(data_root_train, batch_size=BATCH_SIZE):
    transparent_depth_paths = sorted(glob.glob(data_root_train + 'transparent/depth/*.bin'))
    transparent_color_paths = sorted(glob.glob(data_root_train + 'transparent/color/*.png'))
    transparent_pair_paths = list(zip(transparent_depth_paths, transparent_color_paths))
    random.shuffle(transparent_pair_paths)
    print(f"Transparent: {len(transparent_pair_paths)}")
    
    transparent_dataset = tf.data.Dataset.from_generator(generator,
                                                    output_signature=(
                                                        tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
                                                        tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
                                                        tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32)),
                                                    args=[transparent_pair_paths])
    
    return transparent_dataset.map(
        ds_mapper
    ).batch(
        batch_size
    )
    

def get_opaque_dataset(data_root_train, batch_size=BATCH_SIZE):
    opaque_depth_paths = sorted(glob.glob(data_root_train + 'opaque/depth/*.bin'))
    opaque_color_paths = sorted(glob.glob(data_root_train + 'opaque/color/*.png'))
    opaque_pair_paths = list(zip(opaque_depth_paths, opaque_color_paths))
    random.shuffle(opaque_pair_paths)
    print(f"Opaque: {len(opaque_pair_paths)}")

    opaque_dataset = tf.data.Dataset.from_generator(generator,
                                                output_signature=(
                                                    tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
                                                    tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
                                                    tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32)),
                                                args=[opaque_pair_paths])
    
    return opaque_dataset.map(
        ds_mapper
    ).batch(
        batch_size
    )
    
def get_train_dataset(data_root_train='../data/dishwasher2/unpaired/'):
    transparent_dataset_rz = get_transparent_dataset(data_root_train)
    opaque_dataset_rz = get_opaque_dataset(data_root_train)
    
    return transparent_dataset_rz, opaque_dataset_rz


def crop_depth(depth, mask):
    m, n, _ = depth.shape
    if m > n:
        margin = int((m - n) / 2)
        return depth[margin:margin+n,:,:], mask[margin:margin+n,:,:]
    elif n > m:
        margin = int((n - m) / 2)
        return depth[:,margin:margin+m,:], mask[:,margin:margin+m,:]
    return depth, mask


def depth_generator(path):
    depth, mask = helper_functions.parse_depth_bin(path, IMG_HEIGHT, IMG_WIDTH, MIN_DEPTH, MAX_DEPTH, masks=True)
    return depth, mask


def resize_depth(depth, mask):
    return helper_functions.resize_depth(depth, IMG_HEIGHT_NEW, IMG_WIDTH_NEW), helper_functions.resize_depth(mask, IMG_HEIGHT_NEW, IMG_WIDTH_NEW)


def depth_and_img_generator(path_pairs):
    for path_pair in path_pairs:
        depth, mask = depth_generator(path_pair[0])
        depth, mask = crop_depth(depth, mask)
        depth, mask = resize_depth(depth, mask)
        color = cv2.resize(helper_functions.normalize(helper_functions.parse_rgb_img(path_pair[1], IMG_HEIGHT, IMG_WIDTH, mono=False, crop=True)), dsize=(IMG_HEIGHT_NEW, IMG_WIDTH_NEW), interpolation=cv2.INTER_NEAREST)
        yield depth, mask, color
        

def get_transparent_valid(data_root_val):
    transparent_depth_matching_files = sorted(glob.glob(data_root_val + 'transparent/depth/*.bin'))
    transparent_rgb_matching_files = sorted(glob.glob(data_root_val + '/transparent/color/*.png'))
    transparent_matching_files = list(zip(transparent_depth_matching_files, transparent_rgb_matching_files))
    print(f"Transparent depth test size: {len(transparent_depth_matching_files)}")
    return tf.data.Dataset.from_generator(depth_and_img_generator,
                                          args=[transparent_matching_files],
                                          output_signature=(
                                            tf.TensorSpec(shape=(IMG_HEIGHT_NEW, IMG_WIDTH_NEW, 1), dtype=tf.float32),
                                            tf.TensorSpec(shape=(IMG_HEIGHT_NEW, IMG_WIDTH_NEW, 1), dtype=tf.float32),
                                            tf.TensorSpec(shape=(IMG_HEIGHT_NEW, IMG_WIDTH_NEW, 3), dtype=tf.float32)))

    
def get_opaque_valid(data_root_val):
    opaque_depth_matching_files = sorted(glob.glob(data_root_val + '/opaque/depth/*.bin'))
    opaque_rgb_matching_files = sorted(glob.glob(data_root_val + '/opaque/color/*.png'))
    opaque_matching_files = list(zip(opaque_depth_matching_files, opaque_rgb_matching_files))
    print(f"Opaque depth test size: {len(opaque_depth_matching_files)}")
    return tf.data.Dataset.from_generator(depth_and_img_generator,
                                          args=[opaque_matching_files],
                                          output_signature=(
                                            tf.TensorSpec(shape=(IMG_HEIGHT_NEW, IMG_WIDTH_NEW, 1), dtype=tf.float32),
                                            tf.TensorSpec(shape=(IMG_HEIGHT_NEW, IMG_WIDTH_NEW, 1), dtype=tf.float32),
                                            tf.TensorSpec(shape=(IMG_HEIGHT_NEW, IMG_WIDTH_NEW, 3), dtype=tf.float32)))

def get_masks_valid(data_root_val):
    masks_matching_files = sorted(glob.glob(data_root_val + 'masks/*.bin'))
    masks_matching = []
    for path in masks_matching_files:
        masks_matching.append(np.fromfile(path).reshape(512,512))
    return masks_matching


def get_valid_dataset(data_root_val='../data/dishwasher2/paired/'):
    transparent_matching = get_transparent_valid(data_root_val)
    opaque_matching = get_opaque_valid(data_root_val)
    masks_matching = get_masks_valid(data_root_val)
    return list(transparent_matching), list(opaque_matching), masks_matching


def load_stats(stats_path):
    with open(stats_path, "rb") as stats_file:
        stats_dict = pickle.load(stats_file)
                
    return stats_dict


def write_stats(stats_path,
                gen_t2o_losses,
                gen_o2t_losses,
                cycle_losses_o,
                cycle_losses_t,
                total_cycle_losses,
                identity_losses_o2t,
                identity_losses_t2o,
                total_gen_t2o_losses,
                total_gen_o2t_losses,
                disc_t_losses,
                disc_o_losses,
                t2o_d_rmses,
                t2o_c_rmses,
                t2o_masks_rmses,
                o2t_d_rmses,
                o2t_c_rmses,
                o2t_masks_rmses):
    stats_dict = {'gen_t2o_losses': gen_t2o_losses,
                  'gen_o2t_losses': gen_o2t_losses,
                  'cycle_losses_o': cycle_losses_o,
                  'cycle_losses_t': cycle_losses_t,
                  'total_cycle_losses': total_cycle_losses,
                  'identity_losses_o2t': identity_losses_o2t,
                  'identity_losses_t2o': identity_losses_t2o,
                  'total_gen_t2o_losses': total_gen_t2o_losses,
                  'total_gen_o2t_losses': total_gen_o2t_losses,
                  'disc_t_losses': disc_t_losses,
                  'disc_o_losses': disc_o_losses,
                  't2o_d_rmses': t2o_d_rmses,
                  't2o_c_rmses': t2o_c_rmses,
                  't2o_masks_rmses': t2o_masks_rmses,
                  'o2t_d_rmses': o2t_d_rmses,
                  'o2t_c_rmses': o2t_c_rmses,
                  'o2t_masks_rmses': o2t_masks_rmses}

    with open(stats_path, 'wb') as stats_file:
        pickle.dump(stats_dict, stats_file)