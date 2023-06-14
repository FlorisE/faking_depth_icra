import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2

from PIL import Image

#def center_and_scale_depth_bin(mat, max_depth):
def center_and_scale_depth_bin(mat, min_depth, max_depth, min_clip=None):
    #return (mat / (max_depth/2)) - 1.0
    if min_clip is None:
        min_clip = min_depth
    mat = mat.clip(min_clip, max_depth)
    return 2 * ((mat - min_depth) / (max_depth - min_depth)) - 1.0

#def decenter_and_descale_depth_bin(mat, fixed_max=MAX_DEPTH):
#def decenter_and_descale_depth_bin(mat, fixed_max):
def decenter_and_descale_depth_bin(mat, min_depth, max_depth):
    #return (mat + 1.0) * (fixed_max/2)
    return (max_depth - min_depth) * ((mat + 1.0) / 2.0) + min_depth

def plot_depth(img, fixed_min, fixed_max, colorbar=False, save_as=None):
    plt.imshow(img, vmin=fixed_min, vmax=fixed_max)
    plt.axis('off')
    if colorbar: plt.colorbar()
    if save_as is not None:
        plt.savefig(save_as)
    
#def plot_depth_bin(img, fixed_min=MIN_DEPTH, fixed_max=MAX_DEPTH, colorbar=False):
def plot_depth_bin(img, fixed_min, fixed_max, vis_min, vis_max, colorbar=False, save_as=None):
    return plot_depth(decenter_and_descale_depth_bin(img, fixed_min, fixed_max), vis_min, vis_max, colorbar, save_as)
    #return plot_depth(decenter_and_descale_depth_bin(img, fixed_max), vis_min, vis_max, colorbar, save_as)

def parse_depth_bin(filename, height, width, min_depth, max_depth, masks=False, min_clip=None):
    with open(filename, "rb") as f:
        a = f.read()
        mat = np.ndarray(shape=(height, width, 1), dtype='uint16', buffer=a)
        
        mask = np.ndarray(shape=(height, width, 1), dtype='int16')
        mask[mat == 0] = 1.0
        mask[mat != 0] = -1.0
        
        #mat = mat.clip(min_depth, max_depth)
        #mat = center_and_scale_depth_bin(mat, max_depth)
        
        mat = center_and_scale_depth_bin(mat, min_depth, max_depth, min_clip)
        if masks: return mat, mask
        else: return mat
    
def parse_depth_img(filename, height, width):
    image = Image.open(filename)
    image = image.resize((height, width))
    return np.array(image)

def parse_rgb_img(filename, height, width, mono=False, crop=False):
    image = Image.open(filename)
    if crop and height > width:
        top_margin = (height - width) / 2
        image = image.crop((0, top_margin, width, top_margin + width))
    if crop and width > height:
        side_margin = (width - height) / 2
        image = image.crop((side_margin, 0, side_margin + height, height))
    image = image.resize((width, height))
    if mono:
        image = image.convert('L')
    return np.array(image)

def depth_generator(paths, height, width, min_depth, max_depth):
    for path in paths:
        depth = parse_depth_bin(path, height, width, min_depth, max_depth, masks=False)
        yield depth

def generator(path_pairs, height, width, min_depth, max_depth, mono, masks=False):
    for path_pair in path_pairs:
        if masks:
            depth, mask = parse_depth_bin(path_pair[0], height, width, min_depth, max_depth, masks)
            yield (depth, mask, normalize(parse_rgb_img(path_pair[1], height, width, mono)))
        else:
            depth = parse_depth_bin(path_pair[0], height, width, min_depth, max_depth, masks)
            yield (depth, normalize(parse_rgb_img(path_pair[1], height, width, mono)))
        
        
def normalize(img):
    return img / 127.5 - 1

def denormalize(img):
    return img + 1.0 * 127.5

def get_rgb(path, height, width, mono):
    out = normalize(parse_rgb_img(path, height, width, mono))
    if mono:
        return np.expand_dims(out, -1)
    else:
        return out

def rgb_generator(paths, height, width, mono):
    for path in paths:
        yield get_rgb(path, height, width, mono)
        
def random_crop(image_depth, start, width, height):
    dx = np.random.randint(0, start)
    dy = np.random.randint(0, start)
    image_depth = image_depth[dy:dy+height-start, dx:dx+width-start]

    return image_depth

def random_jitter(image_depth, image_color):
    #image_depth = tf.image.resize(image_depth, [IMG_HEIGHT+30, IMG_WIDTH+30],
    #                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #image_color = tf.image.resize(image_color, [IMG_HEIGHT+30, IMG_WIDTH+30],
    #                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    #image_depth, image_color = random_crop((image_depth, image_color), 30, IMG_WIDTH_NEW, IMG_HEIGHT_NEW)

    #flip1 = np.random.randint(0,2)
    #if flip1:
     #   image_depth = tf.image.flip_left_right(image_depth)
     #   image_color = tf.image.flip_left_right(image_color)

    # flip2 = np.random.randint(0,2)
    # if flip2:
    #     image_depth = tf.image.flip_up_down(image_depth)
    #     image_color = tf.image.flip_up_down(image_color)

    return (image_depth, image_color)

def random_flip(image_depth):
    coinflip = np.random.randint(0,2) 
    if coinflip:
        return tf.image.flip_left_right(image_depth)
    else:
        return image_depth
    
def preprocess_image_train(image_depth, image_color):
    image_pair = (random_flip(image_depth), image_color)
    #image_pair = random_jitter(image_depth, image_color)
    image_pair = normalize(image_pair)
    return image_pair

def generate_images(generator_model, discriminator_model, test_input, epoch=0, name=None):
    prediction = generator_model(test_input, training=False)

    plt.figure(figsize=(12, 12))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(resize_for_viz(np.array(display_list[i] * 0.5 + 0.5)))
        plt.axis('off')
    if not name:
        plt.savefig("image_at_epoch_{:04d}.png".format(epoch))
    else:
        plt.savefig(name)
    plt.show()
    
def center_and_scale_cleargrasp(img, scale_factor):
    return (img / scale_factor) * 2 - 1.0

def decenter_and_descale_cleargrasp(img, descale_factor):
    return ((img + 1.0) / 2) * descale_factor

def get_max(from_list):
    curr_max = 0
    for item in from_list:
        consider_max = np.max(item)
        if consider_max > curr_max:
            curr_max = consider_max
    return curr_max

def resize_for_viz(img):
    return cv2.resize(img, dsize=(256, 144), interpolation=cv2.INTER_NEAREST)

# def visualize_cleargrasp_data(rgb,
#                               transparent_depth,
#                               transparent_label,
#                               opaque_depth,
#                               opaque_label,
#                               generated,
#                               difference,
#                               rmse_difference,
#                               mae_difference,
#                               output_path=None,                                 ***
#                               fixed_min_cg_transparent=MIN_DEPTH,
#                               fixed_max_cg_transparent=MAX_DEPTH,
#                               fixed_min_generated=MIN_DEPTH,
#                               fixed_max_generated=MAX_DEPTH,
#                               fixed_min_cg_opaque=MIN_DEPTH,
#                               fixed_max_cg_opaque=MAX_DEPTH):
def visualize_cleargrasp_data(rgb,
                              opaque_depth,
                              opaque_label,
                              generated,
                              difference,
                              rmse_baseline,
                              mae_baseline,
                              rmse_difference,
                              mae_difference,
                              fixed_min_generated,
                              fixed_max_generated,
                              fixed_min_cg_opaque,
                              fixed_max_cg_opaque,
                              output_path=None):
    width = 3
    height = 3
    plt.figure(figsize=(15, 15))
    plt.subplot(width, height, 1)
    plt.imshow(resize_for_viz(rgb))
    plt.axis('off')
    plt.subplot(width, height, 2)
    plt.title("FakingDepth (generated)")
    plot_depth(resize_for_viz(generated), fixed_min_generated, fixed_max_generated, True)
    plt.subplot(width, height, 3)
    plt.title(opaque_label)
    plot_depth(resize_for_viz(opaque_depth), fixed_min_cg_opaque, fixed_max_cg_opaque, True)
    plt.subplot(width, height, (4, 9))
    plt.title("Ours (generated) vs opaque")
    plt.imshow(resize_for_viz(np.abs(difference)), cmap='RdBu_r', vmin=0, vmax=np.max(difference))
    plt.axis('off')
    plt.colorbar()
    plt.figtext(0, 0.70, f"Baseline RMSE: {rmse_baseline}")
    plt.figtext(0, 0.68, f"Ours RMSE: {rmse_difference}")
    plt.figtext(0, 0.66, f"Baseline MAE: {mae_baseline}")
    plt.figtext(0, 0.64, f"Ours MAE: {mae_difference}")
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
        
def visualize_cleargrasp_data_depth2depth(rgb,
                              transparent_depth,
                              transparent_label,
                              opaque_depth,
                              opaque_label,
                              generated,
                              difference,
                              rmse_baseline,
                              mae_baseline,
                              rmse_difference,
                              mae_difference,
                              fixed_min_cg_transparent,
                              fixed_max_cg_transparent,
                              fixed_min_generated,
                              fixed_max_generated,
                              fixed_min_cg_opaque,
                              fixed_max_cg_opaque,
                              output_path=None):
    width = 4
    height = 3
    plt.figure(figsize=(15, 15))
    plt.subplot(width, height, 2)
    plt.imshow(resize_for_viz(rgb))
    plt.axis('off')
    plt.subplot(width, height, 4)
    plt.title(transparent_label)
    plot_depth(resize_for_viz(transparent_depth), fixed_min_cg_transparent, fixed_max_cg_transparent, True)
    plt.subplot(width, height, 5)
    plt.title("FakingDepth (generated)")
    plot_depth(resize_for_viz(generated), fixed_min_generated, fixed_max_generated, True)
    plt.subplot(width, height, 6)
    plt.title(opaque_label)
    plot_depth(resize_for_viz(opaque_depth), fixed_min_cg_opaque, fixed_max_cg_opaque, True)
    plt.subplot(width, height, (7, 12))
    plt.title("Ours (generated) vs opaque")
    plt.imshow(resize_for_viz(np.abs(difference)), cmap='RdBu_r', vmin=0, vmax=np.max(difference))
    plt.axis('off')
    plt.colorbar()
    plt.figtext(0, 0.80, f"Baseline RMSE: {rmse_baseline}")
    plt.figtext(0, 0.78, f"Ours RMSE: {rmse_difference}")
    plt.figtext(0, 0.76, f"Baseline MAE: {mae_baseline}")
    plt.figtext(0, 0.74, f"Ours MAE: {mae_difference}")
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    
def evaluate_single(rgb,
                    transparent_depth,
                    transparent_depth_max,
                    transparent_label,
                    opaque_depth,
                    opaque_depth_max,
                    opaque_label,
                    mask,
                    generated,
                    output_path="",
                    vis=False):
    masked_idx = mask==0
    transparent_depth = decenter_and_descale_cleargrasp(transparent_depth, transparent_depth_max)
    transparent_depth_nomask = transparent_depth.copy()
    transparent_depth[masked_idx]=0
    generated = decenter_and_descale_cleargrasp(generated, transparent_depth_max)
    generated_nomask = generated.copy()
    generated[masked_idx]=0
    opaque_depth = decenter_and_descale_cleargrasp(opaque_depth, opaque_depth_max)
    opaque_depth_nomask = opaque_depth.copy()
    opaque_depth[masked_idx]=0
    difference = np.squeeze(generated) - opaque_depth
    #difference[np.abs(difference)>0.2] = 0
    opaque_to_transparent_difference = opaque_depth - transparent_depth
    rmse_baseline = np.sqrt(
        np.mean(
            np.square(
                opaque_to_transparent_difference[np.abs(opaque_to_transparent_difference)>0])))
    mae_baseline = np.mean(np.abs(opaque_to_transparent_difference[np.abs(opaque_to_transparent_difference)>0]))
    rmse_difference = np.sqrt(np.mean(np.square(difference[np.abs(difference)>0])))
    mae_difference = np.mean(np.abs(difference[np.abs(difference)>0]))

    if vis:
        visualize_cleargrasp_data(rgb,
                                  transparent_depth_nomask,
                                  transparent_label,
                                  opaque_depth_nomask,
                                  opaque_label,
                                  generated_nomask,
                                  difference,
                                  rmse_baseline,
                                  mae_baseline,
                                  rmse_difference,
                                  mae_difference,
                                  output_path,
                                  fixed_min_cg_transparent=0,
                                  fixed_max_cg_transparent=transparent_depth_max,
                                  fixed_min_generated=0,
                                  fixed_max_generated=transparent_depth_max,
                                  fixed_min_cg_opaque=0,
                                  fixed_max_cg_opaque=opaque_depth_max)
    return rmse_baseline, mae_baseline, rmse_difference, mae_difference
    
def evaluate_single_depth2depth(rgb,
                    transparent_depth,
                    transparent_depth_max,
                    transparent_label,
                    opaque_depth,
                    opaque_depth_max,
                    opaque_label,
                    mask,
                    generated,
                    output_path="",
                    vis=False):
    masked_idx = mask==0
    transparent_depth = decenter_and_descale_cleargrasp(transparent_depth, transparent_depth_max)
    transparent_depth_nomask = transparent_depth.copy()
    transparent_depth[masked_idx]=0
    generated = decenter_and_descale_cleargrasp(generated, transparent_depth_max)
    generated_nomask = generated.copy()
    generated[masked_idx]=0
    opaque_depth = decenter_and_descale_cleargrasp(opaque_depth, opaque_depth_max)
    opaque_depth_nomask = opaque_depth.copy()
    opaque_depth[masked_idx]=0
    difference = np.squeeze(generated) - opaque_depth
    #difference[np.abs(difference)>0.2] = 0
    opaque_to_transparent_difference = opaque_depth - transparent_depth
    rmse_baseline = np.sqrt(
        np.mean(
            np.square(
                opaque_to_transparent_difference[np.abs(opaque_to_transparent_difference)>0])))
    mae_baseline = np.mean(np.abs(opaque_to_transparent_difference[np.abs(opaque_to_transparent_difference)>0]))
    rmse_difference = np.sqrt(np.mean(np.square(difference[np.abs(difference)>0])))
    mae_difference = np.mean(np.abs(difference[np.abs(difference)>0]))

    if vis:
        visualize_cleargrasp_data_depth2depth(rgb,
                                  transparent_depth_nomask,
                                  transparent_label,
                                  opaque_depth_nomask,
                                  opaque_label,
                                  generated_nomask,
                                  difference,
                                  rmse_baseline,
                                  mae_baseline,
                                  rmse_difference,
                                  mae_difference,
                                  0,
                                  transparent_depth_max,
                                  0,
                                  transparent_depth_max,
                                  0,
                                  opaque_depth_max,
                                  output_path)
    return rmse_baseline, mae_baseline, rmse_difference, mae_difference

def evaluate_dataset(dataset_transparent_depth,
                     transparent_depth_max,
                     transparent_label,
                     dataset_opaque_depth,
                     opaque_depth_max,
                     opaque_label,
                     dataset_masks,
                     dataset_rgb,
                     generated,
                     label,
                     vis,
                     n):
    sum_rmse_baseline = 0
    sum_mae_baseline = 0
    sum_rmse_difference = 0
    sum_mae_difference = 0

    for index in range(n):
        rmse_baseline, \
        mae_baseline, \
        rmse_difference, \
        mae_difference = evaluate_single_depth2depth(dataset_rgb[index],
                                         dataset_transparent_depth[index],
                                         transparent_depth_max,
                                         transparent_label,
                                         dataset_opaque_depth[index],
                                         opaque_depth_max,
                                         opaque_label,
                                         dataset_masks[index],
                                         generated[index],
                                         f"results/{label}_{index:04d}.png",
                                         vis)
        sum_rmse_baseline += rmse_baseline
        sum_mae_baseline += mae_baseline
        sum_rmse_difference += rmse_difference
        sum_mae_difference += mae_difference
    return sum_rmse_baseline / n, sum_mae_baseline / n, sum_rmse_difference / n, sum_mae_difference / n

def evaluate_single_rgb_to_depth(rgb,
                                 transparent_depth,
                                 transparent_depth_max,
                                 transparent_label,
                                 opaque_depth,
                                 opaque_depth_max,
                                 opaque_label,
                                 mask,
                                 generated,
                                 output_path="",
                                 vis=False):
    masked_idx = mask==0
    transparent_depth = decenter_and_descale_cleargrasp(transparent_depth, transparent_depth_max)
    transparent_depth_nomask = transparent_depth.copy()
    transparent_depth[masked_idx]=0
    generated = decenter_and_descale_cleargrasp(generated, transparent_depth_max)
    generated_nomask = generated.copy()
    generated[masked_idx]=0
    opaque_depth = decenter_and_descale_cleargrasp(opaque_depth, opaque_depth_max)
    opaque_depth_nomask = opaque_depth.copy()
    opaque_depth[masked_idx]=0
    difference = np.squeeze(generated) - opaque_depth
    opaque_to_transparent_difference = opaque_depth - transparent_depth
    rmse_baseline = np.sqrt(
        np.mean(
            np.power(
                opaque_to_transparent_difference[np.abs(opaque_to_transparent_difference)>0],
                2)))
    mae_baseline = np.mean(np.abs(opaque_to_transparent_difference[np.abs(opaque_to_transparent_difference)>0]))
    rmse_difference = np.sqrt(np.mean(np.power(difference[np.abs(difference)>0], 2)))
    mae_difference = np.mean(np.abs(difference[np.abs(difference)>0]))

    if vis:
        visualize_cleargrasp_data(rgb,
                                              opaque_depth_nomask,
                                              opaque_label,
                                              generated_nomask,
                                              difference,
                                              rmse_baseline,
                                              mae_baseline,
                                              rmse_difference,
                                              mae_difference,
                                              fixed_min_generated=0,
                                              fixed_max_generated=opaque_depth_max,
                                              fixed_min_cg_opaque=0,
                                              fixed_max_cg_opaque=opaque_depth_max,
                                              output_path=output_path)
    return rmse_baseline, mae_baseline, rmse_difference, mae_difference

def evaluate_dataset_rgb_to_depth(dataset_transparent_rgb,
                                  dataset_transparent_depth,
                                  transparent_depth_max,
                                  transparent_label,
                                  dataset_opaque_depth,
                                  opaque_depth_max,
                                  opaque_label,
                                  dataset_masks,
                                  generated,
                                  label,
                                  vis,
                                  n):
    sum_rmse_baseline = 0
    sum_mae_baseline = 0
    sum_rmse_difference = 0
    sum_mae_difference = 0

    for index in range(n):
        rmse_baseline, \
        mae_baseline, \
        rmse_difference, \
        mae_difference = evaluate_single_rgb_to_depth(dataset_transparent_rgb[index],
                                                      dataset_transparent_depth[index],
                                                      transparent_depth_max,
                                                      transparent_label,
                                                      dataset_opaque_depth[index],
                                                      opaque_depth_max,
                                                      opaque_label,
                                                      dataset_masks[index],
                                                      generated[index],
                                                      f"results/{label}_{index:04d}.png",
                                                      vis)
        sum_rmse_baseline += rmse_baseline
        sum_mae_baseline += mae_baseline
        sum_rmse_difference += rmse_difference
        sum_mae_difference += mae_difference
    return sum_rmse_baseline / n, sum_mae_baseline / n, sum_rmse_difference / n, sum_mae_difference / n

def generate_samples(dataset, n, generator_g):
    generated_samples = [None] * n
    for i in range(len(generated_samples)):
        generated_samples[i] = np.squeeze(
            np.array(
                generator_g(
                    tf.expand_dims(dataset[i], 0))))
        # generated_samples[i][dataset[i]==0] = 0
    return generated_samples

def resize_depth(mat, height, width):
    mat = tf.image.resize(mat, [height, width], method=tf.image.ResizeMethod.BILINEAR)
    return mat

def resize_rgb(mat, height, width):
    mat = tf.image.resize(mat, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return mat