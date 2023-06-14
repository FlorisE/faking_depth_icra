import numpy as np
import pandas as pd
import tensorflow as tf

import math
import pickle
import sys

sys.path.append('../')
import helper_functions
import my_networks

alpha_0 = 1.0  # generator loss
alpha_2 = 1.0  # supercycle loss
alpha_3 = 1.0  # same result loss
alpha_4 = 0.5 # identity loss
alpha_5 = 1.0  # half cycle loss

MIN_DEPTH = 0
MAX_DEPTH = 2000


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
                o2t_d_rmses,
                o2t_c_rmses):
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
                  'o2t_d_rmses': o2t_d_rmses,
                  'o2t_c_rmses': o2t_c_rmses}

    with open(stats_path, 'wb') as stats_file:
        pickle.dump(stats_dict, stats_file)
        

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss_least_squares(discriminated_generated):
    # using c=1
    return tf.square(discriminated_generated - tf.ones_like(discriminated_generated))


def discriminator_loss_least_squares(discriminated_real, discriminated_generated):
    # using b=1, a=0
    return tf.square(discriminated_real - tf.ones_like(discriminated_real)) + tf.square(discriminated_generated)


def calc_cycle_loss(real_image, cycled_image):
    return tf.reduce_mean(tf.abs(real_image - cycled_image))


def identity_loss(real_image, same_image):
    return tf.reduce_mean(tf.abs(real_image - same_image))


@tf.function
def train_generator(transparent,
                    opaque,
                    generator_t2o,
                    generator_o2t,
                    generator_t2o_optimizer,
                    generator_o2t_optimizer,
                    discriminator_t,
                    discriminator_o,
                    alpha_1):
    with tf.GradientTape(persistent=True) as tape:
        generated_t2o = generator_t2o(transparent, training=True)
        cycled_t2o2t = generator_o2t(generated_t2o, training=True)
        
        generated_o2t = generator_o2t(opaque, training=True)
        cycled_o2t2o = generator_t2o(generated_o2t, training=True)
        
        same_o = generator_t2o(opaque, training=True)
        same_t = generator_o2t(transparent, training=True)
        
        disc_generated_t2o = discriminator_o(generated_t2o, training=False)
        disc_generated_o2t = discriminator_t(generated_o2t, training=False)
        
        gen_t2o_loss = generator_loss_least_squares(disc_generated_t2o)
        gen_o2t_loss = generator_loss_least_squares(disc_generated_o2t)
        
        cycle_loss_o = calc_cycle_loss(opaque, cycled_o2t2o)
        cycle_loss_t = calc_cycle_loss(transparent, cycled_t2o2t)
        total_cycle_loss = cycle_loss_o + cycle_loss_t
        
        identity_loss_t2o = identity_loss(transparent, same_t)
        identity_loss_o2t = identity_loss(opaque, same_o)
        
        total_gen_o2t_loss = alpha_0 * gen_o2t_loss
        total_gen_o2t_loss += alpha_1 * total_cycle_loss
        total_gen_o2t_loss += alpha_4 * identity_loss_o2t
        
        total_gen_t2o_loss = alpha_0 * gen_t2o_loss
        total_gen_t2o_loss += alpha_1 * total_cycle_loss
        total_gen_t2o_loss += alpha_4 * identity_loss_t2o
    
    generator_o2t_gradients = tape.gradient(total_gen_o2t_loss, 
                                          generator_o2t.trainable_variables)
    generator_t2o_gradients = tape.gradient(total_gen_t2o_loss, 
                                          generator_t2o.trainable_variables)
    
    generator_o2t_optimizer.apply_gradients(zip(generator_o2t_gradients, 
                                              generator_o2t.trainable_variables))
    generator_t2o_optimizer.apply_gradients(zip(generator_t2o_gradients, 
                                              generator_t2o.trainable_variables))
    
    return alpha_0 * gen_t2o_loss, \
           alpha_0 * gen_o2t_loss, \
           alpha_1 * cycle_loss_o, \
           alpha_1 * cycle_loss_t, \
           alpha_1 * total_cycle_loss, \
           alpha_4 * identity_loss_o2t, \
           alpha_4 * identity_loss_t2o, \
           total_gen_t2o_loss, \
           total_gen_o2t_loss, \
           generated_t2o, \
           generated_o2t


@tf.function
def train_generator_mult(transparent,
                         opaque,
                         generator_t2o,
                         generator_o2t,
                         generator_t2o_optimizer,
                         generator_o2t_optimizer,
                         discriminator_t,
                         discriminator_o,
                         alpha_1,
                         depth_mult=3,
                         rgb_mult=1):
    with tf.GradientTape(persistent=True) as tape:
        generated_t2o = generator_t2o(transparent, training=True)
        cycled_t2o2t = generator_o2t(generated_t2o, training=True)
        
        generated_o2t = generator_o2t(opaque, training=True)
        cycled_o2t2o = generator_t2o(generated_o2t, training=True)
        
        same_o = generator_t2o(opaque, training=True)
        same_t = generator_o2t(transparent, training=True)
        
        disc_generated_t2o = discriminator_o(generated_t2o, training=False)
        disc_generated_o2t = discriminator_t(generated_o2t, training=False)
        
        gen_t2o_loss = generator_loss_least_squares(disc_generated_t2o)
        gen_o2t_loss = generator_loss_least_squares(disc_generated_o2t)
        
        cycle_loss_o_rgb = rgb_mult * calc_cycle_loss(opaque[:,:,:,1:4], cycled_o2t2o[:,:,:,1:4])
        cycle_loss_t_rgb = rgb_mult * calc_cycle_loss(transparent[:,:,:,1:4], cycled_t2o2t[:,:,:,1:4])
        
        cycle_loss_o_depth = depth_mult * calc_cycle_loss(opaque[:,:,:,0:1], cycled_o2t2o[:,:,:,0:1])
        cycle_loss_t_depth = depth_mult * calc_cycle_loss(transparent[:,:,:,0:1], cycled_t2o2t[:,:,:,0:1])
        
        total_cycle_loss = cycle_loss_o_rgb + cycle_loss_t_rgb + cycle_loss_o_depth + cycle_loss_t_depth
        
        identity_loss_t2o_rgb = identity_loss(transparent[:,:,:,1:4], same_t[:,:,:,1:4])
        identity_loss_o2t_rgb = identity_loss(opaque[:,:,:,1:4], same_o[:,:,:,1:4])
        
        identity_loss_t2o_depth = identity_loss(transparent[:,:,:,0:1], same_t[:,:,:,0:1])
        identity_loss_o2t_depth = identity_loss(opaque[:,:,:,0:1], same_o[:,:,:,0:1])
        
        total_gen_o2t_loss = alpha_0 * gen_o2t_loss
        total_gen_o2t_loss += alpha_1 * total_cycle_loss
        total_gen_o2t_loss += alpha_4 * identity_loss_o2t_rgb + alpha_4 * identity_loss_o2t_depth
        
        total_gen_t2o_loss = alpha_0 * gen_t2o_loss
        total_gen_t2o_loss += alpha_1 * total_cycle_loss
        total_gen_t2o_loss += alpha_4 * identity_loss_t2o_rgb + alpha_4 * identity_loss_t2o_depth
    
    generator_o2t_gradients = tape.gradient(total_gen_o2t_loss, 
                                          generator_o2t.trainable_variables)
    generator_t2o_gradients = tape.gradient(total_gen_t2o_loss, 
                                          generator_t2o.trainable_variables)
    
    generator_o2t_optimizer.apply_gradients(zip(generator_o2t_gradients, 
                                              generator_o2t.trainable_variables))
    generator_t2o_optimizer.apply_gradients(zip(generator_t2o_gradients, 
                                              generator_t2o.trainable_variables))
    
    return alpha_0 * gen_t2o_loss, \
           alpha_0 * gen_o2t_loss, \
           alpha_1 * cycle_loss_o_depth + alpha_1 * cycle_loss_o_rgb, \
           alpha_1 * cycle_loss_t_depth + alpha_1 * cycle_loss_t_rgb, \
           alpha_1 * total_cycle_loss, \
           alpha_4 * identity_loss_o2t_depth + alpha_4 * identity_loss_o2t_rgb, \
           alpha_4 * identity_loss_t2o_depth + alpha_4 * identity_loss_t2o_rgb, \
           total_gen_t2o_loss, \
           total_gen_o2t_loss, \
           generated_t2o, \
           generated_o2t


@tf.function
def train_discriminator(transparent,
                        generated_o2t,
                        opaque,
                        generated_t2o,
                        discriminator_t,
                        discriminator_o,
                        discriminator_t_optimizer,
                        discriminator_o_optimizer):
    with tf.GradientTape(persistent=True) as tape:
        disc_opaque = discriminator_o(opaque, training=True)
        disc_transparent = discriminator_t(transparent, training=True)
    
        disc_generated_t2o = discriminator_o(generated_t2o, training=True)
        disc_generated_o2t = discriminator_t(generated_o2t, training=True)
    
        disc_o_loss = discriminator_loss_least_squares(disc_opaque, disc_generated_t2o)
        disc_t_loss = discriminator_loss_least_squares(disc_transparent, disc_generated_o2t)
    
    discriminator_o_gradients = tape.gradient(disc_o_loss, 
                                              discriminator_o.trainable_variables)
    
    discriminator_t_gradients = tape.gradient(disc_t_loss, 
                                              discriminator_t.trainable_variables)
    
    discriminator_o_optimizer.apply_gradients(zip(discriminator_o_gradients,
                                                  discriminator_o.trainable_variables))
    
    discriminator_t_optimizer.apply_gradients(zip(discriminator_t_gradients,
                                                  discriminator_t.trainable_variables))
    
    return disc_t_loss, disc_o_loss


def batch_it(tensor):
    return ntensor, 0)


rmse = lambda error: np.sqrt(np.mean(np.square(error)))
mae = lambda error: np.mean(np.abs(error))
me = lambda error: np.mean(error)
rel = lambda estimated, gt: np.abs(estimated - gt) / gt
def thresh(estimated, gt, theta, mask):
    r = rel(estimated, gt)[mask]
    if len(r) == 0:
        return 0.0
    return len(r[r < theta]) / len(r)


def measure(transparent_matching_list, opaque_matching_list, masks_matching, generator_t2o, min_depth=MIN_DEPTH, max_depth=MAX_DEPTH, min_depth_valid=450):
    data = []
    n = len(transparent_matching_list)
    just_masked_mae = []
    for i in range(n):
        transparent = tf.concat(transparent_matching_list[i], 2)
        opaque = tf.concat(opaque_matching_list[i], 2)
        generated_opaque = tf.squeeze(generator_t2o(batch_it(transparent)))
        depth = generated_opaque[:,:,0].numpy()
        depth_raw = helper_functions.decenter_and_descale_depth_bin(depth, min_depth, max_depth)
        gt = opaque[:,:,0].numpy()
        gt_raw = helper_functions.decenter_and_descale_depth_bin(gt, min_depth, max_depth)
        generated_valid = depth_raw > min_depth_valid
        gt_valid = gt_raw > min_depth_valid
        mask_valid = masks_matching[i] == 1
        valid = generated_valid & gt_valid
        valid_masked = generated_valid & gt_valid & mask_valid
        baseline = transparent[:,:,0].numpy()
        baseline_raw = helper_functions.decenter_and_descale_depth_bin(baseline, min_depth, max_depth)
        
        rel_depth_gt = rel(depth_raw[gt_valid], gt_raw[gt_valid]) if np.any(gt_valid) else np.inf
        rel_depth_gt_match = rel(depth_raw[gt_valid & mask_valid], gt_raw[gt_valid & mask_valid]) if np.any(gt_valid & mask_valid) else np.inf
        
        masked_mae = mae((depth_raw - gt_raw)[valid_masked]) if np.any(valid_masked) > 0 else np.inf
        
        just_masked_mae.append(masked_mae)
        
        data.append([rmse((depth_raw - gt_raw)[valid]) if np.any(valid) else np.inf,
                     rmse((depth_raw - gt_raw)[valid_masked]) if np.any(valid_masked) else np.inf,
                     mae((depth_raw - gt_raw)[valid]) if np.any(valid) > 0 else np.inf,
                     masked_mae,
                     np.mean(rel_depth_gt),
                     np.mean(rel_depth_gt_match),
                     thresh(depth_raw, gt_raw, 0.05, masks_matching[i] == 1),
                     thresh(depth_raw, gt_raw, 0.10, masks_matching[i] == 1),
                     thresh(depth_raw, gt_raw, 0.25, masks_matching[i] == 1),
                     thresh(depth_raw, gt_raw, (math.pow(1.25, 2) - 1), masks_matching[i] == 1),
                     thresh(depth_raw, gt_raw, (math.pow(1.25, 3) - 1), masks_matching[i] == 1)
                    ])

    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    df = pd.DataFrame(data, 
                      columns=["Generated vs GT RMSE", 
                               "Generated vs GT RMSE (masked)", 
                               "Generated vs GT MAE", 
                               "Generated vs GT MAE (masked)", 
                               "Relative error",
                               "Relative error (masked)",
                               "1.05",
                              "1.10",
                              "1.25",
                              "1.25 ** 2",
                              "1.25 ** 3"])
    df.loc['mean'] = df.mean()
    print(df)
    return np.mean(just_masked_mae)
    


class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, initial_learning_rate, batch_size):
        self.initial_learning_rate = initial_learning_rate
        self.batch_size = batch_size

    def __call__(self, step):
        return tf.cond((step / self.batch_size) < 100, lambda: self.initial_learning_rate, lambda: self.initial_learning_rate - ((step / self.batch_size) - 100) * (self.initial_learning_rate / 100))


def get_networks(g_in_channels, g_out_channels, f_in_channels, f_out_channels, generator_dropout=0.5, conv_after=False, use_attention=False):
    generator_t2o = my_networks.unet_generator(g_out_channels, norm_type='instancenorm', input_channels=g_in_channels, size=512, dropout_rate=generator_dropout, conv_after=conv_after, use_attention=use_attention)
    generator_o2t = my_networks.unet_generator(f_out_channels, norm_type='instancenorm', input_channels=f_in_channels, size=512, dropout_rate=generator_dropout, conv_after=conv_after, use_attention=use_attention)

    # unet and resnet use the same discriminator
    discriminator_t = my_networks.unet_discriminator(norm_type='instancenorm', target=False, channels=f_out_channels, size=512)
    discriminator_o = my_networks.unet_discriminator(norm_type='instancenorm', target=False, channels=g_out_channels, size=512)
    
    return generator_t2o, generator_o2t, discriminator_t, discriminator_o


def get_uresnet_networks(g_in_channels, g_out_channels, f_in_channels, f_out_channels, generator_dropout=0.5):
    generator_t2o = my_networks.uresnet_generator(g_out_channels, norm_type='instancenorm', input_channels=g_in_channels, dropout_rate=generator_dropout)
    generator_o2t = my_networks.uresnet_generator(f_out_channels, norm_type='instancenorm', input_channels=f_in_channels, dropout_rate=generator_dropout)
    
    print(generator_t2o.summary())

    # unet and resnet use the same discriminator
    discriminator_t = my_networks.unet_discriminator(norm_type='instancenorm', target=False, channels=f_out_channels, size=512)
    discriminator_o = my_networks.unet_discriminator(norm_type='instancenorm', target=False, channels=g_out_channels, size=512)
    
    return generator_t2o, generator_o2t, discriminator_t, discriminator_o


def get_resnet_networks(g_in_channels, g_out_channels, f_in_channels, f_out_channels):
    generator_t2o = my_networks.resnet_generator(n_resnet=9, size=512, input_channels=g_in_channels, output_channels=g_out_channels)
    generator_o2t = my_networks.resnet_generator(n_resnet=9, size=512, input_channels=f_in_channels, output_channels=f_out_channels)

    # unet and resnet use the same discriminator
    discriminator_t = my_networks.unet_discriminator(norm_type='instancenorm', target=False, channels=f_out_channels, size=512)
    discriminator_o = my_networks.unet_discriminator(norm_type='instancenorm', target=False, channels=g_out_channels, size=512)
    
    return generator_t2o, generator_o2t, discriminator_t, discriminator_o


def get_optimizers(generator_lr=2e-4, discriminator_lr=1e-4, beta_1=0.5):
    # CycleGAN paper uses 2e-4
    # LSGAN paper uses 1e-3
    generator_t2o_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(generator_lr, 1000), beta_1=beta_1)
    generator_o2t_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(generator_lr, 1000), beta_1=beta_1)

    discriminator_t_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(discriminator_lr, 1000), beta_1=beta_1)
    discriminator_o_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(discriminator_lr, 1000), beta_1=beta_1)
    
    return generator_t2o_optimizer, generator_o2t_optimizer, discriminator_t_optimizer, discriminator_o_optimizer


def get_optimizers_gen_only(generator_lr=2e-4, beta_1=0.5):
    # CycleGAN paper uses 2e-4
    # LSGAN paper uses 1e-3
    generator_t2o_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(generator_lr, 1000), beta_1=beta_1)
    generator_o2t_optimizer = tf.keras.optimizers.Adam(MyLRSchedule(generator_lr, 1000), beta_1=beta_1)
    
    return generator_t2o_optimizer, generator_o2t_optimizer


def get_checkpoint_manager(generator_t2o,
                           generator_o2t,
                           discriminator_t,
                           discriminator_o,
                           generator_t2o_optimizer,
                           generator_o2t_optimizer,
                           discriminator_t_optimizer,
                           discriminator_o_optimizer,
                           checkpoint_path,
                           max_to_keep=5,
                           load=False):
    ckpt = tf.train.Checkpoint(generator_t2o=generator_t2o,
                               generator_o2t=generator_o2t,
                               discriminator_t=discriminator_t,
                               discriminator_o=discriminator_o,
                               generator_t2o_optimizer=generator_t2o_optimizer,
                               generator_o2t_optimizer=generator_o2t_optimizer,
                               discriminator_t_optimizer=discriminator_t_optimizer,
                               discriminator_o_optimizer=discriminator_o_optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=max_to_keep)

    if load and ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored')
    
    return ckpt_manager
    

def get_checkpoint_manager_gen_only(generator_t2o,
                                    generator_o2t,
                                    generator_t2o_optimizer,
                                    generator_o2t_optimizer,
                                    checkpoint_path,
                                    max_to_keep=5,
                                    load=False):
    ckpt = tf.train.Checkpoint(generator_t2o=generator_t2o,
                               generator_o2t=generator_o2t,
                               generator_t2o_optimizer=generator_t2o_optimizer,
                               generator_o2t_optimizer=generator_o2t_optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=max_to_keep)

    if load and ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored')
    
    return ckpt_manager    