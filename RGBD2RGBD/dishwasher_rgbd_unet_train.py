import tensorflow as tf
import pandas as pd
import numpy as np

import sys
import time
import datetime

sys.path.append('../')
from rgbd_dataloader import get_train_dataset, get_test_dataset
from pool import Pool


from dishwasher_rgbd_shared import train_generator, \
                                   train_generator_mult, \
                                   train_discriminator, \
                                   batch_it, \
                                   measure, \
                                   get_networks, \
                                   get_optimizers, \
                                   get_checkpoint_manager, \
                                   load_stats, \
                                   write_stats

from dishwasher_rgbd_shared import get_networks

AUTOTUNE = tf.data.AUTOTUNE

MIN_DEPTH=0
MIN_CLIP=450
MAX_DEPTH=2000
STATS_PATH='dishwasher_rgbd_unet.obj'
BEST_CHECKPOINT_PATH='../data/checkpoints/dishwasherRGBD-unet-best'
LATEST_CHECKPOINT_PATH='../data/checkpoints/dishwasherRGBD-unet-latest'
LOGS='./logs/dishwasher_rgbd_unet/'
POOL_SIZE = 50
CONV_AFTER = False
USE_ATTENTION = False
GENERATOR_LR = 1e-4
BETA_1 = 0.9
train_generator_func = train_generator

def calculate_validation_rmse(generator_t2o, generator_o2t, transparent_matching_list, opaque_matching_list):
    t2o_d_rmses = []
    t2o_c_rmses = []
    o2t_d_rmses = []
    o2t_c_rmses = []
    for i in range(len(transparent_matching_list)):
        transparent = tf.concat(transparent_matching_list[i], 2)
        opaque = tf.concat(opaque_matching_list[i], 2)
        generated_t2o = tf.squeeze(generator_t2o(batch_it(transparent)))
        generated_o2t = tf.squeeze(generator_o2t(batch_it(opaque)))
        t2o_c_rmses.append(np.sqrt(np.mean(np.square(opaque[:,:,1:4] - generated_t2o[:,:,1:4]))))
        t2o_d_rmses.append(np.sqrt(np.mean(np.square(opaque[:,:,0] - generated_t2o[:,:,0]))))
        o2t_c_rmses.append(np.sqrt(np.mean(np.square(transparent[:,:,1:4] - generated_o2t[:,:,1:4]))))
        o2t_d_rmses.append(np.sqrt(np.mean(np.square(transparent[:,:,0] - generated_o2t[:,:,0]))))
    return np.mean(t2o_d_rmses), np.mean(t2o_c_rmses), np.mean(o2t_d_rmses), np.mean(o2t_c_rmses)


def train(num_epochs, 
          bs, 
          dataset,
          generator_t2o, 
          generator_o2t, 
          discriminator_t, 
          discriminator_o, 
          generator_t2o_optimizer, 
          generator_o2t_optimizer, 
          discriminator_t_optimizer,
          discriminator_o_optimizer,
          ckpt_manager_best,
          ckpt_manager_latest,
          transparent_matching,
          opaque_matching,
          masks_matching,
          start_from=0,
          min_depth=MIN_DEPTH,
          max_depth=MAX_DEPTH):
    total_epochs = start_from
    
    if start_from != 0:
        stats_dict = load_stats(STATS_PATH)
        
        gen_t2o_losses = stats_dict['gen_t2o_losses']
        gen_o2t_losses = stats_dict['gen_o2t_losses']
        cycle_losses_o = stats_dict['cycle_losses_o']
        cycle_losses_t = stats_dict['cycle_losses_t']
        total_cycle_losses = stats_dict['total_cycle_losses']
        identity_losses_o2t = stats_dict['identity_losses_o2t']
        identity_losses_t2o = stats_dict['identity_losses_t2o']
        total_gen_t2o_losses = stats_dict['total_gen_t2o_losses']
        total_gen_o2t_losses = stats_dict['total_gen_o2t_losses']
        disc_t_losses = stats_dict['disc_t_losses']
        disc_o_losses = stats_dict['disc_o_losses']
        t2o_d_rmses = stats_dict['t2o_d_rmses']
        t2o_c_rmses = stats_dict['t2o_c_rmses']
        o2t_d_rmses = stats_dict['o2t_d_rmses']
        o2t_c_rmses = stats_dict['o2t_c_rmses']
    else:
        gen_t2o_losses = []
        gen_o2t_losses = []
        cycle_losses_o = []
        cycle_losses_t = []
        total_cycle_losses = []
        identity_losses_o2t = []
        identity_losses_t2o = []
        total_gen_t2o_losses = []
        total_gen_o2t_losses = []
        disc_t_losses = []
        disc_o_losses = []
        t2o_d_rmses = []
        t2o_c_rmses = []
        o2t_d_rmses = []
        o2t_c_rmses = []
    
    best_val_loss = None
    
    generated_transparent_pool = Pool(POOL_SIZE)
    generated_opaque_pool = Pool(POOL_SIZE)
    
    for epoch in range(start_from, num_epochs):
        start = time.time()

        n = 0
        gen_t2o_losses_epoch = []
        gen_o2t_losses_epoch = []
        cycle_losses_o_epoch = []
        cycle_losses_t_epoch = []
        total_cycle_losses_epoch = []
        identity_losses_o2t_epoch = []
        identity_losses_t2o_epoch = []
        total_gen_t2o_losses_epoch = []
        total_gen_o2t_losses_epoch = []

        disc_t_losses_epoch = []
        disc_o_losses_epoch = []

        for transparent_, opaque_ in dataset.shuffle(bs).prefetch(tf.data.AUTOTUNE).take(bs):
            print('.')
            transparent = tf.concat([transparent_[0], transparent_[1]], 3)
            opaque = tf.concat([opaque_[0], opaque_[1]], 3)

            alpha_1 = max(10 - float(total_epochs/10), 1.0)

            gen_t2o_loss, gen_o2t_loss, cycle_loss_o, cycle_loss_t, total_cycle_loss, identity_loss_o2t, identity_loss_t2o, total_gen_t2o_loss, total_gen_o2t_loss, generated_t2o, generated_o2t = \
                train_generator_func(transparent, opaque, generator_t2o, generator_o2t, generator_t2o_optimizer, generator_o2t_optimizer, discriminator_t, discriminator_o, alpha_1)

            gen_t2o_losses_epoch.append(gen_t2o_loss)
            gen_o2t_losses_epoch.append(gen_o2t_loss)
            cycle_losses_o_epoch.append(cycle_loss_o)
            cycle_losses_t_epoch.append(cycle_loss_t)
            total_cycle_losses_epoch.append(total_cycle_loss)
            identity_losses_o2t_epoch.append(identity_loss_o2t)
            identity_losses_t2o_epoch.append(identity_loss_t2o)
            total_gen_t2o_losses_epoch.append(total_gen_t2o_loss)
            total_gen_o2t_losses_epoch.append(total_gen_o2t_loss)

            disc_t_loss, disc_o_loss = train_discriminator(transparent, generated_transparent_pool.query(generated_o2t), opaque, generated_opaque_pool.query(generated_t2o), discriminator_t, discriminator_o, discriminator_t_optimizer, discriminator_o_optimizer)

            disc_t_losses_epoch.append(disc_t_loss)
            disc_o_losses_epoch.append(disc_o_loss)
            
            n += 1

        mean_gen_t2o_losses_epoch = np.mean(gen_t2o_losses_epoch)
        mean_gen_o2t_losses_epoch = np.mean(gen_o2t_losses_epoch)
        mean_cycle_losses_o_epoch = np.mean(cycle_losses_o_epoch)
        mean_cycle_losses_t_epoch = np.mean(cycle_losses_t_epoch)
        mean_total_cycle_losses_epoch = np.mean(total_cycle_losses_epoch)
        mean_identity_losses_o2t_epoch = np.mean(identity_losses_o2t_epoch)
        mean_identity_losses_t2o_epoch = np.mean(identity_losses_t2o_epoch)
        mean_total_gen_o2t_losses_epoch = np.mean(total_gen_o2t_losses_epoch)
        mean_total_gen_t2o_losses_epoch = np.mean(total_gen_t2o_losses_epoch)
        
        mean_disc_t_losses_epoch = np.mean(disc_t_losses_epoch)
        mean_disc_o_losses_epoch = np.mean(disc_o_losses_epoch)
        
        gen_t2o_losses.append(mean_gen_t2o_losses_epoch)
        gen_o2t_losses.append(mean_gen_o2t_losses_epoch)
        cycle_losses_o.append(mean_cycle_losses_o_epoch)
        cycle_losses_t.append(mean_cycle_losses_t_epoch)
        total_cycle_losses.append(mean_total_cycle_losses_epoch)
        identity_losses_o2t.append(mean_identity_losses_o2t_epoch)
        identity_losses_t2o.append(mean_identity_losses_t2o_epoch)
        total_gen_t2o_losses.append(mean_total_gen_t2o_losses_epoch)
        total_gen_o2t_losses.append(mean_total_gen_o2t_losses_epoch)

        disc_t_losses.append(mean_disc_t_losses_epoch)
        disc_o_losses.append(mean_disc_o_losses_epoch)

        t2o_d_rmse, t2o_c_rmse, o2t_d_rmse, o2t_c_rmse = calculate_validation_rmse(generator_t2o, generator_o2t, transparent_matching, opaque_matching)

        t2o_d_rmses.append(t2o_d_rmse)
        t2o_c_rmses.append(t2o_c_rmse)
        o2t_d_rmses.append(o2t_d_rmse)
        o2t_c_rmses.append(o2t_c_rmse)

        masked_mae = measure(transparent_matching, opaque_matching, masks_matching, generator_t2o, min_depth=min_depth, max_depth=max_depth, min_depth_valid=MIN_CLIP)
        
        if best_val_loss is None:
            print('Saving initial best validation loss')
            best_val_loss = masked_mae
            ckpt_manager_best.save()
        elif masked_mae < best_val_loss:
            print('New best validation loss, saving checkpoint...')
            best_val_loss = masked_mae
            ckpt_manager_best.save()
        ckpt_manager_latest.save()

        total_epochs += 1
        
        print('Writing statistics')
        
        write_stats(STATS_PATH,
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
                    o2t_c_rmses)
        
        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                            time.time()-start))


if __name__ == '__main__':
    generator_t2o, generator_o2t, discriminator_t, discriminator_o = get_networks(4, 4, 4, 4, generator_dropout=0.5, conv_after=CONV_AFTER, use_attention=USE_ATTENTION)
    
    generator_t2o_optimizer, generator_o2t_optimizer, discriminator_t_optimizer, discriminator_o_optimizer = get_optimizers(generator_lr=GENERATOR_LR, beta_1=BETA_1)
    
    ckpt_manager_best = get_checkpoint_manager(generator_t2o,
                                               generator_o2t,
                                               discriminator_t,
                                               discriminator_o,
                                               generator_t2o_optimizer,
                                               generator_o2t_optimizer,
                                               discriminator_t_optimizer,
                                               discriminator_o_optimizer,
                                               BEST_CHECKPOINT_PATH,
                                               load=False,
                                                 max_to_keep=1)
    
    ckpt_manager_latest = get_checkpoint_manager(generator_t2o,
                                                 generator_o2t,
                                                 discriminator_t,
                                                 discriminator_o,
                                                 generator_t2o_optimizer,
                                                 generator_o2t_optimizer,
                                                 discriminator_t_optimizer,
                                                 discriminator_o_optimizer,
                                                 LATEST_CHECKPOINT_PATH,
                                                 load=False,
                                                 max_to_keep=1)
        
    transparent_dataset_rz, opaque_dataset_rz = get_train_dataset(min_depth=MIN_DEPTH, min_clip=MIN_CLIP)
        
    dataset = tf.data.Dataset.zip((transparent_dataset_rz, opaque_dataset_rz))
    
    transparent_matching, opaque_matching, masks_matching = get_test_dataset(min_depth=MIN_DEPTH, min_clip=MIN_CLIP)
    
    train(200, 1000, dataset, generator_t2o, generator_o2t, discriminator_t, discriminator_o, generator_t2o_optimizer, generator_o2t_optimizer, discriminator_t_optimizer, discriminator_o_optimizer, ckpt_manager_best, ckpt_manager_latest, transparent_matching, opaque_matching, masks_matching)
