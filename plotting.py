import matplotlib.pyplot as plt
import numpy as np
import helper_functions

def plot_generator_losses(gen_tc2od_losses, 
                          gen_od2tc_losses, 
                          total_cycle_losses,
                          total_cycle_losses_d2d,
                          gen_od2td_losses, 
                          gen_td2od_losses, 
                          total_gen_tc2od_losses, 
                          total_gen_od2tc_losses,
                          total_gen_od2td_losses,
                          total_gen_td2od_losses,
                          identity_losses_od2td,
                          identity_losses_td2od):
                         # same_result_losses,
                         # total_supercycle_losses,
                         # identity_losses_od2td,
                         # identity_losses_td2od,
                         # half_cycle_losses_td2tg,
                         # half_cycle_losses_tg2td
    plt.figure(figsize=(10, 5))
    plt.plot(list(zip(gen_tc2od_losses, 
                      gen_od2tc_losses, 
                      total_cycle_losses, 
                      total_gen_tc2od_losses, 
                      total_gen_od2tc_losses)))
    plt.title("Generator Losses for Transparent Color to Opaque Depth and vice versa")
    plt.legend(["TC2OD GAN",
                "OD2TC GAN loss",
                "Cycle loss",
                "TC2OD loss (GAN+Cycle)",
                "OD2TC loss"])
    plt.savefig(f"generator_color_losses_latest.png")
    
    plt.figure(figsize=(10, 5))
    plt.plot(list(zip(gen_od2td_losses, 
                      gen_td2od_losses, 
                      total_cycle_losses_d2d,
                      identity_losses_od2td,
                      identity_losses_td2od,
                      total_gen_od2td_losses,
                      total_gen_td2od_losses)))
                      # same_result_losses,
                      # total_supercycle_losses,
                      # identity_losses_od2td,
                      # identity_losses_td2od,
                      # half_cycle_losses_td2tg,
                      # half_cycle_losses_tg2td
    plt.title("Generator Losses for Opaque Depth to Transparent Depth and vice versa")
    plt.legend(["OD2TD GAN",
                "TD2OD GAN",
                "Cycle",
                "OD2TD Identity",
                "TD2OD Identity",
                "OD2TD (GAN+Cycle+Identity)",
                "TD2OD"])
                # "Same result loss",
                # "Supercycle loss",
                # "Identity loss OD2TD",
                # "Identity loss TD2OD",
                # "Half cycle loss TD2TG",
                # "Half cycle loss TG2TD"])
    plt.savefig(f"generator_depth_losses_latest.png")
    
def plot_discriminator_losses(disc_x_losses, 
                              disc_y_losses,
                              disc_td2od_losses,
                              disc_td_losses,
                              disc_od_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(list(zip(disc_x_losses, 
                      disc_y_losses,
                      disc_td2od_losses,
                      disc_td_losses,
                      disc_od_losses)))
    plt.title("Discriminator Losses")
    plt.legend(["Discriminator TC loss",
                "Discriminator TC2OD loss",
                "Discriminator TD2OD loss",
                "Discriminator TD loss",
                "Discriminator OD loss (TC2OD + TD2OD)"])
    plt.savefig(f"discriminator_losses_latest.png")
    
def plot_discriminator_performance(dataset,
                                   discriminator_x,
                                   discriminator_y,
                                   discriminator_td,
                                   generator_g,
                                   generator_f,
                                   generator_td2od,
                                   generator_od2td,
                                   discriminator_actual_x_mean,
                                   discriminator_actual_y_mean,
                                   discriminator_actual_td_mean,
                                   discriminator_fake_x_mean,
                                   discriminator_fake_y_mean,
                                   discriminator_fake_td2od_mean,
                                   discriminator_fake_od2td_mean):
    d_actual_x_running_mean = 0
    d_actual_y_running_mean = 0
    d_actual_td_running_mean = 0
    d_fake_x_running_mean = 0
    d_fake_y_running_mean = 0
    d_fake_td2od_running_mean = 0
    d_fake_od2td_running_mean = 0
    n = 0
    for transparent, opaque in dataset.take(10):
        transparent_color = transparent[2]
        opaque_depth = opaque[0]
        transparent_depth = transparent[0]
        fake_x = generator_f(opaque_depth)
        fake_y = generator_g(transparent_color)
        fake_td2od = generator_td2od(transparent_depth)
        fake_od2td = generator_od2td(opaque_depth)
        d_actual_x_running_mean += np.mean(discriminator_x(transparent_color))
        d_actual_y_running_mean += np.mean(discriminator_y(opaque_depth))
        d_actual_td_running_mean += np.mean(discriminator_td(transparent_depth))
        d_fake_x_running_mean += np.mean(discriminator_x(fake_x))
        d_fake_y_running_mean += np.mean(discriminator_y(fake_y))
        d_fake_td2od_running_mean += np.mean(discriminator_y(fake_td2od))
        d_fake_od2td_running_mean += np.mean(discriminator_td(fake_od2td))

        n += 1
    discriminator_actual_x_mean.append(d_actual_x_running_mean / n)
    discriminator_actual_y_mean.append(d_actual_y_running_mean / n)
    discriminator_actual_td_mean.append(d_actual_td_running_mean / n)
    discriminator_fake_x_mean.append(d_fake_x_running_mean / n)
    discriminator_fake_y_mean.append(d_fake_y_running_mean / n)
    discriminator_fake_td2od_mean.append(d_fake_td2od_running_mean / n)
    discriminator_fake_od2td_mean.append(d_fake_od2td_running_mean / n)
    
    plt.figure(figsize=(10, 5))
    plt.plot(list(zip(discriminator_actual_x_mean,
                      discriminator_actual_y_mean,
                      discriminator_fake_x_mean,
                      discriminator_fake_y_mean)))
    plt.title("Discriminator Performance on Transparent Color and Opaque Depth")
    plt.legend(["Discriminator TC on Actual TC",
                "Discriminator OD on Actual OD",
                "Discriminator TC on Generated TC",
                "Discriminator OD on Generated OD (from TC)"])
    plt.savefig(f"discriminator_grayscale_performance_latest.png")
    
    plt.figure(figsize=(10, 5))
    plt.plot(list(zip(discriminator_actual_y_mean,
                      discriminator_actual_td_mean,
                      discriminator_fake_td2od_mean,
                      discriminator_fake_od2td_mean)))
    plt.title("Discriminator Performance on Opaque Depth and Transparent Depth")
    plt.legend(["Discriminator OD on actual OD",
                "Discriminator TD on actual TD",
                "Discriminator OD on Generated OD (from TD)",
                "Discriminator TD on Generated TD"])
    plt.savefig(f"discriminator_depth_performance_latest.png")
    
def plot_depth_to_depth_results(cleargrasp_transparent_rgb_test_cs_rz,
                                cleargrasp_transparent_depth_test_cs_rz,
                                cleargrasp_transparent_depth_test_max,
                                cleargrasp_opaque_depth_test_cs_rz,
                                cleargrasp_opaque_depth_test_max,
                                cleargrasp_masks_test,
                                generate_test,
                                generate_test_td2od,
                                epoch,
                                rmse_baselines_test_tg2od,
                                mae_baselines_test_tg2od,
                                rmse_differences_test_tg2od,
                                mae_differences_test_tg2od,
                                rmse_baselines_test_td2od,
                                mae_baselines_test_td2od,
                                rmse_differences_test_td2od, mae_differences_test_td2od):
    rmse_baseline_test_tg2od, \
    mae_baseline_test_tg2od, \
    rmse_difference_test_tg2od, \
    mae_difference_test_tg2od = \
        helper_functions.evaluate_dataset_rgb_to_depth(cleargrasp_transparent_rgb_test_cs_rz,
                                                       cleargrasp_transparent_depth_test_cs_rz,
                                                       cleargrasp_transparent_depth_test_max,
                                                       "ClearGrasp transparent", 
                                                       cleargrasp_opaque_depth_test_cs_rz,
                                                       cleargrasp_opaque_depth_test_max,
                                                       "ClearGrasp opaque",
                                                       cleargrasp_masks_test,
                                                       generate_test(),
                                                       f"test_set_tg2od_ep{epoch}",
                                                       True,
                                                       len(cleargrasp_transparent_depth_test_cs_rz))
    
    rmse_baseline_test_td2od, \
    mae_baseline_test_td2od, \
    rmse_difference_test_td2od, \
    mae_difference_test_td2od = \
        helper_functions.evaluate_dataset(cleargrasp_transparent_depth_test_cs_rz,
                                          cleargrasp_transparent_depth_test_max,
                                          "ClearGrasp transparent",
                                          cleargrasp_opaque_depth_test_cs_rz,
                                          cleargrasp_opaque_depth_test_max,
                                          "ClearGrasp opaque",
                                          cleargrasp_masks_test,
                                          cleargrasp_transparent_rgb_test_cs_rz,
                                          generate_test_td2od(),
                                          f"test_set_td2od_ep{epoch}",
                                          True,
                                          len(cleargrasp_transparent_depth_test_cs_rz))
    
    rmse_baselines_test_tg2od.append(rmse_baseline_test_tg2od)
    mae_baselines_test_tg2od.append(mae_baseline_test_tg2od)
    rmse_differences_test_tg2od.append(rmse_difference_test_tg2od)
    mae_differences_test_tg2od.append(mae_difference_test_tg2od)
    rmse_baselines_test_td2od.append(rmse_baseline_test_td2od)
    mae_baselines_test_td2od.append(mae_baseline_test_td2od)
    rmse_differences_test_td2od.append(rmse_difference_test_td2od)
    mae_differences_test_td2od.append(mae_difference_test_td2od)
    
    plt.figure(figsize=(10, 5))
    plt.plot(list(zip(rmse_baselines_test_tg2od, 
                      mae_baselines_test_tg2od,
                      rmse_differences_test_tg2od,
                      mae_differences_test_tg2od)))
    plt.title("Evaluation of TC2OD on ClearGrasp")
    plt.legend(["CG Baseline RMSE",
                "CG Baseline MAE",
                "CG TC2OD RMSE",
                "CG TC2OD MAE"])
    plt.savefig(f"tc2od_latest.png")
    
    plt.figure(figsize=(10, 5))
    plt.plot(list(zip(rmse_baselines_test_td2od, 
                      mae_baselines_test_td2od,
                      rmse_differences_test_td2od,
                      mae_differences_test_td2od)))
    plt.title("Evaluation of TD2OD on ClearGrasp")
    plt.legend(["CG Baseline RMSE",
                "CG Baseline MAE",
                "CG TD2OD RMSE",
                "CG TD2OD MAE"])
    plt.savefig(f"td2od_latest.png")
    
def plot_grayscale_to_depth_results(transparent_rgb_matching_cs_rz, transparent_depth_matching, transparent_depth_matching_max, opaque_depth_matching, opaque_depth_matching_max, mask_matching, transparent_rgb_matching, generate_ours_tg2od, generate_ours_td2od, rmse_baselines_ours_tg2od, mae_baselines_ours_tg2od, rmse_differences_ours_tg2od, mae_differences_ours_tg2od, rmse_baselines_ours_td2od, mae_baselines_ours_td2od, rmse_differences_ours_td2od, mae_differences_ours_td2od):
    rmse_baseline_ours_tg2od, \
    mae_baseline_ours_tg2od, \
    rmse_difference_ours_tg2od, \
    mae_difference_ours_tg2od = helper_functions.evaluate_dataset_rgb_to_depth(transparent_rgb_matching_cs_rz,
                                               np.squeeze(np.array(list(iter(transparent_depth_matching)))),
                                               transparent_depth_matching_max,
                                               "Ours transparent",
                                               np.squeeze(np.array(list(iter(opaque_depth_matching)))),
                                               opaque_depth_matching_max,
                                               "Ours opaque",
                                               np.squeeze(np.array(mask_matching)),
                                               generate_ours_tg2od(),
                                               "ours_tg2od",
                                               True,
                                               len(list(iter(transparent_depth_matching))))
    
    rmse_baseline_ours_td2od, \
    mae_baseline_ours_td2od, \
    rmse_difference_ours_td2od, \
    mae_difference_ours_td2od = helper_functions.evaluate_dataset(np.squeeze(np.array(list(iter(transparent_depth_matching)))),
                                  transparent_depth_matching_max,
                                  "Ours transparent",
                                  np.squeeze(np.array(list(iter(opaque_depth_matching)))),
                                  opaque_depth_matching_max,
                                  "Ours opaque",
                                  np.squeeze(np.array(mask_matching)),
                                  np.array(transparent_rgb_matching),
                                  generate_ours_td2od(),
                                  "ours_td2od",
                                  True,
                                  len(list(iter(transparent_depth_matching))))
    
    rmse_baselines_ours_tg2od.append(rmse_baseline_ours_tg2od)
    mae_baselines_ours_tg2od.append(mae_baseline_ours_tg2od)
    rmse_differences_ours_tg2od.append(rmse_difference_ours_tg2od)
    mae_differences_ours_tg2od.append(mae_difference_ours_tg2od)
    rmse_baselines_ours_td2od.append(rmse_baseline_ours_td2od)
    mae_baselines_ours_td2od.append(mae_baseline_ours_td2od)
    rmse_differences_ours_td2od.append(rmse_difference_ours_td2od)
    mae_differences_ours_td2od.append(mae_difference_ours_td2od)
    
    plt.figure(figsize=(10, 5))
    plt.plot(list(zip(rmse_baselines_ours_tg2od, 
                      mae_baselines_ours_tg2od,
                      rmse_differences_ours_tg2od,
                      mae_differences_ours_tg2od)))
    plt.title("Evaluation of TC2OD on Ours")
    plt.legend(["Ours Baseline RMSE",
                "Ours Baseline MAE",
                "Ours TC2OD RMSE",
                "Ours TC2OD MAE"])
    plt.savefig(f"tc2od_latest_ours.png")
    
    plt.figure(figsize=(10, 5))
    plt.plot(list(zip(rmse_baselines_ours_td2od, 
                      mae_baselines_ours_td2od,
                      rmse_differences_ours_td2od,
                      mae_differences_ours_td2od)))
    plt.title("Evaluation of TD2OD on Ours")
    plt.legend(["Ours Baseline RMSE",
                "Ours Baseline MAE",
                "Ours TD2OD RMSE",
                "Ours TD2OD MAE"])
    plt.savefig(f"td2od_latest_ours.png")