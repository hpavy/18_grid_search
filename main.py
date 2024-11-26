import torch
from run import RunSimulation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

############# VARIABLES ################

folder_result_name = "1_first_try"  # name of the result folder

# Uniquement si nouveau modèle

hyper_param_init = {
    "nb_epoch": 1000,  # epoch number
    "save_rate": 50,  # rate to save
    "weight_data": 1,
    "weight_pde": 1,
    "batch_size": 10000,  # for the pde
    "nb_points_pde": 1000000,  # Total number of pde points
    "Re": 100,
    "lr_init": 1e-3,  # Learning rate at the begining of training
    "gamma_scheduler": 0.999,  # Gamma scheduler for lr
    "nb_layers": 10,
    "nb_neurons": 64,
    "n_pde_test": 5000,
    "n_data_test": 5000,
    "nb_points_axes": 12,  # le nombre de points pris par axe par pas de temps
    "x_min": -0.05,
    "x_max": 0.2,
    "y_min": -0.06,
    "y_max": 0.06,
    "t_min": 6.5,
    "t_max": 8,
    "transfert_learning": "None",
}

param_adim = {
    'V': 2.,
    'L': 0.025,
    'rho': 1.2
}

simu = RunSimulation(hyper_param_init, folder_result_name, param_adim)

simu.run()