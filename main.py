import torch
from run import RunSimulation
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

name_simu = "onyx_hugo"

#### Le range des variables #####

nb_layers_min = 10
nb_layers_max = 15
nb_neurons_min = 30
nb_neurons_max = 100
lr_init_min = 5e-4
lr_init_max = 5e-2
nb_points_pde_min = 500000
nb_points_pde_max = 4000000
batch_size_min = 3000
batch_size_max = 30000
nb_border_min = 100
nb_border_max = 1000


# le durée d'une simulation 
time_simu = 7200



############# VARIABLES ################

for nb_simu in range(10):
    folder_result_name = name_simu+'_'+str(nb_simu)  # name of the result folder
    lr_init = np.logspace(np.log10(lr_init_min), np.log10(lr_init_max), 1).item()  # pour avoir une distribution logarithmique 
    nb_points_pde = int(np.logspace(np.log10(nb_points_pde_min), np.log10(nb_points_pde_max), 1).item())
    batch_size = int(np.logspace(np.log10(batch_size_min), np.log10(batch_size_max), 1).item())
    nb_neurons = np.random.randint(nb_neurons_min, nb_neurons_max+1)
    nb_layers = np.random.randint(nb_layers_min, nb_layers_max+1)
    nb_border = int(np.logspace(np.log10(nb_border_min), np.log10(nb_border_max), 1).item())
    # Uniquement si nouveau modèle
    hyper_param_init = {
        "time_simu": time_simu,
        "save_rate": 10,  # rate to save
        "weight_data": 1,
        "weight_pde": 1,
        "batch_size": batch_size,  # for the pde
        "nb_points_pde": nb_points_pde,  # Total number of pde points
        "Re": 100,
        "lr_init": lr_init,  # Learning rate at the begining of training
        "gamma_scheduler": 0.999,  # Gamma scheduler for lr
        "nb_layers": nb_layers,
        "nb_neurons": nb_neurons,
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
        "nb_points_close_cylinder": 100,
        "nb_border": nb_border
    }

    param_adim = {
        'V': 2.,
        'L': 0.025,
        'rho': 1.2
    }

    simu = RunSimulation(hyper_param_init, folder_result_name, param_adim)

    simu.run()