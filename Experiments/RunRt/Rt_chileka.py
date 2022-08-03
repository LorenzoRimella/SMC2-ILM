import numpy as np
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import sys
sys.path.append('Experiments/Scripts/')

from SyntheticHouseholds import *
from filtering_smoothing import *

########################################################
# Load the data
########################################################
n_individuals_chileka          = np.load("Experiments/Data/Input/n_individuals_chileka.npy")
n_households_chileka           = np.load("Experiments/Data/Input/n_households_chileka.npy")
households_longitude_chileka   = np.load("Experiments/Data/Input/households_longitude_chileka.npy")
households_latitude_chileka    = np.load("Experiments/Data/Input/households_latitude_chileka.npy")
loc_H_I_index_chileka          = np.load("Experiments/Data/Input/loc_H_I_index_chileka.npy")
individuals_covariates_chileka = np.load("Experiments/Data/Input/individuals_covariates_chileka.npy")

Y_ecoli_chileka = np.load("Experiments/Data/Input/Y_ecoli_chileka.npy")
Y_klebs_chileka = np.load("Experiments/Data/Input/Y_kpneu_chileka.npy")

theta_chileka_ecoli   = np.load("Experiments/Data/Parameters/theta_best_chileka_ecoli.npy") 
theta_chileka_klebs   = np.load("Experiments/Data/Parameters/theta_best_chileka_klebs.npy") 

########################################################
# Create the household class
########################################################
households_chileka = households_individuals(n_individuals_chileka, n_households_chileka, 
                                             households_longitude_chileka, households_latitude_chileka,
                                             loc_H_I_index_chileka, 
                                             individuals_covariates_chileka)

########################################################
# Simulation keys
########################################################

epidemic_type_dict = {
    "3param_gauss" :        "3 parameters within-across households and fixed recovery",
    "3param_exp":           "3 parameters within-across households and fixed recovery (sqrt)",
    "3param_gauss_noprop" : "3 parameters within-across households (no prop) and fixed recovery",
    "3param_exp_noprop" :   "3 parameters within-across households (no prop) and fixed recovery (sqrt)",
    "3param_exp_season" :   "3 parameters within-across households and fixed recovery, seasonality (sqrt)",
    "3param_exp_season_coef" :      "3 parameters within-across households and fixed recovery, seasonality, individuals coefficients (sqrt)",
    "3param_exp_prop_season_coef" : "3 parameters within-across households (all proportions) and fixed recovery, seasonality, individuals coefficients (sqrt)",
    "3param_exp_prop" :      "3 parameters within-across households (all proportions) and fixed recovery (sqrt)",
    "eps_3param_exp_prop_season_coef" : "fixed effect, 3 parameters within-across households (all proportions) and fixed recovery, seasonality, individuals coefficients (sqrt)",
    "eps_3param_exp_prop_season" : "fixed effect, 3 parameters within-across households (all proportions) and fixed recovery, seasonality (sqrt)",
    "3param_exp_prop_season" : "3 parameters within-across households (all proportions) and fixed recovery, seasonality, individuals coefficients (sqrt)",
    "eps_3param_exp_season_coef" :      "fixed effect, 3 parameters within-across households and fixed recovery, seasonality, individuals coefficients (sqrt)",
    "eps_3param_gauss_prop_season_coef" : "fixed effect, 3 parameters within-across households (all proportions) and fixed recovery, seasonality, individuals coefficients (sq)",
    "eps_3param_gauss_prop_season" : "fixed effect, 3 parameters within-across households (all proportions) and fixed recovery, seasonality (sq)",
    "eps_3param_gauss_season_coef" :      "fixed effect, 3 parameters within-across households and fixed recovery, seasonality, individuals coefficients (sq)"

}

prior_type_dict = {
    "3param_gauss" : "Gaussian-Gaussian-Gaussian-MGaussian-Beta",
    "3param_exp":    "Gaussian-Gaussian-Gaussian-MGaussian-Beta",
    "3param_gauss_noprop" : "Gaussian-Gaussian-Gaussian-MGaussian-Beta",
    "3param_exp_noprop" :   "Gaussian-Gaussian-Gaussian-MGaussian-Beta",
    "3param_exp_season" :      "Gaussian-Gaussian-Gaussian-MGaussian",
    "3param_exp_season_coef" : "Gaussian-Gaussian-Gaussian",
    "3param_exp_prop_season_coef" : "Gaussian-Gaussian-Gaussian",
    "3param_exp_prop_season" :      "Gaussian-Gaussian-Gaussian-MGaussian", 
    "3param_exp_prop" :            "Gaussian-Gaussian-Gaussian-MGaussian-Beta",
    "eps_3param_exp_prop_season_coef" : "Gaussian-Gaussian-Gaussian-Gaussian",
    "eps_3param_exp_prop_season" : "Gaussian-Gaussian-Gaussian-Gaussian-MGaussian",
    "eps_3param_exp_season_coef" :      "Gaussian-Gaussian-Gaussian-Gaussian",
    "eps_3param_gauss_prop_season_coef" :  "Gaussian-Gaussian-Gaussian-Gaussian",
    "eps_3param_gauss_prop_season" : "Gaussian-Gaussian-Gaussian-Gaussian-MGaussian",
    "eps_3param_gauss_season_coef" : "Gaussian-Gaussian-Gaussian-Gaussian",
}

hyperparameters_epidemic_coef_dict = {}

fixed_season_keys = ["3param_exp_season",      "3param_exp_season_coef",          "3param_exp_prop_season_coef", 
                     "3param_exp_prop_season", "eps_3param_exp_prop_season_coef", "eps_3param_exp_prop_season", 
                     "eps_3param_exp_season_coef", "eps_3param_gauss_prop_season_coef",
                     "eps_3param_gauss_prop_season", "eps_3param_gauss_season_coef"] 


epidemic_type_dict_complete = epidemic_type_dict.copy()
prior_type_dict_complete    = prior_type_dict.copy()

for key in epidemic_type_dict.keys():
    
    hyperparameters_epidemic_coef_dict[key] = 0.8
    
    if key in fixed_season_keys:
        
        for coef_value in [0.2, 0.4, 0.6]:
            key_new = key+"_"+str(int(10*coef_value))
            hyperparameters_epidemic_coef_dict[key_new] = coef_value
            epidemic_type_dict_complete[key_new]        = epidemic_type_dict_complete[key]
            prior_type_dict_complete[key_new]           = prior_type_dict_complete[key] 
        
epidemic_type_dict = epidemic_type_dict_complete
prior_type_dict    = prior_type_dict_complete

########################################################
# Set the parameters
########################################################

n_particles  = 50

hyperparameters_epidemic_ecoli = {}
hyperparameters_epidemic_ecoli["initial_colonize_prob"] = 0.13
hyperparameters_epidemic_ecoli["gamma"]                 = 1/10
hyperparameters_epidemic_ecoli["coef"]                  = hyperparameters_epidemic_coef_dict["eps_3param_exp_season_coef_6"]
hyperparameters_epidemic_ecoli["individual_coef"]       = tf.zeros((3, 1), dtype = tf.float32)
hyperparameters_epidemic_ecoli["periodicity"]           = 2*np.pi/365.25
hyperparameters_epidemic_ecoli["translation"]           = 0.55*np.pi 
hyperparameters_epidemic_ecoli["sensitivity"]           = 0.8  
hyperparameters_epidemic_ecoli["specificity"]           = 0.95


hyperparameters_epidemic_klebs = {}
hyperparameters_epidemic_klebs["initial_colonize_prob"] = 0.13
hyperparameters_epidemic_klebs["gamma"]                 = 1/10
hyperparameters_epidemic_klebs["coef"]                  = hyperparameters_epidemic_coef_dict["eps_3param_exp_prop_season_coef_6"]
hyperparameters_epidemic_klebs["individual_coef"]       = tf.zeros((3, 1), dtype = tf.float32)
hyperparameters_epidemic_klebs["periodicity"]           = 2*np.pi/365.25
hyperparameters_epidemic_klebs["translation"]           = 0.55*np.pi 
hyperparameters_epidemic_klebs["sensitivity"]           = 0.8  
hyperparameters_epidemic_klebs["specificity"]           = 0.95


####################################################
# filtering
####################################################

epidemic_type_ecoli = epidemic_type_dict["eps_3param_exp_season_coef_6"]      # choose the type according to the best marginal likelihood
epidemic_type_klebs = epidemic_type_dict["eps_3param_gauss_prop_season_coef"] # choose the type according to the best marginal likelihood

# the max delta_t
max_delta_t_ecoli   = 1
max_delta_t_klebs   = 1


filtering_chileka_ecoli = Particle_filter_smoother( households_chileka,   
                                               n_particles,
                                               theta_chileka_ecoli,
                                               hyperparameters_epidemic_ecoli,
                                               epidemic_type_ecoli)

filtering_chileka_klebs = Particle_filter_smoother( households_chileka,   
                                               n_particles,
                                               theta_chileka_klebs,
                                               hyperparameters_epidemic_klebs,
                                               epidemic_type_klebs)


Y_ecoli =  Y_ecoli_chileka
Y_complement_ecoli    = np.empty((households_chileka.n_individuals - Y_ecoli.shape[0], Y_ecoli.shape[1]))
Y_complement_ecoli[:] = np.NaN
Y_ecoli = np.concatenate((Y_ecoli, Y_complement_ecoli), axis = 0)

Y_klebs =  Y_klebs_chileka
Y_complement_klebs    = np.empty((households_chileka.n_individuals - Y_klebs.shape[0], Y_klebs.shape[1]))
Y_complement_klebs[:] = np.NaN
Y_klebs = np.concatenate((Y_klebs, Y_complement_klebs), axis = 0)


file_name_ecoli = "chileka_ecoli"     
file_name_klebs = "chileka_klebs"   


file_name_ecoli = "chileka_ecoli"     
file_name_klebs = "chileka_klebs"              

output_ecoli = filtering_chileka_ecoli.filter(Y_ecoli, max_delta_t_ecoli)
output_klebs = filtering_chileka_klebs.filter(Y_klebs, max_delta_t_klebs)

R_t_ecoli = tf.concat((output_ecoli[3], output_ecoli[4], output_ecoli[5]), axis = 1)
R_t_klebs = tf.concat((output_klebs[3], output_klebs[4], output_klebs[5]), axis = 1)

data_output_name_ecoli = file_name_ecoli+"_households.npy"
np.save("Experiments/Data/Output/" + data_output_name_ecoli, output_ecoli[2])
data_output_name_klebs = file_name_klebs+"_households.npy"
np.save("Experiments/Data/Output/" + data_output_name_klebs, output_klebs[2])

data_output_name_ecoli = file_name_ecoli+"_Rt.npy"
np.save("Experiments/Data/Output/" + data_output_name_ecoli, R_t_ecoli)
data_output_name_klebs = file_name_klebs+"_Rt.npy"
np.save("Experiments/Data/Output/" + data_output_name_klebs, R_t_klebs)


