import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from utils_SMC2 import *

#######################################
# The SMC proposal class
#######################################

class proposal_epidemic():

    ################################################
    # initialization
    ################################################
    def __init__( self, households_individuals, 
                        n_particles,
                        hyperparameters_epidemic,
                        epidemic_type = "3 parameters within-across households and fixed recovery"):

        super(proposal_epidemic, self).__init__()

        self.households_individuals = households_individuals
        self.distances              = self.households_individuals.households_distances
        
        self.n_particles = n_particles

        self.epidemic_type = epidemic_type

        if self.epidemic_type=="3 parameters within-across households and fixed recovery":
            self.initial_colonize_prob = hyperparameters_epidemic["initial_colonize_prob"]

            self.gamma                 = hyperparameters_epidemic["gamma"]

            self.translation           = hyperparameters_epidemic["translation"]
            self.periodicity           = hyperparameters_epidemic["periodicity"]

            self.sensitivity           = hyperparameters_epidemic["sensitivity"]
            self.specificity           = hyperparameters_epidemic["specificity"]  

        if self.epidemic_type=="3 parameters within-across households and fixed recovery (sqrt)":
            self.initial_colonize_prob = hyperparameters_epidemic["initial_colonize_prob"]

            self.gamma                 = hyperparameters_epidemic["gamma"]

            self.translation           = hyperparameters_epidemic["translation"]
            self.periodicity           = hyperparameters_epidemic["periodicity"]

            self.sensitivity           = hyperparameters_epidemic["sensitivity"]
            self.specificity           = hyperparameters_epidemic["specificity"]  
            

        if self.epidemic_type=="3 parameters within-across households (no prop) and fixed recovery":
            self.initial_colonize_prob = hyperparameters_epidemic["initial_colonize_prob"]

            self.gamma                 = hyperparameters_epidemic["gamma"]

            self.translation           = hyperparameters_epidemic["translation"]
            self.periodicity           = hyperparameters_epidemic["periodicity"]

            self.sensitivity           = hyperparameters_epidemic["sensitivity"]
            self.specificity           = hyperparameters_epidemic["specificity"]  

        if self.epidemic_type=="3 parameters within-across households (no prop) and fixed recovery (sqrt)":
            self.initial_colonize_prob = hyperparameters_epidemic["initial_colonize_prob"]

            self.gamma                 = hyperparameters_epidemic["gamma"]

            self.translation           = hyperparameters_epidemic["translation"]
            self.periodicity           = hyperparameters_epidemic["periodicity"]

            self.sensitivity           = hyperparameters_epidemic["sensitivity"]
            self.specificity           = hyperparameters_epidemic["specificity"]  

        if self.epidemic_type=="3 parameters within-across households and fixed recovery, seasonality (sqrt)":
            self.initial_colonize_prob = hyperparameters_epidemic["initial_colonize_prob"]

            self.gamma                 = hyperparameters_epidemic["gamma"]

            self.coef                  = hyperparameters_epidemic["coef"]

            self.translation           = hyperparameters_epidemic["translation"]
            self.periodicity           = hyperparameters_epidemic["periodicity"]

            self.sensitivity           = hyperparameters_epidemic["sensitivity"]
            self.specificity           = hyperparameters_epidemic["specificity"]    

        if self.epidemic_type == "3 parameters within-across households and fixed recovery, seasonality, individuals coefficients (sqrt)":

            self.initial_colonize_prob = hyperparameters_epidemic["initial_colonize_prob"]

            self.gamma                 = hyperparameters_epidemic["gamma"]

            self.individual_coef       = hyperparameters_epidemic["individual_coef"]

            self.coef                  = hyperparameters_epidemic["coef"]

            self.translation           = hyperparameters_epidemic["translation"]
            self.periodicity           = hyperparameters_epidemic["periodicity"]

            self.sensitivity           = hyperparameters_epidemic["sensitivity"]
            self.specificity           = hyperparameters_epidemic["specificity"]   

        if self.epidemic_type == "3 parameters within-across households (all proportions) and fixed recovery, seasonality, individuals coefficients (sqrt)":

            self.initial_colonize_prob = hyperparameters_epidemic["initial_colonize_prob"]

            self.gamma                 = hyperparameters_epidemic["gamma"]

            self.individual_coef       = hyperparameters_epidemic["individual_coef"]

            self.coef                  = hyperparameters_epidemic["coef"]

            self.translation           = hyperparameters_epidemic["translation"]
            self.periodicity           = hyperparameters_epidemic["periodicity"]

            self.sensitivity           = hyperparameters_epidemic["sensitivity"]
            self.specificity           = hyperparameters_epidemic["specificity"]  

        if self.epidemic_type == "3 parameters within-across households (all proportions) and fixed recovery, seasonality (sqrt)":
    
            self.initial_colonize_prob = hyperparameters_epidemic["initial_colonize_prob"]

            self.gamma                 = hyperparameters_epidemic["gamma"]

            self.coef                  = hyperparameters_epidemic["coef"]

            self.translation           = hyperparameters_epidemic["translation"]
            self.periodicity           = hyperparameters_epidemic["periodicity"]

            self.sensitivity           = hyperparameters_epidemic["sensitivity"]
            self.specificity           = hyperparameters_epidemic["specificity"]    

        if self.epidemic_type == "3 parameters within-across households (all proportions) and fixed recovery (sqrt)":
    
            self.initial_colonize_prob = hyperparameters_epidemic["initial_colonize_prob"]

            self.gamma                 = hyperparameters_epidemic["gamma"]

            self.translation           = hyperparameters_epidemic["translation"]
            self.periodicity           = hyperparameters_epidemic["periodicity"]

            self.sensitivity           = hyperparameters_epidemic["sensitivity"]
            self.specificity           = hyperparameters_epidemic["specificity"]    

        if self.epidemic_type == "fixed effect, 3 parameters within-across households (all proportions) and fixed recovery, seasonality, individuals coefficients (sqrt)":
    
            self.initial_colonize_prob = hyperparameters_epidemic["initial_colonize_prob"]

            self.gamma                 = hyperparameters_epidemic["gamma"]

            self.individual_coef       = hyperparameters_epidemic["individual_coef"]

            self.coef                  = hyperparameters_epidemic["coef"]

            self.translation           = hyperparameters_epidemic["translation"]
            self.periodicity           = hyperparameters_epidemic["periodicity"]

            self.sensitivity           = hyperparameters_epidemic["sensitivity"]
            self.specificity           = hyperparameters_epidemic["specificity"]   

        if self.epidemic_type == "fixed effect, 3 parameters within-across households (all proportions) and fixed recovery, seasonality (sqrt)":
    
            self.initial_colonize_prob = hyperparameters_epidemic["initial_colonize_prob"]

            self.gamma                 = hyperparameters_epidemic["gamma"]

            self.coef                  = hyperparameters_epidemic["coef"]

            self.translation           = hyperparameters_epidemic["translation"]
            self.periodicity           = hyperparameters_epidemic["periodicity"]

            self.sensitivity           = hyperparameters_epidemic["sensitivity"]
            self.specificity           = hyperparameters_epidemic["specificity"]   

        if self.epidemic_type == "fixed effect, 3 parameters within-across households and fixed recovery, seasonality, individuals coefficients (sqrt)":

            self.initial_colonize_prob = hyperparameters_epidemic["initial_colonize_prob"]

            self.gamma                 = hyperparameters_epidemic["gamma"]

            self.individual_coef       = hyperparameters_epidemic["individual_coef"]

            self.coef                  = hyperparameters_epidemic["coef"]

            self.translation           = hyperparameters_epidemic["translation"]
            self.periodicity           = hyperparameters_epidemic["periodicity"]

            self.sensitivity           = hyperparameters_epidemic["sensitivity"]
            self.specificity           = hyperparameters_epidemic["specificity"] 

        if self.epidemic_type == "fixed effect, 3 parameters within-across households (all proportions) and fixed recovery, seasonality, individuals coefficients (sq)":
    
            self.initial_colonize_prob = hyperparameters_epidemic["initial_colonize_prob"]

            self.gamma                 = hyperparameters_epidemic["gamma"]

            self.individual_coef       = hyperparameters_epidemic["individual_coef"]

            self.coef                  = hyperparameters_epidemic["coef"]

            self.translation           = hyperparameters_epidemic["translation"]
            self.periodicity           = hyperparameters_epidemic["periodicity"]

            self.sensitivity           = hyperparameters_epidemic["sensitivity"]
            self.specificity           = hyperparameters_epidemic["specificity"]   

        if self.epidemic_type == "fixed effect, 3 parameters within-across households (all proportions) and fixed recovery, seasonality (sq)":
    
            self.initial_colonize_prob = hyperparameters_epidemic["initial_colonize_prob"]

            self.gamma                 = hyperparameters_epidemic["gamma"]

            self.coef                  = hyperparameters_epidemic["coef"]

            self.translation           = hyperparameters_epidemic["translation"]
            self.periodicity           = hyperparameters_epidemic["periodicity"]

            self.sensitivity           = hyperparameters_epidemic["sensitivity"]
            self.specificity           = hyperparameters_epidemic["specificity"]   

        if self.epidemic_type == "fixed effect, 3 parameters within-across households and fixed recovery, seasonality, individuals coefficients (sq)":

            self.initial_colonize_prob = hyperparameters_epidemic["initial_colonize_prob"]

            self.gamma                 = hyperparameters_epidemic["gamma"]

            self.individual_coef       = hyperparameters_epidemic["individual_coef"]

            self.coef                  = hyperparameters_epidemic["coef"]

            self.translation           = hyperparameters_epidemic["translation"]
            self.periodicity           = hyperparameters_epidemic["periodicity"]

            self.sensitivity           = hyperparameters_epidemic["sensitivity"]
            self.specificity           = hyperparameters_epidemic["specificity"]   

    ################################################
    # initialize the colonization
    def proposal_epidemic_0(self, Y_0):    

        if Y_0.shape[1]!=1:
            print("Y_0 should be a column vector")

        if self.epidemic_type == "3 parameters within-across households and fixed recovery":
            P_0 = tf.ones((self.households_individuals.n_individuals, 1))*self.initial_colonize_prob

            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_0), dtype = tf.float32)

            Y_0_without_nan = replacenan(Y_0)

            logit_P_0_0 = (1-P_0)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_0_without_nan)*tf.math.pow(1-self.specificity,     Y_0_without_nan)*(1 - P_0)*(detected_individuals_t)
            logit_P_0_1 =   (P_0)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_0_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_0_without_nan)*(    P_0)*(detected_individuals_t)

            proposal_1 = logit_P_0_1/(logit_P_0_0 + logit_P_0_1)

            C_0p = tf.transpose(tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample(self.n_particles)[:,:,0])

            W_0p = (logit_P_0_0 + logit_P_0_1)*tf.ones((C_0p.shape))

            del P_0, detected_individuals_t, Y_0_without_nan, logit_P_0_0, logit_P_0_1, proposal_1
            
        if self.epidemic_type == "3 parameters within-across households and fixed recovery (sqrt)":
            P_0 = tf.ones((self.households_individuals.n_individuals, 1))*self.initial_colonize_prob

            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_0), dtype = tf.float32)

            Y_0_without_nan = replacenan(Y_0)

            logit_P_0_0 = (1-P_0)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_0_without_nan)*tf.math.pow(1-self.specificity,     Y_0_without_nan)*(1 - P_0)*(detected_individuals_t)
            logit_P_0_1 =   (P_0)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_0_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_0_without_nan)*(    P_0)*(detected_individuals_t)

            proposal_1 = logit_P_0_1/(logit_P_0_0 + logit_P_0_1)

            C_0p = tf.transpose(tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample(self.n_particles)[:,:,0])

            W_0p = (logit_P_0_0 + logit_P_0_1)*tf.ones((C_0p.shape))

            del P_0, detected_individuals_t, Y_0_without_nan, logit_P_0_0, logit_P_0_1, proposal_1
            
        if self.epidemic_type == "3 parameters within-across households (no prop) and fixed recovery":
            P_0 = tf.ones((self.households_individuals.n_individuals, 1))*self.initial_colonize_prob

            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_0), dtype = tf.float32)

            Y_0_without_nan = replacenan(Y_0)

            logit_P_0_0 = (1-P_0)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_0_without_nan)*tf.math.pow(1-self.specificity,     Y_0_without_nan)*(1 - P_0)*(detected_individuals_t)
            logit_P_0_1 =   (P_0)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_0_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_0_without_nan)*(    P_0)*(detected_individuals_t)

            proposal_1 = logit_P_0_1/(logit_P_0_0 + logit_P_0_1)

            C_0p = tf.transpose(tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample(self.n_particles)[:,:,0])

            W_0p = (logit_P_0_0 + logit_P_0_1)*tf.ones((C_0p.shape))

            del P_0, detected_individuals_t, Y_0_without_nan, logit_P_0_0, logit_P_0_1, proposal_1
            
        if self.epidemic_type == "3 parameters within-across households (no prop) and fixed recovery (sqrt)":
            P_0 = tf.ones((self.households_individuals.n_individuals, 1))*self.initial_colonize_prob

            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_0), dtype = tf.float32)

            Y_0_without_nan = replacenan(Y_0)

            logit_P_0_0 = (1-P_0)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_0_without_nan)*tf.math.pow(1-self.specificity,     Y_0_without_nan)*(1 - P_0)*(detected_individuals_t)
            logit_P_0_1 =   (P_0)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_0_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_0_without_nan)*(    P_0)*(detected_individuals_t)

            proposal_1 = logit_P_0_1/(logit_P_0_0 + logit_P_0_1)

            C_0p = tf.transpose(tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample(self.n_particles)[:,:,0])

            W_0p = (logit_P_0_0 + logit_P_0_1)*tf.ones((C_0p.shape))

            del P_0, detected_individuals_t, Y_0_without_nan, logit_P_0_0, logit_P_0_1, proposal_1
                        
        if self.epidemic_type == "3 parameters within-across households and fixed recovery, seasonality (sqrt)":
            P_0 = tf.ones((self.households_individuals.n_individuals, 1))*self.initial_colonize_prob

            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_0), dtype = tf.float32)

            Y_0_without_nan = replacenan(Y_0)

            logit_P_0_0 = (1-P_0)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_0_without_nan)*tf.math.pow(1-self.specificity,     Y_0_without_nan)*(1 - P_0)*(detected_individuals_t)
            logit_P_0_1 =   (P_0)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_0_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_0_without_nan)*(    P_0)*(detected_individuals_t)

            proposal_1 = logit_P_0_1/(logit_P_0_0 + logit_P_0_1)

            C_0p = tf.transpose(tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample(self.n_particles)[:,:,0])

            W_0p = (logit_P_0_0 + logit_P_0_1)*tf.ones((C_0p.shape))

            del P_0, detected_individuals_t, Y_0_without_nan, logit_P_0_0, logit_P_0_1, proposal_1

        if self.epidemic_type == "3 parameters within-across households and fixed recovery, seasonality, individuals coefficients (sqrt)":
            P_0 = tf.ones((self.households_individuals.n_individuals, 1))*self.initial_colonize_prob

            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_0), dtype = tf.float32)

            Y_0_without_nan = replacenan(Y_0)

            logit_P_0_0 = (1-P_0)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_0_without_nan)*tf.math.pow(1-self.specificity,     Y_0_without_nan)*(1 - P_0)*(detected_individuals_t)
            logit_P_0_1 =   (P_0)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_0_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_0_without_nan)*(    P_0)*(detected_individuals_t)

            proposal_1 = logit_P_0_1/(logit_P_0_0 + logit_P_0_1)

            C_0p = tf.transpose(tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample(self.n_particles)[:,:,0])

            W_0p = (logit_P_0_0 + logit_P_0_1)*tf.ones((C_0p.shape))

            del P_0, detected_individuals_t, Y_0_without_nan, logit_P_0_0, logit_P_0_1, proposal_1

        if self.epidemic_type == "3 parameters within-across households (all proportions) and fixed recovery, seasonality, individuals coefficients (sqrt)":
            P_0 = tf.ones((self.households_individuals.n_individuals, 1))*self.initial_colonize_prob

            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_0), dtype = tf.float32)

            Y_0_without_nan = replacenan(Y_0)

            logit_P_0_0 = (1-P_0)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_0_without_nan)*tf.math.pow(1-self.specificity,     Y_0_without_nan)*(1 - P_0)*(detected_individuals_t)
            logit_P_0_1 =   (P_0)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_0_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_0_without_nan)*(    P_0)*(detected_individuals_t)

            proposal_1 = logit_P_0_1/(logit_P_0_0 + logit_P_0_1)

            C_0p = tf.transpose(tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample(self.n_particles)[:,:,0])

            W_0p = (logit_P_0_0 + logit_P_0_1)*tf.ones((C_0p.shape))

            del P_0, detected_individuals_t, Y_0_without_nan, logit_P_0_0, logit_P_0_1, proposal_1    

        if self.epidemic_type == "3 parameters within-across households (all proportions) and fixed recovery, seasonality (sqrt)":
            P_0 = tf.ones((self.households_individuals.n_individuals, 1))*self.initial_colonize_prob

            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_0), dtype = tf.float32)

            Y_0_without_nan = replacenan(Y_0)

            logit_P_0_0 = (1-P_0)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_0_without_nan)*tf.math.pow(1-self.specificity,     Y_0_without_nan)*(1 - P_0)*(detected_individuals_t)
            logit_P_0_1 =   (P_0)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_0_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_0_without_nan)*(    P_0)*(detected_individuals_t)

            proposal_1 = logit_P_0_1/(logit_P_0_0 + logit_P_0_1)

            C_0p = tf.transpose(tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample(self.n_particles)[:,:,0])

            W_0p = (logit_P_0_0 + logit_P_0_1)*tf.ones((C_0p.shape))

            del P_0, detected_individuals_t, Y_0_without_nan, logit_P_0_0, logit_P_0_1, proposal_1    

        if self.epidemic_type == "3 parameters within-across households (all proportions) and fixed recovery (sqrt)":
            P_0 = tf.ones((self.households_individuals.n_individuals, 1))*self.initial_colonize_prob

            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_0), dtype = tf.float32)

            Y_0_without_nan = replacenan(Y_0)

            logit_P_0_0 = (1-P_0)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_0_without_nan)*tf.math.pow(1-self.specificity,     Y_0_without_nan)*(1 - P_0)*(detected_individuals_t)
            logit_P_0_1 =   (P_0)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_0_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_0_without_nan)*(    P_0)*(detected_individuals_t)

            proposal_1 = logit_P_0_1/(logit_P_0_0 + logit_P_0_1)

            C_0p = tf.transpose(tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample(self.n_particles)[:,:,0])

            W_0p = (logit_P_0_0 + logit_P_0_1)*tf.ones((C_0p.shape))

            del P_0, detected_individuals_t, Y_0_without_nan, logit_P_0_0, logit_P_0_1, proposal_1    

        if self.epidemic_type == "fixed effect, 3 parameters within-across households (all proportions) and fixed recovery, seasonality, individuals coefficients (sqrt)":
            P_0 = tf.ones((self.households_individuals.n_individuals, 1))*self.initial_colonize_prob

            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_0), dtype = tf.float32)

            Y_0_without_nan = replacenan(Y_0)

            logit_P_0_0 = (1-P_0)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_0_without_nan)*tf.math.pow(1-self.specificity,     Y_0_without_nan)*(1 - P_0)*(detected_individuals_t)
            logit_P_0_1 =   (P_0)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_0_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_0_without_nan)*(    P_0)*(detected_individuals_t)

            proposal_1 = logit_P_0_1/(logit_P_0_0 + logit_P_0_1)

            C_0p = tf.transpose(tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample(self.n_particles)[:,:,0])

            W_0p = (logit_P_0_0 + logit_P_0_1)*tf.ones((C_0p.shape))

            del P_0, detected_individuals_t, Y_0_without_nan, logit_P_0_0, logit_P_0_1, proposal_1    

        if self.epidemic_type == "fixed effect, 3 parameters within-across households (all proportions) and fixed recovery, seasonality (sqrt)":
            P_0 = tf.ones((self.households_individuals.n_individuals, 1))*self.initial_colonize_prob

            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_0), dtype = tf.float32)

            Y_0_without_nan = replacenan(Y_0)

            logit_P_0_0 = (1-P_0)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_0_without_nan)*tf.math.pow(1-self.specificity,     Y_0_without_nan)*(1 - P_0)*(detected_individuals_t)
            logit_P_0_1 =   (P_0)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_0_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_0_without_nan)*(    P_0)*(detected_individuals_t)

            proposal_1 = logit_P_0_1/(logit_P_0_0 + logit_P_0_1)

            C_0p = tf.transpose(tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample(self.n_particles)[:,:,0])

            W_0p = (logit_P_0_0 + logit_P_0_1)*tf.ones((C_0p.shape))

            del P_0, detected_individuals_t, Y_0_without_nan, logit_P_0_0, logit_P_0_1, proposal_1     

        if self.epidemic_type == "fixed effect, 3 parameters within-across households and fixed recovery, seasonality, individuals coefficients (sqrt)":
            P_0 = tf.ones((self.households_individuals.n_individuals, 1))*self.initial_colonize_prob

            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_0), dtype = tf.float32)

            Y_0_without_nan = replacenan(Y_0)

            logit_P_0_0 = (1-P_0)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_0_without_nan)*tf.math.pow(1-self.specificity,     Y_0_without_nan)*(1 - P_0)*(detected_individuals_t)
            logit_P_0_1 =   (P_0)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_0_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_0_without_nan)*(    P_0)*(detected_individuals_t)

            proposal_1 = logit_P_0_1/(logit_P_0_0 + logit_P_0_1)

            C_0p = tf.transpose(tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample(self.n_particles)[:,:,0])

            W_0p = (logit_P_0_0 + logit_P_0_1)*tf.ones((C_0p.shape))

            del P_0, detected_individuals_t, Y_0_without_nan, logit_P_0_0, logit_P_0_1, proposal_1
            
        if self.epidemic_type == "fixed effect, 3 parameters within-across households (all proportions) and fixed recovery, seasonality, individuals coefficients (sq)":
            P_0 = tf.ones((self.households_individuals.n_individuals, 1))*self.initial_colonize_prob

            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_0), dtype = tf.float32)

            Y_0_without_nan = replacenan(Y_0)

            logit_P_0_0 = (1-P_0)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_0_without_nan)*tf.math.pow(1-self.specificity,     Y_0_without_nan)*(1 - P_0)*(detected_individuals_t)
            logit_P_0_1 =   (P_0)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_0_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_0_without_nan)*(    P_0)*(detected_individuals_t)

            proposal_1 = logit_P_0_1/(logit_P_0_0 + logit_P_0_1)

            C_0p = tf.transpose(tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample(self.n_particles)[:,:,0])

            W_0p = (logit_P_0_0 + logit_P_0_1)*tf.ones((C_0p.shape))

            del P_0, detected_individuals_t, Y_0_without_nan, logit_P_0_0, logit_P_0_1, proposal_1    

        if self.epidemic_type == "fixed effect, 3 parameters within-across households (all proportions) and fixed recovery, seasonality (sq)":
            P_0 = tf.ones((self.households_individuals.n_individuals, 1))*self.initial_colonize_prob

            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_0), dtype = tf.float32)

            Y_0_without_nan = replacenan(Y_0)

            logit_P_0_0 = (1-P_0)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_0_without_nan)*tf.math.pow(1-self.specificity,     Y_0_without_nan)*(1 - P_0)*(detected_individuals_t)
            logit_P_0_1 =   (P_0)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_0_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_0_without_nan)*(    P_0)*(detected_individuals_t)

            proposal_1 = logit_P_0_1/(logit_P_0_0 + logit_P_0_1)

            C_0p = tf.transpose(tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample(self.n_particles)[:,:,0])

            W_0p = (logit_P_0_0 + logit_P_0_1)*tf.ones((C_0p.shape))

            del P_0, detected_individuals_t, Y_0_without_nan, logit_P_0_0, logit_P_0_1, proposal_1     

        if self.epidemic_type == "fixed effect, 3 parameters within-across households and fixed recovery, seasonality, individuals coefficients (sq)":
            P_0 = tf.ones((self.households_individuals.n_individuals, 1))*self.initial_colonize_prob

            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_0), dtype = tf.float32)

            Y_0_without_nan = replacenan(Y_0)

            logit_P_0_0 = (1-P_0)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_0_without_nan)*tf.math.pow(1-self.specificity,     Y_0_without_nan)*(1 - P_0)*(detected_individuals_t)
            logit_P_0_1 =   (P_0)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_0_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_0_without_nan)*(    P_0)*(detected_individuals_t)

            proposal_1 = logit_P_0_1/(logit_P_0_0 + logit_P_0_1)

            C_0p = tf.transpose(tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample(self.n_particles)[:,:,0])

            W_0p = (logit_P_0_0 + logit_P_0_1)*tf.ones((C_0p.shape))

            del P_0, detected_individuals_t, Y_0_without_nan, logit_P_0_0, logit_P_0_1, proposal_1

        return C_0p, W_0p

        

    ################################################
    # one step proposal
    def proposal_epidemic_t(self, t, theta, C_tp, Y_t, delta_t):

        if Y_t.shape[1]!=1 and len(theta.shape)!=1 and len(C_tp.shape)!=2:
            print("Invalid input")

        if self.epidemic_type == "3 parameters within-across households and fixed recovery":
            beta_1          = tf.exp(theta[0])
            beta_2          = tf.exp(theta[1])
            phi             = tf.exp(theta[2])
            individual_coef = tf.reshape(theta[3:-1], shape = (theta[3:-1].shape[0], 1) )
            coef            = theta[-1]

            season = (1+coef*np.cos(self.periodicity*t+self.translation))
        
            # compute how infected in household contribute to infection of each individual in the same household: L_{H,I}^T L_{H,I} C_tp
            infected_per_household        = tf.sparse.sparse_dense_matmul(self.households_individuals.loc_H_I, C_tp)
            infection_rate_from_household = beta_1*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, infected_per_household, adjoint_a = True)

            # compute how other households contribute to infect individuals L_{H,I}^T D L_{H,I} C_tp 
            # (D should esclude the diagonal to avoid repeating the per household factor)
            distance_matrix                             = tf.exp(-self.distances/(2*phi*phi))-tf.eye(self.distances.shape[0])
            weighted_infection_per_household_prevalence = tf.tensordot( distance_matrix, infected_per_household/self.households_individuals.households_size, axes = 1)
            infection_rate_across_household             = beta_2*season*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, 
                                                                                                       weighted_infection_per_household_prevalence, adjoint_a = True)

            # individuals' covariates
            infection_rate_from_covariates = tf.exp( tf.tensordot(self.households_individuals.individuals, tf.reshape(individual_coef, shape = (individual_coef.shape[0], 1)), axes = 1) )

            # infection probability
            prob_lambda = (1- tf.exp(-delta_t*infection_rate_from_covariates*(infection_rate_from_household + infection_rate_across_household)))*(1-C_tp)
            # recovery probability
            prob_gamma  = tf.exp(-delta_t*self.gamma)*C_tp
            P_t = prob_lambda + prob_gamma

            # among the detected individuals select the ones that at current time have been detected
            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_t), dtype = tf.float32)

            # remove the nans, detected_individuals_t will do the job
            Y_t_without_nan = replacenan(Y_t)

            # create the logits
            logit_P_t_0 = (1-P_t)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_t_without_nan)*tf.math.pow(1-self.specificity,     Y_t_without_nan)*(1 - P_t)*(detected_individuals_t)
            logit_P_t_1 =   (P_t)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_t_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_t_without_nan)*(    P_t)*(detected_individuals_t)

            proposal_1 = logit_P_t_1/(logit_P_t_0 + logit_P_t_1) 
            W_0p       = (logit_P_t_0 + logit_P_t_1) 

            # sample the epidemic
            C_tp1p = tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample()   

            del infected_per_household, infection_rate_from_household, distance_matrix, 
            del weighted_infection_per_household_prevalence, infection_rate_across_household, 
            del infection_rate_from_covariates, prob_lambda, prob_gamma, P_t, 
            del detected_individuals_t, Y_t_without_nan, logit_P_t_0, logit_P_t_1, proposal_1,


        if self.epidemic_type == "3 parameters within-across households and fixed recovery (sqrt)":
            beta_1          = tf.exp(theta[0])
            beta_2          = tf.exp(theta[1])
            phi             = tf.exp(theta[2])
            individual_coef = tf.reshape(theta[3:-1], shape = (theta[3:-1].shape[0], 1) )
            coef            = theta[-1]

            season = (1+coef*np.cos(self.periodicity*t+self.translation))
        
            # compute how infected in household contribute to infection of each individual in the same household: L_{H,I}^T L_{H,I} C_tp
            infected_per_household        = tf.sparse.sparse_dense_matmul(self.households_individuals.loc_H_I, C_tp)
            infection_rate_from_household = beta_1*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, infected_per_household, adjoint_a = True)

            # compute how other households contribute to infect individuals L_{H,I}^T D L_{H,I} C_tp 
            # (D should esclude the diagonal to avoid repeating the per household factor)
            distance_matrix                             = tf.exp( -tf.math.sqrt( self.distances/(phi*phi) ) )-tf.eye(self.distances.shape[0])
            weighted_infection_per_household_prevalence = tf.tensordot( distance_matrix, infected_per_household/self.households_individuals.households_size, axes = 1)
            infection_rate_across_household             = beta_2*season*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, 
                                                                                                       weighted_infection_per_household_prevalence, adjoint_a = True)

            # individuals' covariates
            infection_rate_from_covariates = tf.exp( tf.tensordot(self.households_individuals.individuals, tf.reshape(individual_coef, shape = (individual_coef.shape[0], 1)), axes = 1) )

            # infection probability
            prob_lambda = (1- tf.exp(-delta_t*infection_rate_from_covariates*(infection_rate_from_household + infection_rate_across_household)))*(1-C_tp)
            # recovery probability
            prob_gamma  = tf.exp(-delta_t*self.gamma)*C_tp
            P_t = prob_lambda + prob_gamma

            # among the detected individuals select the ones that at current time have been detected
            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_t), dtype = tf.float32)

            # remove the nans, detected_individuals_t will do the job
            Y_t_without_nan = replacenan(Y_t)

            # create the logits
            logit_P_t_0 = (1-P_t)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_t_without_nan)*tf.math.pow(1-self.specificity,     Y_t_without_nan)*(1 - P_t)*(detected_individuals_t)
            logit_P_t_1 =   (P_t)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_t_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_t_without_nan)*(    P_t)*(detected_individuals_t)

            proposal_1 = logit_P_t_1/(logit_P_t_0 + logit_P_t_1) 
            W_0p       = (logit_P_t_0 + logit_P_t_1) 

            # sample the epidemic
            C_tp1p = tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample()   

            del infected_per_household, infection_rate_from_household, distance_matrix, 
            del weighted_infection_per_household_prevalence, infection_rate_across_household, 
            del infection_rate_from_covariates, prob_lambda, prob_gamma, P_t, 
            del detected_individuals_t, Y_t_without_nan, logit_P_t_0, logit_P_t_1, proposal_1,


        if self.epidemic_type == "3 parameters within-across households (no prop) and fixed recovery":
            beta_1          = tf.exp(theta[0])
            beta_2          = tf.exp(theta[1])
            phi             = tf.exp(theta[2])
            individual_coef = tf.reshape(theta[3:-1], shape = (theta[3:-1].shape[0], 1) )
            coef            = theta[-1]

            season = (1+coef*np.cos(self.periodicity*t+self.translation))
        
            # compute how infected in household contribute to infection of each individual in the same household: L_{H,I}^T L_{H,I} C_tp
            infected_per_household        = tf.sparse.sparse_dense_matmul(self.households_individuals.loc_H_I, C_tp)
            infection_rate_from_household = beta_1*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, infected_per_household, adjoint_a = True)

            # compute how other households contribute to infect individuals L_{H,I}^T D L_{H,I} C_tp 
            # (D should esclude the diagonal to avoid repeating the per household factor)
            distance_matrix                             = tf.exp(-self.distances/(2*phi*phi))-tf.eye(self.distances.shape[0])
            weighted_infection_per_household_prevalence = tf.tensordot( distance_matrix, infected_per_household, axes = 1)
            infection_rate_across_household             = beta_2*season*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, 
                                                                                                       weighted_infection_per_household_prevalence, adjoint_a = True)

            # individuals' covariates
            infection_rate_from_covariates = tf.exp( tf.tensordot(self.households_individuals.individuals, tf.reshape(individual_coef, shape = (individual_coef.shape[0], 1)), axes = 1) )

            # infection probability
            prob_lambda = (1- tf.exp(-delta_t*infection_rate_from_covariates*(infection_rate_from_household + infection_rate_across_household)))*(1-C_tp)
            # recovery probability
            prob_gamma  = tf.exp(-delta_t*self.gamma)*C_tp
            P_t = prob_lambda + prob_gamma

            # among the detected individuals select the ones that at current time have been detected
            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_t), dtype = tf.float32)

            # remove the nans, detected_individuals_t will do the job
            Y_t_without_nan = replacenan(Y_t)

            # create the logits
            logit_P_t_0 = (1-P_t)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_t_without_nan)*tf.math.pow(1-self.specificity,     Y_t_without_nan)*(1 - P_t)*(detected_individuals_t)
            logit_P_t_1 =   (P_t)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_t_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_t_without_nan)*(    P_t)*(detected_individuals_t)

            proposal_1 = logit_P_t_1/(logit_P_t_0 + logit_P_t_1) 
            W_0p       = (logit_P_t_0 + logit_P_t_1) 

            # sample the epidemic
            C_tp1p = tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample()   

            del infected_per_household, infection_rate_from_household, distance_matrix, 
            del weighted_infection_per_household_prevalence, infection_rate_across_household, 
            del infection_rate_from_covariates, prob_lambda, prob_gamma, P_t, 
            del detected_individuals_t, Y_t_without_nan, logit_P_t_0, logit_P_t_1, proposal_1,


        if self.epidemic_type == "3 parameters within-across households (no prop) and fixed recovery (sqrt)":
            beta_1          = tf.exp(theta[0])
            beta_2          = tf.exp(theta[1])
            phi             = tf.exp(theta[2])
            individual_coef = tf.reshape(theta[3:-1], shape = (theta[3:-1].shape[0], 1) )
            coef            = theta[-1]

            season = (1+coef*np.cos(self.periodicity*t+self.translation))
        
            # compute how infected in household contribute to infection of each individual in the same household: L_{H,I}^T L_{H,I} C_tp
            infected_per_household        = tf.sparse.sparse_dense_matmul(self.households_individuals.loc_H_I, C_tp)
            infection_rate_from_household = beta_1*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, infected_per_household, adjoint_a = True)

            # compute how other households contribute to infect individuals L_{H,I}^T D L_{H,I} C_tp 
            # (D should esclude the diagonal to avoid repeating the per household factor)
            distance_matrix                             = tf.exp( -tf.math.sqrt( self.distances/(phi*phi) ) )-tf.eye(self.distances.shape[0])
            weighted_infection_per_household_prevalence = tf.tensordot( distance_matrix, infected_per_household, axes = 1)
            infection_rate_across_household             = beta_2*season*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, 
                                                                                                       weighted_infection_per_household_prevalence, adjoint_a = True)

            # individuals' covariates
            infection_rate_from_covariates = tf.exp( tf.tensordot(self.households_individuals.individuals, tf.reshape(individual_coef, shape = (individual_coef.shape[0], 1)), axes = 1) )

            # infection probability
            prob_lambda = (1- tf.exp(-delta_t*infection_rate_from_covariates*(infection_rate_from_household + infection_rate_across_household)))*(1-C_tp)
            # recovery probability
            prob_gamma  = tf.exp(-delta_t*self.gamma)*C_tp
            P_t = prob_lambda + prob_gamma

            # among the detected individuals select the ones that at current time have been detected
            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_t), dtype = tf.float32)

            # remove the nans, detected_individuals_t will do the job
            Y_t_without_nan = replacenan(Y_t)

            # create the logits
            logit_P_t_0 = (1-P_t)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_t_without_nan)*tf.math.pow(1-self.specificity,     Y_t_without_nan)*(1 - P_t)*(detected_individuals_t)
            logit_P_t_1 =   (P_t)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_t_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_t_without_nan)*(    P_t)*(detected_individuals_t)

            proposal_1 = logit_P_t_1/(logit_P_t_0 + logit_P_t_1) 
            W_0p       = (logit_P_t_0 + logit_P_t_1) 

            # sample the epidemic
            C_tp1p = tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample()   

            del infected_per_household, infection_rate_from_household, distance_matrix, 
            del weighted_infection_per_household_prevalence, infection_rate_across_household, 
            del infection_rate_from_covariates, prob_lambda, prob_gamma, P_t, 
            del detected_individuals_t, Y_t_without_nan, logit_P_t_0, logit_P_t_1, proposal_1,
            

        if self.epidemic_type == "3 parameters within-across households and fixed recovery, seasonality (sqrt)":
            beta_1          = tf.exp(theta[0])
            beta_2          = tf.exp(theta[1])
            phi             = tf.exp(theta[2])
            individual_coef = tf.reshape(theta[3:], shape = (theta[3:].shape[0], 1) )

            season = (1+self.coef*np.cos(self.periodicity*t+self.translation))
        
            # compute how infected in household contribute to infection of each individual in the same household: L_{H,I}^T L_{H,I} C_tp
            infected_per_household        = tf.sparse.sparse_dense_matmul(self.households_individuals.loc_H_I, C_tp)
            infection_rate_from_household = beta_1*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, infected_per_household, adjoint_a = True)

            # compute how other households contribute to infect individuals L_{H,I}^T D L_{H,I} C_tp 
            # (D should esclude the diagonal to avoid repeating the per household factor)
            distance_matrix                             = tf.exp( -tf.math.sqrt( self.distances/(phi*phi) ) )-tf.eye(self.distances.shape[0])
            weighted_infection_per_household_prevalence = tf.tensordot( distance_matrix, infected_per_household/self.households_individuals.households_size, axes = 1)
            infection_rate_across_household             = beta_2*season*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, 
                                                                                                       weighted_infection_per_household_prevalence, adjoint_a = True)

            # individuals' covariates
            infection_rate_from_covariates = tf.exp( tf.tensordot(self.households_individuals.individuals, tf.reshape(individual_coef, shape = (individual_coef.shape[0], 1)), axes = 1) )

            # infection probability
            prob_lambda = (1- tf.exp(-delta_t*infection_rate_from_covariates*(infection_rate_from_household + infection_rate_across_household)))*(1-C_tp)
            # recovery probability
            prob_gamma  = tf.exp(-delta_t*self.gamma)*C_tp
            P_t = prob_lambda + prob_gamma

            # among the detected individuals select the ones that at current time have been detected
            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_t), dtype = tf.float32)

            # remove the nans, detected_individuals_t will do the job
            Y_t_without_nan = replacenan(Y_t)

            # create the logits
            logit_P_t_0 = (1-P_t)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_t_without_nan)*tf.math.pow(1-self.specificity,     Y_t_without_nan)*(1 - P_t)*(detected_individuals_t)
            logit_P_t_1 =   (P_t)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_t_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_t_without_nan)*(    P_t)*(detected_individuals_t)

            proposal_1 = logit_P_t_1/(logit_P_t_0 + logit_P_t_1) 
            W_0p       = (logit_P_t_0 + logit_P_t_1) 

            # sample the epidemic
            C_tp1p = tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample()   

            del infected_per_household, infection_rate_from_household, distance_matrix, 
            del weighted_infection_per_household_prevalence, infection_rate_across_household, 
            del infection_rate_from_covariates, prob_lambda, prob_gamma, P_t, 
            del detected_individuals_t, Y_t_without_nan, logit_P_t_0, logit_P_t_1, proposal_1,

        if self.epidemic_type == "3 parameters within-across households and fixed recovery, seasonality, individuals coefficients (sqrt)":
            beta_1          = tf.exp(theta[0])
            beta_2          = tf.exp(theta[1])
            phi             = tf.exp(theta[2])
            individual_coef = self.individual_coef

            season = (1+self.coef*np.cos(self.periodicity*t+self.translation))
        
            # compute how infected in household contribute to infection of each individual in the same household: L_{H,I}^T L_{H,I} C_tp
            infected_per_household        = tf.sparse.sparse_dense_matmul(self.households_individuals.loc_H_I, C_tp)
            infection_rate_from_household = beta_1*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, infected_per_household, adjoint_a = True)

            # compute how other households contribute to infect individuals L_{H,I}^T D L_{H,I} C_tp 
            # (D should esclude the diagonal to avoid repeating the per household factor)
            distance_matrix                             = tf.exp( -tf.math.sqrt( self.distances/(phi*phi) ) )-tf.eye(self.distances.shape[0])
            weighted_infection_per_household_prevalence = tf.tensordot( distance_matrix, infected_per_household/self.households_individuals.households_size, axes = 1)
            infection_rate_across_household             = beta_2*season*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, 
                                                                                                       weighted_infection_per_household_prevalence, adjoint_a = True)

            # individuals' covariates
            infection_rate_from_covariates = tf.exp( tf.tensordot(self.households_individuals.individuals, tf.reshape(individual_coef, shape = (individual_coef.shape[0], 1)), axes = 1) )

            # infection probability
            prob_lambda = (1- tf.exp(-delta_t*infection_rate_from_covariates*(infection_rate_from_household + infection_rate_across_household)))*(1-C_tp)
            # recovery probability
            prob_gamma  = tf.exp(-delta_t*self.gamma)*C_tp
            P_t = prob_lambda + prob_gamma

            # among the detected individuals select the ones that at current time have been detected
            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_t), dtype = tf.float32)

            # remove the nans, detected_individuals_t will do the job
            Y_t_without_nan = replacenan(Y_t)

            # create the logits
            logit_P_t_0 = (1-P_t)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_t_without_nan)*tf.math.pow(1-self.specificity,     Y_t_without_nan)*(1 - P_t)*(detected_individuals_t)
            logit_P_t_1 =   (P_t)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_t_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_t_without_nan)*(    P_t)*(detected_individuals_t)

            proposal_1 = logit_P_t_1/(logit_P_t_0 + logit_P_t_1) 
            W_0p       = (logit_P_t_0 + logit_P_t_1) 

            # sample the epidemic
            C_tp1p = tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample()   

            del infected_per_household, infection_rate_from_household, distance_matrix, 
            del weighted_infection_per_household_prevalence, infection_rate_across_household, 
            del infection_rate_from_covariates, prob_lambda, prob_gamma, P_t, 
            del detected_individuals_t, Y_t_without_nan, logit_P_t_0, logit_P_t_1, proposal_1,


        if self.epidemic_type == "3 parameters within-across households (all proportions) and fixed recovery, seasonality, individuals coefficients (sqrt)":
            beta_1          = tf.exp(theta[0])
            beta_2          = tf.exp(theta[1])
            phi             = tf.exp(theta[2])
            individual_coef = self.individual_coef

            season = (1+self.coef*np.cos(self.periodicity*t+self.translation))
        
            # compute how infected in household contribute to infection of each individual in the same household: L_{H,I}^T L_{H,I} C_tp
            infected_per_household        = tf.sparse.sparse_dense_matmul(self.households_individuals.loc_H_I, C_tp)
            infection_rate_from_household = beta_1*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, infected_per_household/self.households_individuals.households_size, adjoint_a = True)

            # compute how other households contribute to infect individuals L_{H,I}^T D L_{H,I} C_tp 
            # (D should esclude the diagonal to avoid repeating the per household factor)
            distance_matrix                             = tf.exp( -tf.math.sqrt( self.distances/(phi*phi) ) )-tf.eye(self.distances.shape[0])
            weighted_infection_per_household_prevalence = tf.tensordot( distance_matrix, infected_per_household/self.households_individuals.households_size, axes = 1)
            infection_rate_across_household             = beta_2*season*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, 
                                                                                                       weighted_infection_per_household_prevalence, adjoint_a = True)

            # individuals' covariates
            infection_rate_from_covariates = tf.exp( tf.tensordot(self.households_individuals.individuals, tf.reshape(individual_coef, shape = (individual_coef.shape[0], 1)), axes = 1) )

            # infection probability
            prob_lambda = (1- tf.exp(-delta_t*infection_rate_from_covariates*(infection_rate_from_household + infection_rate_across_household)))*(1-C_tp)
            # recovery probability
            prob_gamma  = tf.exp(-delta_t*self.gamma)*C_tp
            P_t = prob_lambda + prob_gamma

            # among the detected individuals select the ones that at current time have been detected
            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_t), dtype = tf.float32)

            # remove the nans, detected_individuals_t will do the job
            Y_t_without_nan = replacenan(Y_t)

            # create the logits
            logit_P_t_0 = (1-P_t)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_t_without_nan)*tf.math.pow(1-self.specificity,     Y_t_without_nan)*(1 - P_t)*(detected_individuals_t)
            logit_P_t_1 =   (P_t)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_t_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_t_without_nan)*(    P_t)*(detected_individuals_t)

            proposal_1 = logit_P_t_1/(logit_P_t_0 + logit_P_t_1) 
            W_0p       = (logit_P_t_0 + logit_P_t_1) 

            # sample the epidemic
            C_tp1p = tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample()   

            del infected_per_household, infection_rate_from_household, distance_matrix, 
            del weighted_infection_per_household_prevalence, infection_rate_across_household, 
            del infection_rate_from_covariates, prob_lambda, prob_gamma, P_t, 
            del detected_individuals_t, Y_t_without_nan, logit_P_t_0, logit_P_t_1, proposal_1,
            
        if self.epidemic_type == "3 parameters within-across households (all proportions) and fixed recovery, seasonality (sqrt)":
            beta_1          = tf.exp(theta[0])
            beta_2          = tf.exp(theta[1])
            phi             = tf.exp(theta[2])
            individual_coef = tf.reshape(theta[3:], shape = (theta[3:].shape[0], 1) )

            season = (1+self.coef*np.cos(self.periodicity*t+self.translation))
        
            # compute how infected in household contribute to infection of each individual in the same household: L_{H,I}^T L_{H,I} C_tp
            infected_per_household        = tf.sparse.sparse_dense_matmul(self.households_individuals.loc_H_I, C_tp)
            infection_rate_from_household = beta_1*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, infected_per_household/self.households_individuals.households_size, adjoint_a = True)

            # compute how other households contribute to infect individuals L_{H,I}^T D L_{H,I} C_tp 
            # (D should esclude the diagonal to avoid repeating the per household factor)
            distance_matrix                             = tf.exp( -tf.math.sqrt( self.distances/(phi*phi) ) )-tf.eye(self.distances.shape[0])
            weighted_infection_per_household_prevalence = tf.tensordot( distance_matrix, infected_per_household/self.households_individuals.households_size, axes = 1)
            infection_rate_across_household             = beta_2*season*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, 
                                                                                                       weighted_infection_per_household_prevalence, adjoint_a = True)

            # individuals' covariates
            infection_rate_from_covariates = tf.exp( tf.tensordot(self.households_individuals.individuals, tf.reshape(individual_coef, shape = (individual_coef.shape[0], 1)), axes = 1) )

            # infection probability
            prob_lambda = (1- tf.exp(-delta_t*infection_rate_from_covariates*(infection_rate_from_household + infection_rate_across_household)))*(1-C_tp)
            # recovery probability
            prob_gamma  = tf.exp(-delta_t*self.gamma)*C_tp
            P_t = prob_lambda + prob_gamma

            # among the detected individuals select the ones that at current time have been detected
            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_t), dtype = tf.float32)

            # remove the nans, detected_individuals_t will do the job
            Y_t_without_nan = replacenan(Y_t)

            # create the logits
            logit_P_t_0 = (1-P_t)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_t_without_nan)*tf.math.pow(1-self.specificity,     Y_t_without_nan)*(1 - P_t)*(detected_individuals_t)
            logit_P_t_1 =   (P_t)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_t_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_t_without_nan)*(    P_t)*(detected_individuals_t)

            proposal_1 = logit_P_t_1/(logit_P_t_0 + logit_P_t_1) 
            W_0p       = (logit_P_t_0 + logit_P_t_1) 

            # sample the epidemic
            C_tp1p = tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample()   

            del infected_per_household, infection_rate_from_household, distance_matrix, 
            del weighted_infection_per_household_prevalence, infection_rate_across_household, 
            del infection_rate_from_covariates, prob_lambda, prob_gamma, P_t, 
            del detected_individuals_t, Y_t_without_nan, logit_P_t_0, logit_P_t_1, proposal_1,


        if self.epidemic_type == "3 parameters within-across households (all proportions) and fixed recovery (sqrt)":
            beta_1          = tf.exp(theta[0])
            beta_2          = tf.exp(theta[1])
            phi             = tf.exp(theta[2])
            individual_coef = tf.reshape(theta[3:-1], shape = (theta[3:-1].shape[0], 1) )
            coef            = theta[-1]

            season = (1+coef*np.cos(self.periodicity*t+self.translation))
        
            # compute how infected in household contribute to infection of each individual in the same household: L_{H,I}^T L_{H,I} C_tp
            infected_per_household        = tf.sparse.sparse_dense_matmul(self.households_individuals.loc_H_I, C_tp)
            infection_rate_from_household = beta_1*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, infected_per_household/self.households_individuals.households_size, adjoint_a = True)

            # compute how other households contribute to infect individuals L_{H,I}^T D L_{H,I} C_tp 
            # (D should esclude the diagonal to avoid repeating the per household factor)
            distance_matrix                             = tf.exp( -tf.math.sqrt( self.distances/(phi*phi) ) )-tf.eye(self.distances.shape[0])
            weighted_infection_per_household_prevalence = tf.tensordot( distance_matrix, infected_per_household/self.households_individuals.households_size, axes = 1)
            infection_rate_across_household             = beta_2*season*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, 
                                                                                                       weighted_infection_per_household_prevalence, adjoint_a = True)

            # individuals' covariates
            infection_rate_from_covariates = tf.exp( tf.tensordot(self.households_individuals.individuals, tf.reshape(individual_coef, shape = (individual_coef.shape[0], 1)), axes = 1) )

            # infection probability
            prob_lambda = (1- tf.exp(-delta_t*infection_rate_from_covariates*(infection_rate_from_household + infection_rate_across_household)))*(1-C_tp)
            # recovery probability
            prob_gamma  = tf.exp(-delta_t*self.gamma)*C_tp
            P_t = prob_lambda + prob_gamma

            # among the detected individuals select the ones that at current time have been detected
            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_t), dtype = tf.float32)

            # remove the nans, detected_individuals_t will do the job
            Y_t_without_nan = replacenan(Y_t)

            # create the logits
            logit_P_t_0 = (1-P_t)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_t_without_nan)*tf.math.pow(1-self.specificity,     Y_t_without_nan)*(1 - P_t)*(detected_individuals_t)
            logit_P_t_1 =   (P_t)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_t_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_t_without_nan)*(    P_t)*(detected_individuals_t)

            proposal_1 = logit_P_t_1/(logit_P_t_0 + logit_P_t_1) 
            W_0p       = (logit_P_t_0 + logit_P_t_1) 

            # sample the epidemic
            C_tp1p = tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample()   

            del infected_per_household, infection_rate_from_household, distance_matrix, 
            del weighted_infection_per_household_prevalence, infection_rate_across_household, 
            del infection_rate_from_covariates, prob_lambda, prob_gamma, P_t, 
            del detected_individuals_t, Y_t_without_nan, logit_P_t_0, logit_P_t_1, proposal_1,


        if self.epidemic_type == "fixed effect, 3 parameters within-across households (all proportions) and fixed recovery, seasonality, individuals coefficients (sqrt)":
            beta_1          = tf.exp(theta[0])
            beta_2          = tf.exp(theta[1])
            phi             = tf.exp(theta[2])
            fixed_effect    = tf.exp(theta[3])
            individual_coef = self.individual_coef

            season = (1+self.coef*np.cos(self.periodicity*t+self.translation))
        
            # compute how infected in household contribute to infection of each individual in the same household: L_{H,I}^T L_{H,I} C_tp
            infected_per_household        = tf.sparse.sparse_dense_matmul(self.households_individuals.loc_H_I, C_tp)
            infection_rate_from_household = beta_1*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, infected_per_household/self.households_individuals.households_size, adjoint_a = True)

            # compute how other households contribute to infect individuals L_{H,I}^T D L_{H,I} C_tp 
            # (D should esclude the diagonal to avoid repeating the per household factor)
            distance_matrix                             = tf.exp( -tf.math.sqrt( self.distances/(phi*phi) ) )-tf.eye(self.distances.shape[0])
            weighted_infection_per_household_prevalence = tf.tensordot( distance_matrix, infected_per_household/self.households_individuals.households_size, axes = 1)
            infection_rate_across_household             = beta_2*season*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, 
                                                                                                       weighted_infection_per_household_prevalence, adjoint_a = True)

            # individuals' covariates
            infection_rate_from_covariates = tf.exp( tf.tensordot(self.households_individuals.individuals, tf.reshape(individual_coef, shape = (individual_coef.shape[0], 1)), axes = 1) )

            # infection probability
            prob_lambda = (1- tf.exp(-delta_t*(infection_rate_from_covariates*(infection_rate_from_household + infection_rate_across_household)+ fixed_effect)))*(1-C_tp)
            # recovery probability
            prob_gamma  = tf.exp(-delta_t*self.gamma)*C_tp
            P_t = prob_lambda + prob_gamma

            # among the detected individuals select the ones that at current time have been detected
            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_t), dtype = tf.float32)

            # remove the nans, detected_individuals_t will do the job
            Y_t_without_nan = replacenan(Y_t)

            # create the logits
            logit_P_t_0 = (1-P_t)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_t_without_nan)*tf.math.pow(1-self.specificity,     Y_t_without_nan)*(1 - P_t)*(detected_individuals_t)
            logit_P_t_1 =   (P_t)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_t_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_t_without_nan)*(    P_t)*(detected_individuals_t)

            proposal_1 = logit_P_t_1/(logit_P_t_0 + logit_P_t_1) 
            W_0p       = (logit_P_t_0 + logit_P_t_1) 

            # sample the epidemic
            C_tp1p = tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample()   

            del infected_per_household, infection_rate_from_household, distance_matrix, 
            del weighted_infection_per_household_prevalence, infection_rate_across_household, 
            del infection_rate_from_covariates, prob_lambda, prob_gamma, P_t, 
            del detected_individuals_t, Y_t_without_nan, logit_P_t_0, logit_P_t_1, proposal_1,       
            



        if self.epidemic_type == "fixed effect, 3 parameters within-across households (all proportions) and fixed recovery, seasonality (sqrt)":
            beta_1          = tf.exp(theta[0])
            beta_2          = tf.exp(theta[1])
            phi             = tf.exp(theta[2])
            fixed_effect    = tf.exp(theta[3])
            individual_coef = tf.exp(theta[4:])

            season = (1+self.coef*np.cos(self.periodicity*t+self.translation))
        
            # compute how infected in household contribute to infection of each individual in the same household: L_{H,I}^T L_{H,I} C_tp
            infected_per_household        = tf.sparse.sparse_dense_matmul(self.households_individuals.loc_H_I, C_tp)
            infection_rate_from_household = beta_1*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, infected_per_household/self.households_individuals.households_size, adjoint_a = True)

            # compute how other households contribute to infect individuals L_{H,I}^T D L_{H,I} C_tp 
            # (D should esclude the diagonal to avoid repeating the per household factor)
            distance_matrix                             = tf.exp( -tf.math.sqrt( self.distances/(phi*phi) ) )-tf.eye(self.distances.shape[0])
            weighted_infection_per_household_prevalence = tf.tensordot( distance_matrix, infected_per_household/self.households_individuals.households_size, axes = 1)
            infection_rate_across_household             = beta_2*season*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, 
                                                                                                       weighted_infection_per_household_prevalence, adjoint_a = True)

            # individuals' covariates
            infection_rate_from_covariates = tf.exp( tf.tensordot(self.households_individuals.individuals, tf.reshape(individual_coef, shape = (individual_coef.shape[0], 1)), axes = 1) )

            # infection probability
            prob_lambda = (1- tf.exp(-delta_t*(infection_rate_from_covariates*(infection_rate_from_household + infection_rate_across_household)+ fixed_effect)))*(1-C_tp)
            # recovery probability
            prob_gamma  = tf.exp(-delta_t*self.gamma)*C_tp
            P_t = prob_lambda + prob_gamma

            # among the detected individuals select the ones that at current time have been detected
            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_t), dtype = tf.float32)

            # remove the nans, detected_individuals_t will do the job
            Y_t_without_nan = replacenan(Y_t)

            # create the logits
            logit_P_t_0 = (1-P_t)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_t_without_nan)*tf.math.pow(1-self.specificity,     Y_t_without_nan)*(1 - P_t)*(detected_individuals_t)
            logit_P_t_1 =   (P_t)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_t_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_t_without_nan)*(    P_t)*(detected_individuals_t)

            proposal_1 = logit_P_t_1/(logit_P_t_0 + logit_P_t_1) 
            W_0p       = (logit_P_t_0 + logit_P_t_1) 

            # sample the epidemic
            C_tp1p = tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample()   

            del infected_per_household, infection_rate_from_household, distance_matrix, 
            del weighted_infection_per_household_prevalence, infection_rate_across_household, 
            del infection_rate_from_covariates, prob_lambda, prob_gamma, P_t, 
            del detected_individuals_t, Y_t_without_nan, logit_P_t_0, logit_P_t_1, proposal_1,       
            

        if self.epidemic_type == "fixed effect, 3 parameters within-across households and fixed recovery, seasonality, individuals coefficients (sqrt)":
            beta_1          = tf.exp(theta[0])
            beta_2          = tf.exp(theta[1])
            phi             = tf.exp(theta[2])
            fixed_effect    = tf.exp(theta[3])
            individual_coef = self.individual_coef

            season = (1+self.coef*np.cos(self.periodicity*t+self.translation))
        
            # compute how infected in household contribute to infection of each individual in the same household: L_{H,I}^T L_{H,I} C_tp
            infected_per_household        = tf.sparse.sparse_dense_matmul(self.households_individuals.loc_H_I, C_tp)
            infection_rate_from_household = beta_1*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, infected_per_household, adjoint_a = True)

            # compute how other households contribute to infect individuals L_{H,I}^T D L_{H,I} C_tp 
            # (D should esclude the diagonal to avoid repeating the per household factor)
            distance_matrix                             = tf.exp( -tf.math.sqrt( self.distances/(phi*phi) ) )-tf.eye(self.distances.shape[0])
            weighted_infection_per_household_prevalence = tf.tensordot( distance_matrix, infected_per_household/self.households_individuals.households_size, axes = 1)
            infection_rate_across_household             = beta_2*season*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, 
                                                                                                       weighted_infection_per_household_prevalence, adjoint_a = True)

            # individuals' covariates
            infection_rate_from_covariates = tf.exp( tf.tensordot(self.households_individuals.individuals, tf.reshape(individual_coef, shape = (individual_coef.shape[0], 1)), axes = 1) )

            # infection probability
            prob_lambda = (1- tf.exp(-delta_t*(infection_rate_from_covariates*(infection_rate_from_household + infection_rate_across_household)+ fixed_effect)))*(1-C_tp)
            # recovery probability
            prob_gamma  = tf.exp(-delta_t*self.gamma)*C_tp
            P_t = prob_lambda + prob_gamma

            # among the detected individuals select the ones that at current time have been detected
            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_t), dtype = tf.float32)

            # remove the nans, detected_individuals_t will do the job
            Y_t_without_nan = replacenan(Y_t)

            # create the logits
            logit_P_t_0 = (1-P_t)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_t_without_nan)*tf.math.pow(1-self.specificity,     Y_t_without_nan)*(1 - P_t)*(detected_individuals_t)
            logit_P_t_1 =   (P_t)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_t_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_t_without_nan)*(    P_t)*(detected_individuals_t)

            proposal_1 = logit_P_t_1/(logit_P_t_0 + logit_P_t_1) 
            W_0p       = (logit_P_t_0 + logit_P_t_1) 

            # sample the epidemic
            C_tp1p = tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample()   

            del infected_per_household, infection_rate_from_household, distance_matrix, 
            del weighted_infection_per_household_prevalence, infection_rate_across_household, 
            del infection_rate_from_covariates, prob_lambda, prob_gamma, P_t, 
            del detected_individuals_t, Y_t_without_nan, logit_P_t_0, logit_P_t_1, proposal_1,      

            

        if self.epidemic_type == "fixed effect, 3 parameters within-across households (all proportions) and fixed recovery, seasonality, individuals coefficients (sq)":
            beta_1          = tf.exp(theta[0])
            beta_2          = tf.exp(theta[1])
            phi             = tf.exp(theta[2])
            fixed_effect    = tf.exp(theta[3])
            individual_coef = self.individual_coef

            season = (1+self.coef*np.cos(self.periodicity*t+self.translation))
        
            # compute how infected in household contribute to infection of each individual in the same household: L_{H,I}^T L_{H,I} C_tp
            infected_per_household        = tf.sparse.sparse_dense_matmul(self.households_individuals.loc_H_I, C_tp)
            infection_rate_from_household = beta_1*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, infected_per_household/self.households_individuals.households_size, adjoint_a = True)

            # compute how other households contribute to infect individuals L_{H,I}^T D L_{H,I} C_tp 
            # (D should esclude the diagonal to avoid repeating the per household factor)
            distance_matrix                             = tf.exp( -( self.distances/(phi*phi) ) )-tf.eye(self.distances.shape[0])
            weighted_infection_per_household_prevalence = tf.tensordot( distance_matrix, infected_per_household/self.households_individuals.households_size, axes = 1)
            infection_rate_across_household             = beta_2*season*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, 
                                                                                                       weighted_infection_per_household_prevalence, adjoint_a = True)

            # individuals' covariates
            infection_rate_from_covariates = tf.exp( tf.tensordot(self.households_individuals.individuals, tf.reshape(individual_coef, shape = (individual_coef.shape[0], 1)), axes = 1) )

            # infection probability
            prob_lambda = (1- tf.exp(-delta_t*(infection_rate_from_covariates*(infection_rate_from_household + infection_rate_across_household)+ fixed_effect)))*(1-C_tp)
            # recovery probability
            prob_gamma  = tf.exp(-delta_t*self.gamma)*C_tp
            P_t = prob_lambda + prob_gamma

            # among the detected individuals select the ones that at current time have been detected
            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_t), dtype = tf.float32)

            # remove the nans, detected_individuals_t will do the job
            Y_t_without_nan = replacenan(Y_t)

            # create the logits
            logit_P_t_0 = (1-P_t)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_t_without_nan)*tf.math.pow(1-self.specificity,     Y_t_without_nan)*(1 - P_t)*(detected_individuals_t)
            logit_P_t_1 =   (P_t)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_t_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_t_without_nan)*(    P_t)*(detected_individuals_t)

            proposal_1 = logit_P_t_1/(logit_P_t_0 + logit_P_t_1) 
            W_0p       = (logit_P_t_0 + logit_P_t_1) 

            # sample the epidemic
            C_tp1p = tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample()   

            del infected_per_household, infection_rate_from_household, distance_matrix, 
            del weighted_infection_per_household_prevalence, infection_rate_across_household, 
            del infection_rate_from_covariates, prob_lambda, prob_gamma, P_t, 
            del detected_individuals_t, Y_t_without_nan, logit_P_t_0, logit_P_t_1, proposal_1,       
            



        if self.epidemic_type == "fixed effect, 3 parameters within-across households (all proportions) and fixed recovery, seasonality (sq)":
            beta_1          = tf.exp(theta[0])
            beta_2          = tf.exp(theta[1])
            phi             = tf.exp(theta[2])
            fixed_effect    = tf.exp(theta[3])
            individual_coef = tf.exp(theta[4:])

            season = (1+self.coef*np.cos(self.periodicity*t+self.translation))
        
            # compute how infected in household contribute to infection of each individual in the same household: L_{H,I}^T L_{H,I} C_tp
            infected_per_household        = tf.sparse.sparse_dense_matmul(self.households_individuals.loc_H_I, C_tp)
            infection_rate_from_household = beta_1*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, infected_per_household/self.households_individuals.households_size, adjoint_a = True)

            # compute how other households contribute to infect individuals L_{H,I}^T D L_{H,I} C_tp 
            # (D should esclude the diagonal to avoid repeating the per household factor)
            distance_matrix                             = tf.exp( -( self.distances/(phi*phi) ) )-tf.eye(self.distances.shape[0])
            weighted_infection_per_household_prevalence = tf.tensordot( distance_matrix, infected_per_household/self.households_individuals.households_size, axes = 1)
            infection_rate_across_household             = beta_2*season*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, 
                                                                                                       weighted_infection_per_household_prevalence, adjoint_a = True)

            # individuals' covariates
            infection_rate_from_covariates = tf.exp( tf.tensordot(self.households_individuals.individuals, tf.reshape(individual_coef, shape = (individual_coef.shape[0], 1)), axes = 1) )

            # infection probability
            prob_lambda = (1- tf.exp(-delta_t*(infection_rate_from_covariates*(infection_rate_from_household + infection_rate_across_household)+ fixed_effect)))*(1-C_tp)
            # recovery probability
            prob_gamma  = tf.exp(-delta_t*self.gamma)*C_tp
            P_t = prob_lambda + prob_gamma

            # among the detected individuals select the ones that at current time have been detected
            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_t), dtype = tf.float32)

            # remove the nans, detected_individuals_t will do the job
            Y_t_without_nan = replacenan(Y_t)

            # create the logits
            logit_P_t_0 = (1-P_t)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_t_without_nan)*tf.math.pow(1-self.specificity,     Y_t_without_nan)*(1 - P_t)*(detected_individuals_t)
            logit_P_t_1 =   (P_t)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_t_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_t_without_nan)*(    P_t)*(detected_individuals_t)

            proposal_1 = logit_P_t_1/(logit_P_t_0 + logit_P_t_1) 
            W_0p       = (logit_P_t_0 + logit_P_t_1) 

            # sample the epidemic
            C_tp1p = tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample()   

            del infected_per_household, infection_rate_from_household, distance_matrix, 
            del weighted_infection_per_household_prevalence, infection_rate_across_household, 
            del infection_rate_from_covariates, prob_lambda, prob_gamma, P_t, 
            del detected_individuals_t, Y_t_without_nan, logit_P_t_0, logit_P_t_1, proposal_1,       
            

        if self.epidemic_type == "fixed effect, 3 parameters within-across households and fixed recovery, seasonality, individuals coefficients (sq)":
            beta_1          = tf.exp(theta[0])
            beta_2          = tf.exp(theta[1])
            phi             = tf.exp(theta[2])
            fixed_effect    = tf.exp(theta[3])
            individual_coef = self.individual_coef

            season = (1+self.coef*np.cos(self.periodicity*t+self.translation))
        
            # compute how infected in household contribute to infection of each individual in the same household: L_{H,I}^T L_{H,I} C_tp
            infected_per_household        = tf.sparse.sparse_dense_matmul(self.households_individuals.loc_H_I, C_tp)
            infection_rate_from_household = beta_1*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, infected_per_household, adjoint_a = True)

            # compute how other households contribute to infect individuals L_{H,I}^T D L_{H,I} C_tp 
            # (D should esclude the diagonal to avoid repeating the per household factor)
            distance_matrix                             = tf.exp( -( self.distances/(phi*phi) ) )-tf.eye(self.distances.shape[0])
            weighted_infection_per_household_prevalence = tf.tensordot( distance_matrix, infected_per_household/self.households_individuals.households_size, axes = 1)
            infection_rate_across_household             = beta_2*season*tf.sparse.sparse_dense_matmul( self.households_individuals.loc_H_I, 
                                                                                                       weighted_infection_per_household_prevalence, adjoint_a = True)

            # individuals' covariates
            infection_rate_from_covariates = tf.exp( tf.tensordot(self.households_individuals.individuals, tf.reshape(individual_coef, shape = (individual_coef.shape[0], 1)), axes = 1) )

            # infection probability
            prob_lambda = (1- tf.exp(-delta_t*(infection_rate_from_covariates*(infection_rate_from_household + infection_rate_across_household)+ fixed_effect)))*(1-C_tp)
            # recovery probability
            prob_gamma  = tf.exp(-delta_t*self.gamma)*C_tp
            P_t = prob_lambda + prob_gamma

            # among the detected individuals select the ones that at current time have been detected
            detected_individuals_t = 1 - tf.cast(tf.math.is_nan(Y_t), dtype = tf.float32)

            # remove the nans, detected_individuals_t will do the job
            Y_t_without_nan = replacenan(Y_t)

            # create the logits
            logit_P_t_0 = (1-P_t)*(1- detected_individuals_t) + tf.math.pow(self.specificity, 1 - Y_t_without_nan)*tf.math.pow(1-self.specificity,     Y_t_without_nan)*(1 - P_t)*(detected_individuals_t)
            logit_P_t_1 =   (P_t)*(1- detected_individuals_t) + tf.math.pow(self.sensitivity,     Y_t_without_nan)*tf.math.pow(1-self.sensitivity, 1 - Y_t_without_nan)*(    P_t)*(detected_individuals_t)

            proposal_1 = logit_P_t_1/(logit_P_t_0 + logit_P_t_1) 
            W_0p       = (logit_P_t_0 + logit_P_t_1) 

            # sample the epidemic
            C_tp1p = tfp.distributions.Bernoulli(probs = proposal_1, dtype = tf.float32).sample()   

            del infected_per_household, infection_rate_from_household, distance_matrix, 
            del weighted_infection_per_household_prevalence, infection_rate_across_household, 
            del infection_rate_from_covariates, prob_lambda, prob_gamma, P_t, 
            del detected_individuals_t, Y_t_without_nan, logit_P_t_0, logit_P_t_1, proposal_1,       

        return C_tp1p, W_0p 