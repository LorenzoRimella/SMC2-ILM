from tkinter import Y
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from utils_SMC2 import *

#######################################
# The SMC proposal class
#######################################

class simulation():

    ################################################
    # initialization
    ################################################
    def __init__( self, households_individuals, 
                        theta,
                        hyperparameters_epidemic):

        super(simulation, self).__init__()

        self.households_individuals = households_individuals
        self.distances              = self.households_individuals.households_distances

        self.theta                  = theta

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
    def proposal_epidemic_0(self):    

        P_0 = tf.ones((self.households_individuals.n_individuals, 1))*self.initial_colonize_prob

        C_0p = tfp.distributions.Bernoulli(probs = P_0, dtype = tf.float32).sample()

        sensitivity = self.sensitivity*tf.ones(C_0p.shape)
        specificity = 1-self.specificity*tf.ones(C_0p.shape)
        emission    = tf.math.pow(sensitivity, C_0p)*tf.math.pow(specificity, 1-C_0p)

        Y_0p        = tfp.distributions.Bernoulli(probs = emission, dtype = tf.float32).sample()

        return C_0p, Y_0p

        

    ################################################
    # one step proposal
    def proposal_epidemic_t(self, t, C_tp, delta_t):

        beta_1          = tf.exp(self.theta[0])
        beta_2          = tf.exp(self.theta[1])
        phi             = tf.exp(self.theta[2])
        fixed_effect    = tf.exp(self.theta[3])
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

        # sample the epidemic
        C_tp1p = tfp.distributions.Bernoulli(probs = P_t, dtype = tf.float32).sample()   

        sensitivity = self.sensitivity*tf.ones(C_tp1p.shape)
        specificity = 1-self.specificity*tf.ones(C_tp1p.shape)
        emission    = tf.math.pow(sensitivity, C_tp1p)*tf.math.pow(specificity, 1-C_tp1p)

        Y_tp1p      = tfp.distributions.Bernoulli(probs = emission, dtype = tf.float32).sample()

        return C_tp1p, Y_tp1p

    def run(self, y):

        T                   = y.shape[1]
        sampled_individuals = y.shape[0]

        C_t, Y_t    = self.proposal_epidemic_0()
        Y_t_sampled = tf.reshape(Y_t[:sampled_individuals,0]+(y[:,0]-tf.cast(y[:,0]==1, dtype = tf.float32)), (sampled_individuals, 1))

        C = C_t
        Y = Y_t
        Y_sampled = Y_t_sampled

        for t in range(1, T):

            C_tp1, Y_tp1 = self.proposal_epidemic_t(t, C_t, 1)
            Y_t_sampled  = tf.reshape(Y_tp1[:sampled_individuals,0]+(y[:,t]-tf.cast(y[:,t]==1, dtype = tf.float32)), (sampled_individuals, 1))

            C = tf.concat((C, C_tp1), axis = 1)
            Y = tf.concat((Y, Y_tp1), axis = 1)
            Y_sampled = tf.concat((Y_sampled, Y_t_sampled), axis = 1)

            C_t = C_tp1

        return C, Y, Y_sampled