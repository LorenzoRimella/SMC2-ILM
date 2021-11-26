import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from math import gamma as gamma_func, tau

import time

from utils_SMC2 import *
from proposal_epidemic import *

class Particle_filter_smoother():
    
    ################################################
    # initialization
    ################################################
    def __init__( self, households_individuals, 
                        n_particles,
                        theta,
                        hyperparameters_epidemic,
                        epidemic_type = "3 parameters within-across households and fixed recovery"):

        super(Particle_filter_smoother, self).__init__()
        
        ################################################
        # Household graph and individual detection
        self.households_individuals = households_individuals

        ################################################
        # number of particles and parameters
        self.n_particles  = n_particles
        self.n_parameters = theta.shape[0]

        self.theta        = tf.convert_to_tensor(theta, dtype = tf.float32)

        # epidemic proposal
        self.proposal_epidemic = proposal_epidemic( households_individuals, 
                                                    n_particles,
                                                    hyperparameters_epidemic,
                                                    epidemic_type)


    ################################################
    # compute the weights
    def compute_weights(self, w_tind):

        # Use a log transfromation to avoid underflow 
        log_w_t     = tf.reduce_sum(tf.math.log(w_tind), axis = 0)
        max_log_w_t = tf.math.reduce_max(log_w_t, axis = 0)

        w_t_shifted = tf.exp(log_w_t - max_log_w_t)

        w_t = w_t_shifted*tf.exp(max_log_w_t)
        W_t = w_t_shifted/tf.reduce_sum(w_t_shifted)

        del log_w_t, max_log_w_t, w_t_shifted
        
        return tf.reshape(w_t, shape = (self.n_particles, 1) ), tf.reshape(W_t, shape = (self.n_particles, 1) )

    ################################################
    # resample the particles
    def resampling(self, weights_t):

        A_t =  tfp.distributions.Categorical(probs = weights_t, dtype=tf.int32).sample((self.n_particles))

        return A_t

    ################################################
    # Initialization of the SMC
    def step_0(self, Y_0):

        C_tp, w_tind = self.proposal_epidemic.proposal_epidemic_0(Y_0)

        w_t, W_t = self.compute_weights(w_tind)

        A_t = self.resampling(W_t[:, 0])

        log_p_y_t_given_y_1_to_tm1 = tf.math.log(tf.reduce_mean(w_t))

        return C_tp, A_t, log_p_y_t_given_y_1_to_tm1

    ################################################
    # General t step of the SMC
    def step_t(self, t, theta, C_tp, A_t, Y_t, delta_t):

        tildeC_tm1 = tf.gather(C_tp, A_t, axis = 1)

        C_tp, w_tind = self.proposal_epidemic.proposal_epidemic_t( t, theta, tildeC_tm1, Y_t, delta_t)

        w_t, W_t = self.compute_weights(w_tind)

        A_t = self.resampling(W_t[:, 0])

        log_p_y_t_given_y_1_to_tm1 = tf.math.log(tf.reduce_mean(w_t))

        return C_tp, A_t, log_p_y_t_given_y_1_to_tm1

    ################################################
    # Complete SMC to run during rejuvenation
    def filter(self, Y, max_delta_t):

        # create a schedule of time increments to use
        delta_list = create_delta_t_list(Y, max_delta_t)

        # The the index set for the map_fn loop
        index = tf.cast(tf.linspace((0.), (self.n_parameters-1), (self.n_parameters)), dtype = tf.int32)

        # Initialize the observations, the reshape is needed 
        # It require Y to be a numpy array with all the nans
        Y_t = tf.reshape(Y[:,0], shape = (len(Y[:,0]), 1) )

        # Run the step 0 of the SMC
        step_0_multi                          = lambda i: self.step_0(Y_t)
        C_tp, A_t, log_p_y_t_given_y_1_to_tm1 = tf.map_fn(fn = step_0_multi, elems = index, dtype=(tf.float32, tf.int32, tf.float32))
        # Initialize the log-likelihood estimate
        # log_p_y_1_to_T = tf.reshape(log_p_y_t_given_y_1_to_tm1, shape = (log_p_y_t_given_y_1_to_tm1.shape[0], 1))

        ancestors = tf.reshape(A_t,                           shape = (C_tp.shape[0], C_tp.shape[2], 1))
        infected  = tf.reshape(tf.reduce_sum(C_tp, axis = 1), shape = (C_tp.shape[0], C_tp.shape[2], 1))

        loc_H_I_multi = lambda i: tf.sparse.sparse_dense_matmul(self.households_individuals.loc_H_I, C_tp[i,:,:])
        H_tp          = tf.map_fn(fn = loc_H_I_multi, elems = index, dtype = (tf.float32))
        households    = tf.reshape(tf.reduce_mean(H_tp, axis = 2), shape = (H_tp.shape[0], H_tp.shape[1], 1))

        # Set the time and the initial index in the deltas schedule
        index_delta_t = 0
        t=0
        while index_delta_t< len(delta_list):       

            # Set the current time and delta
            t       = t + delta_list[index_delta_t]
            # print(t)
            delta_t = float(delta_list[index_delta_t])

            # Set the current Y_t
            Y_t = tf.reshape(Y[:,t], shape = (len(Y[:,t]), 1) )

            step_t_multi             = lambda i: self.step_t( t, self.theta[i,:], C_tp[i,:,:], A_t[i,:], Y_t, delta_t )
            C_tp, A_t, log_p_y_t_given_y_1_to_tm1 = tf.map_fn(fn = step_t_multi, elems = index, dtype = (tf.float32, tf.int32, tf.float32))

            ancestors = tf.concat( (ancestors, tf.reshape(A_t,                           shape = (C_tp.shape[0], C_tp.shape[2], 1)) ), axis = 2)
            infected  = tf.concat( (infected,  tf.reshape(tf.reduce_sum(C_tp, axis = 1), shape = (C_tp.shape[0], C_tp.shape[2], 1)) ), axis = 2)     

            loc_H_I_multi = lambda i: tf.sparse.sparse_dense_matmul(self.households_individuals.loc_H_I, C_tp[i,:,:])
            H_tp          = tf.map_fn(fn = loc_H_I_multi, elems = index, dtype = (tf.float32))
            households    = tf.concat( (households, tf.reshape(tf.reduce_mean(H_tp, axis = 2), shape = (H_tp.shape[0], H_tp.shape[1], 1))), axis = 2) 

            # log_p_y_1_to_T = log_p_y_1_to_T + tf.reshape(log_p_y_t_given_y_1_to_tm1, shape = (log_p_y_t_given_y_1_to_tm1.shape[0], 1))
            index_delta_t = index_delta_t + 1

        return ancestors, infected, households


    # def smoother(self, ancestors, infected):

    #     # The the index set for the map_fn loop
    #     index = tf.cast(tf.linspace((0.), (self.n_parameters-1), (self.n_parameters)), dtype = tf.int32)

    #     ancestors_track = ancestors[:,:,-1]

    #     reverse_gather = lambda i: tf.gather(infected[i,:,-1], ancestors_track[i,:], axis = 0)
    #     infected_t     = tf.map_fn(fn = reverse_gather, elems = index, dtype=(tf.float32))

    #     infected_smoothing = tf.reshape(infected_t, shape = (infected_t.shape[0], infected_t.shape[1], 1))

    #     for t in range(ancestors.shape[2]-2, -1, -1):

    #         ancestors_track = tf.gather(ancestors[0,:,t], ancestors_track)

    #         reverse_gather = lambda i: tf.gather(infected[i,:,t], ancestors_track[i,:], axis = 0)
    #         infected_t     = tf.reshape(tf.map_fn(fn = reverse_gather, elems = index, dtype=(tf.float32)), shape = (infected_t.shape[0], infected_t.shape[1], 1))

    #         infected_smoothing = tf.concat((infected_t, infected_smoothing), axis =2)

    #     return infected_smoothing