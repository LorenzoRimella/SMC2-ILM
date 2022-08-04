import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from math import gamma as gamma_func

import time

from utils_SMC2 import *
from proposal_epidemic import *
from proposal_parameters import *

#######################################
# SMC2
#######################################

class SMC2_epidemic():

    ################################################
    # initialization
    ################################################
    def __init__( self, households_individuals, 
                        n_particles, n_parameters,
                        ESS_threshold,
                        prior_parameters_epidemic,
                        hyperparameters_epidemic,
                        proposal_parameters_parameters,
                        epidemic_type = "3 parameters within-across households and fixed recovery",
                        proposal_type = "Gaussian random walk",
                        prior_type    = "Gaussian-Gaussian-Gaussian-MGaussian-Beta"):

        super(SMC2_epidemic, self).__init__()
        
        ################################################
        # Household graph and individual detection
        self.households_individuals = households_individuals

        ################################################
        # number of particles and parameters
        self.n_particles  = n_particles
        self.n_parameters = n_parameters

        # epidemic proposal
        self.proposal_epidemic = proposal_epidemic( households_individuals, 
                                                    n_particles,
                                                    hyperparameters_epidemic,
                                                    epidemic_type)

        # parameters proposal
        self.proposal_parameters = proposal_parameters( n_parameters,
                                                        proposal_parameters_parameters,
                                                        proposal_type)

        # ESS threshold
        self.ESS_threshold = ESS_threshold

        if prior_type == "Gaussian-Gaussian-Gaussian-MGaussian-Beta":
            ################################################
            # Set the priors over the parameters
            # this can be done in a smarter way by just giving a list of priors names and a list of priors parameters
            # prior_parameters_epidemic = [[-9, 0.5],[...],...] and prior_type_epidemic = ["Univariate Gaussian",... ]
            log_beta_1_prior      = prior( "Univariate Gaussian",   prior_parameters_epidemic["log_beta_1"] )
            log_beta_2_prior      = prior( "Univariate Gaussian",   prior_parameters_epidemic["log_beta_2"] )
            log_phi_prior         = prior( "Univariate Gaussian",   prior_parameters_epidemic["log_phi"] )
            individual_coef_prior = prior( "Multivariate Gaussian", prior_parameters_epidemic["individual_coef"] )
            coef_prior            = prior( "Beta",                  prior_parameters_epidemic["coef"] )

            n_covariates = prior_parameters_epidemic["individual_coef"]["mean"].shape[0]
            prior_list = [log_beta_1_prior, log_beta_2_prior, log_phi_prior, individual_coef_prior, coef_prior,]
            prior_dim  = [1,                1,                1,             n_covariates,                      1]

            self.prior = joint_prior(prior_list, prior_dim)  

            
        if prior_type == "Gaussian-Gaussian-Gaussian-MGaussian":
            ################################################
            # Set the priors over the parameters
            # this can be done in a smarter way by just giving a list of priors names and a list of priors parameters
            # prior_parameters_epidemic = [[-9, 0.5],[...],...] and prior_type_epidemic = ["Univariate Gaussian",... ]
            log_beta_1_prior      = prior( "Univariate Gaussian",   prior_parameters_epidemic["log_beta_1"] )
            log_beta_2_prior      = prior( "Univariate Gaussian",   prior_parameters_epidemic["log_beta_2"] )
            log_phi_prior         = prior( "Univariate Gaussian",   prior_parameters_epidemic["log_phi"] )
            individual_coef_prior = prior( "Multivariate Gaussian", prior_parameters_epidemic["individual_coef"] )

            n_covariates = prior_parameters_epidemic["individual_coef"]["mean"].shape[0]
            prior_list = [log_beta_1_prior, log_beta_2_prior, log_phi_prior, individual_coef_prior]
            prior_dim  = [1,                1,                1,             n_covariates]

            self.prior = joint_prior(prior_list, prior_dim)  


        if prior_type == "Gaussian-Gaussian-MGaussian-Beta":
            ################################################
            # Set the priors over the parameters
            # this can be done in a smarter way by just giving a list of priors names and a list of priors parameters
            # prior_parameters_epidemic = [[-9, 0.5],[...],...] and prior_type_epidemic = ["Univariate Gaussian",... ]
            log_beta_1_prior      = prior( "Univariate Gaussian",   prior_parameters_epidemic["log_beta_1"] )
            log_phi_prior         = prior( "Univariate Gaussian",   prior_parameters_epidemic["log_phi"] )
            individual_coef_prior = prior( "Multivariate Gaussian", prior_parameters_epidemic["individual_coef"] )
            coef_prior            = prior( "Beta",                  prior_parameters_epidemic["coef"] )

            n_covariates = prior_parameters_epidemic["individual_coef"]["mean"].shape[0]
            prior_list = [log_beta_1_prior, log_phi_prior, individual_coef_prior, coef_prior,]
            prior_dim  = [1,                1,             n_covariates,          1]

            self.prior = joint_prior(prior_list, prior_dim)  
            

        if prior_type == "Gaussian-Gaussian-MGaussian":
            ################################################
            # Set the priors over the parameters
            # this can be done in a smarter way by just giving a list of priors names and a list of priors parameters
            # prior_parameters_epidemic = [[-9, 0.5],[...],...] and prior_type_epidemic = ["Univariate Gaussian",... ]
            log_beta_1_prior      = prior( "Univariate Gaussian",   prior_parameters_epidemic["log_beta_1"] )
            log_phi_prior         = prior( "Univariate Gaussian",   prior_parameters_epidemic["log_phi"] )
            individual_coef_prior = prior( "Multivariate Gaussian", prior_parameters_epidemic["individual_coef"] )

            n_covariates = prior_parameters_epidemic["individual_coef"]["mean"].shape[0]
            prior_list = [log_beta_1_prior, log_phi_prior, individual_coef_prior,]
            prior_dim  = [1,                1,             n_covariates,         ]

            self.prior = joint_prior(prior_list, prior_dim)  

        
            
        if prior_type == "Gaussian-Gaussian-Gaussian":
            ################################################
            # Set the priors over the parameters
            # this can be done in a smarter way by just giving a list of priors names and a list of priors parameters
            # prior_parameters_epidemic = [[-9, 0.5],[...],...] and prior_type_epidemic = ["Univariate Gaussian",... ]
            log_beta_1_prior      = prior( "Univariate Gaussian",   prior_parameters_epidemic["log_beta_1"] )
            log_beta_2_prior      = prior( "Univariate Gaussian",   prior_parameters_epidemic["log_beta_2"] )
            log_phi_prior         = prior( "Univariate Gaussian",   prior_parameters_epidemic["log_phi"] )

            prior_list = [log_beta_1_prior, log_beta_2_prior, log_phi_prior]
            prior_dim  = [1,                1,                1,           ]

            self.prior = joint_prior(prior_list, prior_dim)  

            
        if prior_type == "Gaussian-Gaussian-Gaussian-Gaussian":
            ################################################
            # Set the priors over the parameters
            # this can be done in a smarter way by just giving a list of priors names and a list of priors parameters
            # prior_parameters_epidemic = [[-9, 0.5],[...],...] and prior_type_epidemic = ["Univariate Gaussian",... ]
            log_beta_1_prior       = prior( "Univariate Gaussian",   prior_parameters_epidemic["log_beta_1"] )
            log_beta_2_prior       = prior( "Univariate Gaussian",   prior_parameters_epidemic["log_beta_2"] )
            log_phi_prior          = prior( "Univariate Gaussian",   prior_parameters_epidemic["log_phi"] )
            log_fixed_effect_prior = prior( "Univariate Gaussian",   prior_parameters_epidemic["log_fixed_effect"] )

            prior_list = [log_beta_1_prior, log_beta_2_prior, log_phi_prior, log_fixed_effect_prior]
            prior_dim  = [1,                1,                1            , 1]

            self.prior = joint_prior(prior_list, prior_dim)  


            
        if prior_type == "Gaussian-Gaussian-Gaussian-Gaussian-MGaussian":
            ################################################
            # Set the priors over the parameters
            # this can be done in a smarter way by just giving a list of priors names and a list of priors parameters
            # prior_parameters_epidemic = [[-9, 0.5],[...],...] and prior_type_epidemic = ["Univariate Gaussian",... ]
            log_beta_1_prior       = prior( "Univariate Gaussian",   prior_parameters_epidemic["log_beta_1"] )
            log_beta_2_prior       = prior( "Univariate Gaussian",   prior_parameters_epidemic["log_beta_2"] )
            log_phi_prior          = prior( "Univariate Gaussian",   prior_parameters_epidemic["log_phi"] )
            log_fixed_effect_prior = prior( "Univariate Gaussian",   prior_parameters_epidemic["log_fixed_effect"] )
            individual_coef_prior  = prior( "Multivariate Gaussian", prior_parameters_epidemic["individual_coef"] )

            n_covariates = prior_parameters_epidemic["individual_coef"]["mean"].shape[0]
            prior_list   = [log_beta_1_prior, log_beta_2_prior, log_phi_prior, log_fixed_effect_prior, individual_coef_prior]
            prior_dim    = [1,                1,                1            , 1                     , n_covariates]

            self.prior = joint_prior(prior_list, prior_dim)  


    ################################################
    # Main functions
    #################################################

    ################################################
    # compute the weights
    def evidence(self, log_p_y_t_given_y_1_to_tm1, logw_theta):

        p_y_t_given_y_1_to_tm1 = tf.exp(tf.reshape(log_p_y_t_given_y_1_to_tm1, shape = (log_p_y_t_given_y_1_to_tm1.shape[0], 1)))

        # Use a log transfromation to avoid underflow 
        max_log_w_theta = tf.math.reduce_max(logw_theta, axis = 0)

        w_theta_shifted = tf.exp(logw_theta - max_log_w_theta)

        W_theta = w_theta_shifted/tf.reduce_sum(w_theta_shifted)

        del max_log_w_theta, w_theta_shifted,
        
        return tf.reduce_sum(W_theta*p_y_t_given_y_1_to_tm1, keepdims =True)

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
    def SMC(self, current_index, theta, Y, delta_list):

        # The the index set for the map_fn loop
        index = tf.cast(tf.linspace((0.), (self.n_parameters-1), (self.n_parameters)), dtype = tf.int32)

        # Initialize the observations, the reshape is needed 
        # It require Y to be a numpy array with all the nans
        Y_t = tf.reshape(Y[:,0], shape = (len(Y[:,0]), 1) )

        # Run the step 0 of the SMC
        step_0_multi                          = lambda i: self.step_0(Y_t)
        C_tp, A_t, log_p_y_t_given_y_1_to_tm1 = tf.map_fn(fn = step_0_multi, elems = index, dtype=(tf.float32, tf.int32, tf.float32))
        # Initialize the log-likelihood estimate
        log_p_y_1_to_T = tf.reshape(log_p_y_t_given_y_1_to_tm1, shape = (log_p_y_t_given_y_1_to_tm1.shape[0], 1))

        # Set the time and the initial index in the deltas schedule
        index_delta_t = 0
        t=0
        while index_delta_t<= current_index:       

            # Set the current time and delta
            t       = t + delta_list[index_delta_t]
            delta_t = float(delta_list[index_delta_t])

            # Set the current Y_t
            Y_t = tf.reshape(Y[:,t], shape = (len(Y[:,t]), 1) )

            step_t_multi             = lambda i: self.step_t( t, theta[i,:], C_tp[i,:,:], A_t[i,:], Y_t, delta_t )
            C_tp, A_t, log_p_y_t_given_y_1_to_tm1 = tf.map_fn(fn = step_t_multi, elems = index, dtype = (tf.float32, tf.int32, tf.float32))

            log_p_y_1_to_T = log_p_y_1_to_T + tf.reshape(log_p_y_t_given_y_1_to_tm1, shape = (log_p_y_t_given_y_1_to_tm1.shape[0], 1))
            index_delta_t = index_delta_t + 1

        return log_p_y_1_to_T, C_tp, A_t

    ################################################
    # ESS computation
    def ESS(self, log_p_y_1_to_T):

        log_p_y_1_to_T_max = tf.reduce_max(log_p_y_1_to_T)
        p_y_1_to_T_shifted = tf.exp(log_p_y_1_to_T- log_p_y_1_to_T_max)
        
        return p_y_1_to_T_shifted, tf.math.pow(tf.reduce_sum(p_y_1_to_T_shifted, axis = 0), 2)/tf.reduce_sum(tf.math.pow(p_y_1_to_T_shifted, 2), axis = 0)


    ################################################
    # Rejuvenation step
    def Rejuvenation_step(self, theta, log_p_y_1_to_T, 
                                current_index, Y, delta_list,
                                C_tp, A_t, outputfile_name):

        # propose new parameters
        theta_tilde, log_p_theta_tilde_given_theta, log_p_theta_given_theta_tilde = self.proposal_parameters.proposal_parameters_K(theta, log_p_y_1_to_T)  

        # run the SMC
        log_p_y_1_to_T_tilde, C_tp_tilde, A_t_tilde = self.SMC(current_index, theta_tilde, Y, delta_list)

        # Compute the prior score
        log_prior       = tf.reduce_sum(self.prior.log_pdf(theta),       axis = 1, keepdims = True)
        log_prior_tilde = tf.reduce_sum(self.prior.log_pdf(theta_tilde), axis = 1, keepdims = True)
    

        # Compute the Metropolis-Hastings probability
        logmetropolis_probability = tf.reduce_sum(log_p_y_1_to_T_tilde - log_p_y_1_to_T, axis = 1, keepdims = True) + log_prior_tilde - log_prior + log_p_theta_given_theta_tilde  - log_p_theta_tilde_given_theta           
        metropolis_probability = tf.exp(logmetropolis_probability)
        metropolis_probability = tf.reduce_min(tf.concat((metropolis_probability, tf.ones(metropolis_probability.shape)), axis =1), axis = 1, keepdims = True)

        del log_prior, log_prior_tilde, log_p_theta_tilde_given_theta, log_p_theta_given_theta_tilde

        # accept/reject according to the MH ratio
        U = tf.random.uniform((self.n_parameters, 1), 0, 1)
        indexing     = tf.cast(metropolis_probability>U, dtype = tf.float32)
        indexing_int = tf.cast(metropolis_probability>U, dtype = tf.int32)

        theta = theta_tilde*indexing + theta*(1-indexing)

        C_tp = C_tp_tilde*tf.expand_dims(indexing, 1) + C_tp*(1-tf.expand_dims(indexing, 1))   
        A_t = A_t_tilde*indexing_int + A_t*(1-indexing_int)

        log_p_y_1_to_T = log_p_y_1_to_T_tilde*indexing + log_p_y_1_to_T*(1-indexing)

        logw = tf.zeros(log_p_y_1_to_T.shape)

        # compute acceptance rate and print it in the output file
        acceptance_rate = tf.reduce_sum(indexing)/self.n_parameters
        string1 = ["Acceptance rate during rejuvenation: "+str(acceptance_rate.numpy()), "\n"]
        f= open(outputfile_name,"a")
        f.writelines(string1)
        f.close()

        del C_tp_tilde, A_t_tilde, log_p_y_1_to_T_tilde, theta_tilde, U, indexing, indexing_int

        return theta, C_tp, A_t, log_p_y_1_to_T, logw

    ##########################################################
    # SMC^2
    def SMC2(self, Y, max_delta_t, outputfile_name = "example"):

        # Parameters initialization: sample from the prior
        theta = self.prior.sample(self.n_parameters) 

        # Initialize the parameters particles history
        theta_particles_history = tf.reshape(theta, shape = (theta.shape[0], theta.shape[1], 1)).numpy()

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

        # Store the history of infection over time
        #infected_history =  tf.reduce_sum(C_tp, axis = 1, keepdims = True).numpy()

        # Initialize the log-likelihood estimate
        log_p_y_1_to_T = tf.reshape(log_p_y_t_given_y_1_to_tm1, shape = (log_p_y_t_given_y_1_to_tm1.shape[0], 1))

        # Initialize the likelihood history
        p_y_t_given_y_1_to_tm1_history = tf.reduce_mean(tf.exp(log_p_y_1_to_T), keepdims=True).numpy()

        # Initialize the theta weights
        logw_theta = log_p_y_1_to_T

        # Check the ESS, use a shifted version to avoid underflow
        logw_theta_shifted, ESS = self.ESS(logw_theta)
        
        # Rejuvenation step
        current_index = 0
        rejuvenation_time_history = tf.zeros((1)).numpy()

        if ESS < self.ESS_threshold*self.n_parameters: 

            string = ["Time step "+ str(t), "\n"]
            f= open(outputfile_name,"a")
            f.writelines(string)
            f.close()
            
            string = ["Rejuvenation step "+ str(ESS[0].numpy()), "\n"]
            f= open(outputfile_name,"a")
            f.writelines(string)
            f.close()

            # Resample theta
            theta_probab = tf.reshape(logw_theta_shifted/tf.reduce_sum(logw_theta_shifted), shape = (self.n_parameters,))
            Atheta_t     = tfp.distributions.Categorical(probs = theta_probab, dtype = tf.int32).sample((self.n_parameters))
            theta        = tf.gather(theta, Atheta_t, axis = 0)

            theta, C_tp, A_t, log_p_y_1_to_T, logw_theta = self.Rejuvenation_step( theta, log_p_y_1_to_T, 
                                                                                   current_index, Y, delta_list, 
                                                                                   C_tp, A_t, outputfile_name )

        # Set the time and the initial index in the deltas schedule
        index_delta_t = 0
        t=0

        while index_delta_t< len(delta_list):       

            # Set the current time and delta
            t       = t + delta_list[index_delta_t]
            delta_t = float(delta_list[index_delta_t])

            if index_delta_t%10==0:
                string = ["Time step "+ str(t), "\n"]
                f= open(outputfile_name,"a")
                f.writelines(string)
                f.close()

            # Set the current Y_t
            Y_t = tf.reshape(Y[:,t], shape = (len(Y[:,t]), 1) )

            step_t_multi             = lambda i: self.step_t( t, theta[i,:], C_tp[i,:,:], A_t[i,:], Y_t, delta_t )
            C_tp, A_t, log_p_y_t_given_y_1_to_tm1 = tf.map_fn(fn = step_t_multi, elems = index, dtype = (tf.float32, tf.int32, tf.float32))
            #infected_history = np.concatenate((infected_history, tf.reduce_sum(C_tp, axis = 1, keepdims = True).numpy()), axis = 1)

            p_y_t_given_y_1_to_tm1_history = np.concatenate((p_y_t_given_y_1_to_tm1_history, self.evidence(log_p_y_t_given_y_1_to_tm1, logw_theta).numpy()), axis = 1 )

            log_p_y_1_to_T         = log_p_y_1_to_T + tf.reshape(log_p_y_t_given_y_1_to_tm1, shape = (log_p_y_t_given_y_1_to_tm1.shape[0], 1))
            logw_theta             = logw_theta     + tf.reshape(log_p_y_t_given_y_1_to_tm1, shape = (log_p_y_t_given_y_1_to_tm1.shape[0], 1))

            # Check the ESS, use a shifted version to avoid underflow
            logw_theta_shifted, ESS = self.ESS(logw_theta)

            # Rejuvenation step
            current_index = index_delta_t
            if ESS < self.ESS_threshold*self.n_parameters: 
                string = ["Time step "+ str(t), "\n"]
                f= open(outputfile_name,"a")
                f.writelines(string)
                f.close()

                rejuvenation_time_history = np.concatenate((rejuvenation_time_history, np.array([t], dtype = rejuvenation_time_history[0].dtype)), axis = 0)

                string = ["Rejuvenation step "+ str(ESS[0].numpy()), "\n"]
                f= open(outputfile_name,"a")
                f.writelines(string)
                f.close()

                # Resample theta
                theta_probab = tf.reshape(logw_theta_shifted/tf.reduce_sum(logw_theta_shifted), shape = (self.n_parameters,))
                Atheta_t     = tfp.distributions.Categorical(probs = theta_probab, dtype = tf.int32).sample((self.n_parameters))
                theta        = tf.gather(theta, Atheta_t, axis = 0)

                theta, C_tp, A_t, log_p_y_1_to_T, logw_theta = self.Rejuvenation_step( theta, log_p_y_1_to_T, 
                                                                                       current_index, Y, delta_list, 
                                                                                       C_tp, A_t, outputfile_name)

            theta_particles_history = np.concatenate((theta_particles_history, tf.reshape(theta, shape = (theta.shape[0], theta.shape[1], 1)).numpy()), axis = 2 )

            index_delta_t = index_delta_t +1


        return theta_particles_history, p_y_t_given_y_1_to_tm1_history, rejuvenation_time_history










