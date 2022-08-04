import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from utils_SMC2 import *

class proposal_parameters():
    
    ################################################
    # initialization
    ################################################
    def __init__( self, n_parameters,
                        proposal_parameters_parameters,
                        type = "Gaussian random walk"): 
        
        self.type         = type 
        self.n_parameters = n_parameters

        if self.type =="Gaussian random walk":

            # parameter proposal parameter
            self.c     = proposal_parameters_parameters["c"]
            self.diag  = proposal_parameters_parameters["diag"]



        # self.sigma = proposal_parameters["sigma"]

        # self.coin_mixture = proposal_parameters["coin_mixture"]    
  
   ################################################
    # proposal for the parameters
    def proposal_parameters_K(self, theta, logWtheta):

        logWtheta_max = tf.reduce_max(logWtheta)
        Wtheta_shifted = tf.exp(logWtheta - logWtheta_max)
        Wtheta_normalized = Wtheta_shifted/tf.reduce_sum(Wtheta_shifted)

        theta_probabilities = tf.reshape(Wtheta_normalized, shape = (self.n_parameters,))
        Atheta_t            = tfp.distributions.Categorical(probs = theta_probabilities, dtype=tf.int32).sample((self.n_parameters))

        theta_preresampling = theta
        theta               = tf.gather(theta_preresampling, Atheta_t, axis = 0)

        if self.type =="Gaussian random walk":

            # Compute the proposal parameters
            mu_hat           = tf.reduce_sum(Wtheta_normalized*theta_preresampling, axis = 0)
            Sigma_hat        = tf.linalg.matmul(Wtheta_normalized*(theta_preresampling- mu_hat), (theta_preresampling- mu_hat), transpose_a = True)
            Sigma_hat_stable = self.c*( Sigma_hat + self.diag*tf.linalg.diag(tf.ones(mu_hat.shape[0])))

            # propose a paramer
            theta_rv  = tfp.distributions.MultivariateNormalFullCovariance( loc = theta, covariance_matrix = Sigma_hat_stable )
            theta_tilde = theta_rv.sample() 

            # compute the p(\tilde{\theta}|\theta)
            sigma_hat_inv  = tf.linalg.inv(Sigma_hat_stable)
            log_p_theta_tilde_given_theta = Gaussian_log_pdf_no_Z(theta_tilde, theta,       sigma_hat_inv) 
            log_p_theta_given_theta_tilde = Gaussian_log_pdf_no_Z(theta,       theta_tilde, sigma_hat_inv)  

        return theta_tilde, log_p_theta_tilde_given_theta, log_p_theta_given_theta_tilde



    # ################################################
    # # proposal for the parameters
    # def proposal_one_step_parameters(self, t, theta, theta_preres, logWtheta):

    #     logWtheta_max = tf.reduce_max(logWtheta)
    #     Wtheta_shifted = tf.exp(logWtheta - logWtheta_max)

    #     Wtheta_normalized = Wtheta_shifted/tf.reduce_sum(Wtheta_shifted)

    #     # Compute the proposal parameters
    #     mu_hat    = tf.reduce_sum(Wtheta_normalized*theta_preres, axis = 0)
    #     sigma_hat_1 = self.c*(tf.linalg.matmul(Wtheta_normalized*(theta_preres- mu_hat), theta_preres- mu_hat, transpose_a = True) + self.b*tf.linalg.diag(tf.ones(mu_hat.shape[0])))
    #     sigma_hat_2 = self.a*tf.linalg.diag(tf.ones(theta.shape[1]))

    #     theta_rv_1  = tfp.distributions.MultivariateNormalFullCovariance( loc = theta, covariance_matrix = sigma_hat_1 )
    #     theta_rv_2  = tfp.distributions.MultivariateNormalFullCovariance( loc = theta, covariance_matrix = sigma_hat_2 )

    #     coin_prob = self.coin_mixture 
    #     coin  = tf.expand_dims(tfp.distributions.Bernoulli( probs=coin_prob, dtype=tf.float32, ).sample(self.n_parameters), 1)
    #     theta_tilde = coin*theta_rv_1.sample() + (1-coin)*theta_rv_2.sample() # theta_rv_1.sample() # 

    #     # compute the p(\tilde{\theta}|\theta)
    #     sigma_hat_inv_1  = tf.linalg.inv(sigma_hat_1)
    #     sigma_hat_det_1  = tf.linalg.det(sigma_hat_1)

    #     sigma_hat_inv_2  = tf.linalg.inv(sigma_hat_2)
    #     sigma_hat_det_2  = tf.linalg.det(sigma_hat_2)

    #     # p_theta_tilde_given_theta_1 = Gaussian_pdf(theta_tilde, theta,       sigma_hat_inv_2, sigma_hat_det_1)
    #     # p_theta_given_theta_tilde_1 = Gaussian_pdf(theta,       theta_tilde, sigma_hat_inv_2, sigma_hat_det_1) 

    #     # p_theta_tilde_given_theta_2 = Gaussian_pdf(theta_tilde, theta,       sigma_hat_inv_1, sigma_hat_det_2)
    #     # p_theta_given_theta_tilde_2 = Gaussian_pdf(theta,       theta_tilde, sigma_hat_inv_1, sigma_hat_det_2)    

    #     # log_p_theta_tilde_given_theta = Gaussian_log_pdf_no_Z(theta_tilde, theta,       sigma_hat_inv_2) # tf.math.log( coin_prob*p_theta_tilde_given_theta_1 + ((1-coin_prob)*p_theta_tilde_given_theta_2))
    #     # log_p_theta_given_theta_tilde = Gaussian_log_pdf_no_Z(theta,       theta_tilde, sigma_hat_inv_2)  # tf.math.log( coin_prob*p_theta_given_theta_tilde_1 + ((1-coin_prob)*p_theta_given_theta_tilde_2))

    #     return theta_tilde # , log_p_theta_tilde_given_theta, log_p_theta_given_theta_tilde