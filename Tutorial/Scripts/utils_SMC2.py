import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from math import gamma as gamma_func

class prior():

    def __init__(self, prior_type, param):
        super(prior, self).__init__()    

        self.prior_type = prior_type

        # X distributed as Gamma
        if self.prior_type == "Gamma":
            self.shape     = param["shape"]
            self.scale     = param["scale"]

            self.prior_rv = tfp.distributions.Gamma( concentration = [self.shape], rate = [1/self.scale] )

        # X distributed as a multivariate Gaussian
        if self.prior_type == "Multivariate Gaussian":
            self.mean       = param["mean"]
            self.covariance = param["covariance_matrix"]
            self.covariance_inv = tf.linalg.inv(self.covariance)

            # self.prior_rv = tfp.distributions.MultivariateNormalTriL( loc = self.mean, scale_tril = tf.linalg.cholesky(self.covariance) )
            self.prior_rv = tfp.distributions.MultivariateNormalFullCovariance( loc = self.mean, covariance_matrix = self.covariance )

        if self.prior_type == "Univariate Gaussian":
            self.mean = param["mean"]
            self.std  = param["std"]

            self.prior_rv = tfp.distributions.Normal( loc = [self.mean], scale = [self.std] )

        if self.prior_type == "Beta":
            self.alpha = param["alpha"]
            self.beta  = param["beta"]

            self.prior_rv = tfp.distributions.Beta( concentration1 = [self.alpha], concentration0 = [self.beta] )


    def sample(self, n = 1):

        return self.prior_rv.sample(n)


    def log_pdf(self, x):

        if self.prior_type == "Gamma":
            y = (self.shape - 1)*tf.math.log(x)-(x)/self.scale #- self.shape*tf.math.log(self.scale) - np.log(gamma_func(self.shape))
            y = y + tf.cast(x<0, dtype = tf.float32)*-200

        if self.prior_type == "Multivariate Gaussian":
            y = -0.5*tf.reduce_sum(tf.linalg.matmul( (x - self.mean), self.covariance_inv)*(x - self.mean), axis =1, keepdims = True)

        if self.prior_type == "Univariate Gaussian":
            # to avoid underflow remove the normalizing const
            y = -tf.square(x-self.mean)/(2*self.std*self.std)
            # Z = tf.sqrt(2*np.pi*self.std*self.std)
            # y = tf.exp(y)/Z

        if self.prior_type ==  "Beta":
            x_nonzero = tf.cast(x>0, dtype = tf.float32)*tf.cast(x<1, dtype = tf.float32)*x

            # to avoid underflow remove the normalizing const
            y = (self.alpha - 1)*tf.math.log(x_nonzero) + (self.beta - 1)*tf.math.log(1-x_nonzero) #- self.shape*tf.math.log(self.scale) - np.log(gamma_func(self.shape))

            y + tf.cast(x>0, dtype = tf.float32)*tf.cast(x<1, dtype = tf.float32)*-200

        return y



# Gaussian_pdf define the pdf of a multivariate Gaussian, this is used in proposing the parameters
def Gaussian_pdf(x, mean, covariance_inv, sigma_hat_dete):

    y = tf.reduce_sum(tf.linalg.matmul( (x - mean), covariance_inv)*(x - mean), axis =1, keepdims = True)
    Z = ((2*np.pi)**(0.5*mean.shape[0]))*tf.math.pow(tf.abs(sigma_hat_dete), 0.5)

    return tf.exp((-0.5*y) - tf.math.log(Z))

# Gaussian_pdf define the pdf of a multivariate Gaussian, this is used in proposing the parameters
def Gaussian_log_pdf_no_Z(x, mean, covariance_inv):

    y = tf.reduce_sum(tf.linalg.matmul( (x - mean), covariance_inv)*(x - mean), axis =1, keepdims = True)

    return (-0.5*y)

# Replace nan
def replacenan(t):
    return tf.cast(tf.where(tf.math.is_nan(t), tf.zeros_like(t), t), dtype = tf.float32)


class joint_prior():

    def __init__(self, prior_list, prior_dim):
        super(joint_prior, self).__init__()    

        self.prior_list = prior_list
        self.prior_dim  = prior_dim
        self.n_priors = len(prior_list)
        
    def sample(self, n_parameters):

        return tf.concat(tuple(self.prior_list[i].sample((n_parameters)) for i in range(self.n_priors)), axis = 1)

    def log_pdf(self, theta):

        prior_sizes_cumsum     = np.zeros(len(self.prior_list)+1, dtype = np.int)
        prior_sizes_cumsum[1:] = np.cumsum(self.prior_dim)

        return tf.concat(tuple(self.prior_list[i].log_pdf(theta[:,prior_sizes_cumsum[i]:prior_sizes_cumsum[i+1]]) for i in range(0, self.n_priors)), axis = 1)


def create_delta_t_list(Y, max_delta_t):
    T = Y.shape[1]
    indeces = (1-np.all(np.isnan(Y), axis = 0)).astype(bool)
    schedule = np.linspace(0, Y.shape[1]-1, Y.shape[1]).astype(int)[indeces]
    if schedule[0]!=0:
        new_schedule = np.zeros(len(schedule)+1, dtype = int)
        new_schedule[1:] = schedule
        schedule = new_schedule
        
    if schedule[-1]!=(T-1):
        new_schedule = np.zeros(len(schedule)+1, dtype = int) + (T-1)
        new_schedule[0:-1] = schedule
        schedule = new_schedule

    delta_t_list = np.array([])

    for index in range(1, len(schedule)):
        schedule_differenece = (schedule[index]- schedule[index-1])
        if schedule_differenece%max_delta_t==0:
            n_max_delta_t = int(schedule_differenece/max_delta_t)
            new_delta_t_list = np.array([max_delta_t]*n_max_delta_t)
            delta_t_list = np.concatenate((delta_t_list, new_delta_t_list))
        else:
            rest = np.array([schedule_differenece%max_delta_t])
            delta_t_list = np.concatenate((delta_t_list, rest))
            schedule_differenece = schedule_differenece - rest
            n_max_delta_t = int(schedule_differenece/max_delta_t)
            new_delta_t_list = np.array([max_delta_t]*n_max_delta_t)
            delta_t_list = np.concatenate((delta_t_list, new_delta_t_list))
        delta_t_list = np.array(delta_t_list, dtype = int)
    # delta_t_list = np.concatenate((delta_t_list, np.array([1], dtype = int)))

    return delta_t_list