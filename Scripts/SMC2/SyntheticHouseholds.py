import tensorflow as tf
import tensorflow_probability as tfp

# import time
import numpy as np

class households_individuals:

    def __init__( self, n_individuals, n_households, 
                        households_longitude, households_latitude,
                        loc_H_I_index, 
                        individuals_covariates):

        super(households_individuals, self).__init__()

        # NR HOUSEHOLDS & NR INDIVIDUALS
        self.n_households  = n_households
        self.n_individuals = n_individuals
    
        # HOUSEHOLDS' LOCATIONS
        self.households_longitude = tf.convert_to_tensor(households_longitude.reshape(self.n_households, 1), dtype = tf.float32)
        self.households_latitude  = tf.convert_to_tensor(households_latitude.reshape(self.n_households, 1), dtype = tf.float32)

        # SAMPLE INDIVIDUALS' LOCATIONS
        self.loc_H_I = tf.sparse.SparseTensor( loc_H_I_index, tf.ones(self.n_individuals), dense_shape = [self.n_households, self.n_individuals] )

        ## STORE THE MATRIX OF FEATURES
        self.individuals = tf.convert_to_tensor( individuals_covariates, dtype = tf.float32 )

    # COMPUTE NR INDIVIDUALS PER HOUSEHOLD- use a 
    @property
    def households_size(self):

        return tf.reshape(tf.sparse.reduce_sum(self.loc_H_I, axis = 1), shape = (self.n_households, 1))

    # COMPUTE DISTANCES AMONG HOUSEHOLDS- use a matrix of dimension: nr households X nr households  
    @property
    def households_distances(self):

        # this is computing the euclidean distance between two households (the distances are in m so divide by 1000 to get the km)
        return (tf.square(self.households_longitude - tf.reshape( self.households_longitude, shape = (1, self.n_households))) + tf.square(self.households_latitude - tf.reshape( self.households_latitude, shape = (1, self.n_households))))/(1000**2)

    # COMPUTE HOW MANY COLONIZED PER HOUSEHOLD
    def households_colony(self, C_t):

        return tf.sparse.sparse_dense_matmul(self.loc_H_I, C_t)      


