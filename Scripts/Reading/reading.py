import os
import numpy as np
import matplotlib.pyplot as plt

from proposal_epidemic import *


def read_output(path, size):

    slash = "\\"

    theta_chikwawa = {}
    theta_chileka = {}
    theta_ndirande = {} 

    p_y_t_given_y_1_to_tm1_chikwawa = {}
    p_y_t_given_y_1_to_tm1_chileka = {}
    p_y_t_given_y_1_to_tm1_ndirande = {}
    
    rejuvenation_time_chikwawa = {}
    rejuvenation_time_chileka = {}
    rejuvenation_time_ndirande = {}

    for i in range(1, size+1):

        for root, dirs, files in os.walk(path):

            for name in files:

                if name[0:5]=="theta":
                    if name[0:14]=="theta_"+str(i)+"_chikwa":
                        theta_chikwawa[str(i)] = np.load(path + slash + name)
                    elif name[0:13]=="theta_"+str(i)+"_chile":
                        theta_chileka[str(i)]  = np.load(path + slash + name)
                    elif name[0:14]=="theta_"+str(i)+"_ndiran":
                        theta_ndirande[str(i)] = np.load(path + slash + name)
                elif name[0:3]=="p_y":
                    if   name[0:29]   == "p_y_t_given_y_1_to_tm1_"+str(i)+"_chik":
                        p_y_t_given_y_1_to_tm1_chikwawa[str(i)] = np.load(path + slash + name)
                    elif name[0:29] == "p_y_t_given_y_1_to_tm1_"+str(i)+"_chil":
                        p_y_t_given_y_1_to_tm1_chileka[str(i)]  = np.load(path + slash + name)
                    elif name[0:29] == "p_y_t_given_y_1_to_tm1_"+str(i)+"_ndir":
                        p_y_t_given_y_1_to_tm1_ndirande[str(i)] = np.load(path + slash + name)
                        
                elif name[0:8]=="rejuvena":
                    if   name[0:24] == "rejuvenation_time_" + str(i) + "_chik":
                        rejuvenation_time_chikwawa[str(i)] = np.load(path + slash + name)
                    elif name[0:24] == "rejuvenation_time_" + str(i) + "_chil":
                        rejuvenation_time_chileka[str(i)]  = np.load(path + slash + name)
                    elif name[0:24] == "rejuvenation_time_" + str(i) + "_ndir":
                        rejuvenation_time_ndirande[str(i)] = np.load(path + slash + name)

    theta = {}
    theta["chikwawa"] = theta_chikwawa
    theta["chileka"]  = theta_chileka
    theta["ndirande"] = theta_ndirande
    
    p_y_t_given_y_1_to_tm1 = {}
    p_y_t_given_y_1_to_tm1["chikwawa"] = p_y_t_given_y_1_to_tm1_chikwawa
    p_y_t_given_y_1_to_tm1["chileka"]  = p_y_t_given_y_1_to_tm1_chileka
    p_y_t_given_y_1_to_tm1["ndirande"] = p_y_t_given_y_1_to_tm1_ndirande
    
    rejuvenation_time = {}
    rejuvenation_time["chikwawa"] = rejuvenation_time_chikwawa
    rejuvenation_time["chileka"]  = rejuvenation_time_chileka
    rejuvenation_time["ndirande"] = rejuvenation_time_ndirande

    return theta, p_y_t_given_y_1_to_tm1, rejuvenation_time



def plot_parameter(theta, parameter_index, prior_dict, colors, title, histogram_threshold =20):

    cities = ["chikwawa", "chileka", "ndirande"]

    fig, (ax) = plt.subplots(2, 3, figsize=(75, 25), dpi=20)

    for city_index in range(len(cities)):

        ax[0, city_index].xaxis.set_tick_params(labelsize=40)
        ax[0, city_index].yaxis.set_tick_params(labelsize=40)
        ax[0, city_index].set_title(title+" in "+cities[city_index], fontsize = 40)

        ax[1,city_index].xaxis.set_tick_params(labelsize=40)
        ax[1,city_index].yaxis.set_tick_params(labelsize=40)  

        for j in theta[cities[city_index]].keys():
            for i in range(theta[cities[city_index]][j].shape[0]):
                ax[0, city_index].plot(theta[cities[city_index]][j][i,parameter_index,:], color = colors[int(j)-1])

            if len(np.unique(theta[cities[city_index]][j][:,parameter_index,-1]))>histogram_threshold:
                ax[1,city_index].hist(theta[cities[city_index]][j][:,parameter_index,-1], color = colors[int(j)-1], density = True)
                ax[1,city_index].plot(prior_dict["x"], prior_dict["func"](prior_dict["x"], prior_dict["parameters"]), 'r-', lw=5, alpha=0.6)

def find_best_simulation(p_y_t_given_y_1_to_tm1):

    best_simulation = {}
    for city in p_y_t_given_y_1_to_tm1.keys():

        best_city = np.sum(np.log(p_y_t_given_y_1_to_tm1[city][str(1)]))
        for j in p_y_t_given_y_1_to_tm1[city]:

            if best_city<= np.sum(np.log(p_y_t_given_y_1_to_tm1[city][str(j)])):
                best_simulation[city] = j
                best_city             = np.sum(np.log(p_y_t_given_y_1_to_tm1[city][str(j)]))

    return best_simulation

def find_two_best(p_y_t_given_y_1_to_tm1, best_simulation, str_list):

    best_bayes_factor = {}

    o_key = list(best_simulation.keys())[0]

    for city in best_simulation[o_key].keys():
        
        best_bayes_factor[city] = ["name", 0]

        best_bayes_factor[city][0] = str_list[0]
        best_bayes_factor[city][1] = np.sum(np.log(p_y_t_given_y_1_to_tm1[str_list[0]][city][best_simulation[str_list[0]][city]]))

        for str_index in range(len(str_list)):

            best_simulation_index = best_simulation[str_list[str_index]][city]
            
            marginal_log_likelihood = np.sum(np.log(p_y_t_given_y_1_to_tm1[str_list[str_index]][city][best_simulation_index]))

            if marginal_log_likelihood>best_bayes_factor[city][1]:
                best_bayes_factor[city][0] = str_list[str_index]
                best_bayes_factor[city][1] = marginal_log_likelihood            

    second_best_bayes_factor = {}

    for city in best_simulation[o_key].keys():
        
        second_best_bayes_factor[city] = ["name", -10000]

        for str_index in range(len(str_list)):

            best_simulation_index = best_simulation[str_list[str_index]][city]
            
            marginal_log_likelihood = np.sum(np.log(p_y_t_given_y_1_to_tm1[str_list[str_index]][city][best_simulation_index]))

            if marginal_log_likelihood>second_best_bayes_factor[city][1] and str_list[str_index]!=best_bayes_factor[city][0]:
                second_best_bayes_factor[city][0] = str_list[str_index]
                second_best_bayes_factor[city][1] = marginal_log_likelihood       
            
    return best_bayes_factor, second_best_bayes_factor


def plot_parameter_best(theta, best, best_simulation, parameter_index_list, prior_dict, color_, histogram_threshold =20):
    
    cities = list(best.keys())

    fig, (ax) = plt.subplots(2, 3, figsize=(75, 25), dpi=20)

    for city_index in range(len(cities)):
 
        parameter_index = parameter_index_list[city_index]

        ax[0, city_index].xaxis.set_tick_params(labelsize=40)
        ax[0, city_index].yaxis.set_tick_params(labelsize=40)
        ax[0, city_index].set_title(best[cities[city_index]][0]+" in "+cities[city_index], fontsize = 40)

        ax[1,city_index].xaxis.set_tick_params(labelsize=40)
        ax[1,city_index].yaxis.set_tick_params(labelsize=40)  

        j = best_simulation[best[cities[city_index]][0]][cities[city_index]]

        for i in range(theta[best[cities[city_index]][0]][cities[city_index]][j].shape[0]):
            ax[0, city_index].plot(theta[best[cities[city_index]][0]][cities[city_index]][j][i,parameter_index,:], color = color_)

        if len(np.unique(theta[best[cities[city_index]][0]][cities[city_index]][j][:,parameter_index,-1]))>histogram_threshold:
            ax[1,city_index].hist(theta[best[cities[city_index]][0]][cities[city_index]][j][:,parameter_index,-1], color = color_, density = True)
            ax[1,city_index].plot(prior_dict["x"], prior_dict["func"](prior_dict["x"], prior_dict["parameters"]), 'r-', lw=5, alpha=0.6)


def save_parameter_best(theta, best, best_simulation, file_names_list):
    
    cities = list(best.keys())

    for city_index in range(len(cities)):

        file_name = file_names_list[city_index]

        j = best_simulation[best[cities[city_index]][0]][cities[city_index]]

        np.save("Data/"+file_name, theta[best[cities[city_index]][0]][cities[city_index]][j])





def prevalence_calculation(Y):

    prevalence = np.nan*np.zeros(Y.shape[1])

    if np.isnan(Y[:,0]).all()==False:
        prevalence[0] = np.nansum(Y[:,0])/np.sum(1-np.isnan(Y[:,0]))

    for t in range(1, Y.shape[1]):

        if np.isnan(Y[:,t]).all():
            prevalence[t] = prevalence[t-1]

        else:
            prevalence[t] = np.nansum(Y[:,t])/np.sum(1-np.isnan(Y[:,t]))

    return prevalence

def R_0_estimate(households_individuals, hyperparameters_epidemic, epidemic_type, time, city, best, best_simulation, theta,):
    # epidemic proposal
    proposal_epidemic_ = proposal_epidemic( households_individuals, 
                                            100,
                                            hyperparameters_epidemic,
                                            epidemic_type)

    simulation = best[city][0] 

    last_theta = theta[simulation][city][best_simulation[simulation][city]][:,:,-1]

    index = tf.cast(tf.linspace((0.), (last_theta.shape[0]-1), (last_theta.shape[0])), dtype = tf.int32)

    # Run the step 0 of the SMC
    R_0_multi = lambda i: proposal_epidemic_.R_0_calculation( time, last_theta[i])

    R_0, R_0_beta1 = tf.map_fn(fn = R_0_multi, elems = index, dtype=(tf.float32, tf.float32))

    return R_0, R_0_beta1

