# SMC2-Individual Level Model
We propose a novel stochastic model for the spread of antimicrobial-resistant bacteria in a population, together with an efficient algorithm for fitting such a model to sample data. We introduce an individual-based model for the epidemic, with the state of the model determining which individuals are colonised by the bacteria. The transmission rate of the epidemic takes into account both individuals' locations, individuals' covariates, seasonality and environmental effects. The state of our model is only partially observed, with data consisting of test results from individuals from a sample of households taken roughly twice a week for 19 months. Fitting our model to data is challenging due to the large state space of our model. We develop an efficient SMC^2 algorithm to estimate parameters and compare models for the transmission rate. We implement this algorithm in a computationally efficient manner by using the scale invariance properties of the underlying epidemic model, which means we can define and fit our model for a population on the order of tens of thousands of individuals rather than millions. Our motivating application focuses on the dynamics of community-acquired Extended-Spectrum Beta-Lactamase-producing _Escherichia coli_ (_E. coli_) and _Klebsiella pneumoniae_ (_K. pneumoniae_), using data collected as part of the Drivers of Resistance in Uganda and Malawi project. We infer the parameters of the model and learn key epidemic quantities such as the effective reproduction number, spatial distribution of prevalence, household cluster dynamics, and seasonality.

# Contents
There are three folders in the repository:
- Cleaning: containing a Jupyter notebook and some data folders for cleaning the data. The main steps consists of:
  - clean the collected data and extract information about location, gender, income, age and colonization state over time;
  - clean the synthetic data and extract information about location, gender, income, age
  - join the data and get: time series to analyse + population data.
- Experiments: containing all the main scripts for running the experiments on an a cluster, there are 3 main folders:
  - RunEcoli: run the experiments on _E. coli_ data;
  - RunKpneumoniae: run the experiments on _K. penumoniae_ data;
  - RunRt: given the optimal parameters compute effective R and colonization state over time.
- Tutorial: containing a tutroail on how to use the SMC^2 code and all the key quantities to set-up.

# Synthetic Data
We provide the synthetic data in "Cleaning/Data/Synthetic".

