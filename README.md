# snowpack_lstm

This repository includes the code necessary to train and run LSTM neural networks to predict snowpack in the western United States on three distinct spatial scales: single column, regional (statewide), and national. Additionally, code for comparisons with two physically-based snow models, ParFlow-CLM and the University of Arizona SWE model, is implemented. This repository was developed as part of a senior thesis project at Princeton University. 

The folders are as follows: 
lstm: Includes all of the code necessary to run single column, regional, and national models. All models share an LSTM structure. 
pfclm: Includes necessary code for running ParFlow-CLM at single column, regional, and national scales. 
