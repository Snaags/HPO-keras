import ConfigSpace as CS 
import ConfigSpace.hyperparameters as CSH
import os
import csv
import time 


cs = CS.ConfigurationSpace()
###Training Configuration###
epochs = CSH.Constant(name = "epochs",value = 50)
batch_size = CSH.Constant(name = "batch_size",value = 32)
num_layers = CSH.Constant(name = "num_layers",value =4)

###Optimiser###
optimiser  = CSH.Constant(name = "optimiser", value = "Adam"		) 
lr =CSH.UniformFloatHyperparameter(name = "optimiser_lr",			lower = 1e-8,upper = 5e-1 ,log = True ) #lr
window = CSH.UniformIntegerHyperparameter(name= "window", lower = 1, upper = 16)


###Topology Definition]###
layer_1_type = CSH.Constant(name = "layer_1_type", value = "Conv1D")
layer_1_padding = CSH.Constant(name = "layer_1_padding",value = "same")
layer_1_filters = CSH.UniformIntegerHyperparameter(name = "layer_1_filters", lower = 16 ,upper = 128)
layer_1_BatchNormalization = CSH.UniformIntegerHyperparameter(name = "layer_1_BatchNormalization", lower = 0,upper = 1)
layer_1_kernel_size = CSH.UniformIntegerHyperparameter(name = "layer_1_kernel_size", lower =1 ,upper =6)




layer_2_type = CSH.Constant(name = "layer_2_type", value = "Conv1D")
layer_2_padding = CSH.Constant(name = "layer_2_padding",value = "same")
layer_2_filters = CSH.UniformIntegerHyperparameter(name = "layer_2_filters", lower = 16 ,upper = 128)
layer_2_BatchNormalization = CSH.UniformIntegerHyperparameter(name = "layer_2_BatchNormalization", lower = 0,upper = 1)
layer_2_kernel_size = CSH.UniformIntegerHyperparameter(name = "layer_2_kernel_size", lower =1 ,upper =6)

layer_3_type = CSH.Constant(name = "layer_3_type", value = "Conv1D")
layer_3_padding = CSH.Constant(name = "layer_3_padding",value = "same")
layer_3_filters = CSH.UniformIntegerHyperparameter(name = "layer_3_filters", lower = 16 ,upper = 128)
layer_3_BatchNormalization = CSH.UniformIntegerHyperparameter(name = "layer_3_BatchNormalization", lower = 0,upper = 1)
layer_3_kernel_size = CSH.UniformIntegerHyperparameter(name = "layer_3_kernel_size", lower =1 ,upper =6)


layer_4_type = CSH.Constant(name = "layer_4_type", value = "Dense")
layer_4_units = CSH.UniformIntegerHyperparameter(name = "layer_4_units", lower = 16, upper = 128)


#layer_4_padding = CSH.Constant(name = "padding",value = "same")
#layer_4_filters = CSH.UniformIntegrerHyperparameter(name = "filters", lower = 16 ,upper = 128)
#layer_4_BatchNormalization = CSH.UniformIntegrerHyperparameter(name = "BatchNormalization", lower = 0,upper = 1)
#layer_4_kernel_size = CSH.UniformIntegrerHyperparameter(name = "kernel_size_size", lower =1 ,upper =6)

####List of Hyperparameters

hp_list = [
    layer_1_type,
    layer_1_padding,
    layer_1_filters,
    layer_1_BatchNormalization,
    layer_1_kernel_size,
    layer_2_type,
    layer_2_padding,
    layer_2_filters,
    layer_2_BatchNormalization,
    layer_2_kernel_size,
    layer_3_type,
    layer_3_padding,
    layer_3_filters,
    layer_3_BatchNormalization,
    layer_3_kernel_size,
    layer_4_type,
    window,
    layer_4_units,
    epochs,
    batch_size,
    optimiser,
    lr,
    num_layers
]
cs.add_hyperparameters(hp_list)

scores = []
validation = []
log_file = "HPO_log.txt"
score_file = "TEPS_Random_scores.csv"
config_file = "TEPS_Random_configs.csv"


from HPO_utils import Dispatcher
from algorithms import GA
import ford_worker as train_function
from matplotlib import pyplot as plt
config_list = cs.sample_configuration(size = 32)
current_iter = 0


ga = GA(pop=32,elite = 0.25)
##Init ID dict
config_ID_dict = {} 
scores_ID_dict = {}
for index, conf in enumerate(config_list):
    config_ID_dict[index] = conf

log_file = "HPO_log.txt"
score_file = "TEPS_Random_scores.csv"
config_file = "TEPS_Random_configs.csv"


def save_to_file(csv_file,dict):
    with open(csv_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        for ID in config_ID_dict:
            writer.writerow([ID,dict[ID]])






while current_iter < 10:

    results = Dispatcher(config_list,1,validation_flag=False,train_func=train_function)
    scores = []
    validation = []

    ##Update scores dictionary
    if len(scores_ID_dict.keys()) > 0:
        start_index = max(scores_ID_dict.keys())  + 1
    else:
        start_index = 0
    for index, score in enumerate(results):
        idx = start_index + index
        scores_ID_dict[idx] = score

    save_to_file(config_file,config_ID_dict)
    save_to_file(score_file,scores_ID_dict)
    print(config_ID_dict,scores_ID_dict)
    config_list = ga.mutate(config_ID_dict,scores_ID_dict)

    ##Update Configs dictionary
    start_index = max(scores_ID_dict.keys())  + 1
    for index, conf in enumerate(config_list):
        idx = start_index + index
        config_ID_dict[idx] = conf

