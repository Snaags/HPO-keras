import ConfigSpace as CS 
import ConfigSpace.hyperparameters as CSH
import os
import csv
import time 
def generate_layer_hp(hp_list, layer_num):
    for idx, i in enumerate(hp_list):
        i.replace("@",layer_num)
        hp_list[idx] = eval(i)
    return hp_list
def batch_add_cond(cs,a_list, b,num):
    for a in a_list:
        cond = CS.GreaterThanCondition(a,b,num)
        cs.add_condition(cond)  
    return 0
def init_config():
    max_conv_layers = 5
    max_dense_layers = 3
    cs = CS.ConfigurationSpace()
    ###Training Configuration###
    epochs = CSH.Constant(name = "epochs",value = 50)
    batch_size = CSH.Constant(name = "batch_size",value = 32)
    num_conv_layers = CSH.UniformIntegerHyperparameter(name = "num_conv_layers",lower =1 ,upper = max_conv_layers)
    num_dense_layers = CSH.UniformIntegerHyperparameter(name = "num_dense_layers",lower =1 ,upper = max_conv_layers)


    ###Optimiser###
    optimiser  = CSH.Constant(name = "optimiser", value = "Adam"		) 
    lr =CSH.UniformFloatHyperparameter(name = "optimiser_lr",			lower = 1e-8,upper = 5e-1 ,log = True ) #lr



    window_size = CSH.UniformIntegerHyperparameter(name = "window_size", lower = 1 ,upper = 16)
    ###Topology Definition]###
    conv_1_type = CSH.Constant(name = "conv_1_type", value = "Conv1D")
    conv_1_padding = CSH.Constant(name = "conv_1_padding",value = "same")
    conv_1_filters = CSH.UniformIntegerHyperparameter(name = "conv_1_filters", lower = 16 ,upper = 128)
    conv_1_BatchNormalization = CSH.UniformIntegerHyperparameter(name = "conv_1_BatchNormalization", lower = 0,upper = 1)
    conv_1_kernel_size = CSH.UniformIntegerHyperparameter(name = "conv_1_kernel_size", lower =1 ,upper =16)




    conv_2_type = CSH.Constant(name = "conv_2_type", value = "Conv1D")
    conv_2_padding = CSH.Constant(name = "conv_2_padding",value = "same")
    conv_2_filters = CSH.UniformIntegerHyperparameter(name = "conv_2_filters", lower = 16 ,upper = 128)
    conv_2_BatchNormalization = CSH.UniformIntegerHyperparameter(name = "conv_2_BatchNormalization", lower = 0,upper = 1)
    conv_2_kernel_size = CSH.UniformIntegerHyperparameter(name = "conv_2_kernel_size", lower =1 ,upper =16)

    conv_3_type = CSH.Constant(name = "conv_3_type", value = "Conv1D")
    conv_3_padding = CSH.Constant(name = "conv_3_padding",value = "same")
    conv_3_filters = CSH.UniformIntegerHyperparameter(name = "conv_3_filters", lower = 16 ,upper = 128)
    conv_3_BatchNormalization = CSH.UniformIntegerHyperparameter(name = "conv_3_BatchNormalization", lower = 0,upper = 1)
    conv_3_kernel_size = CSH.UniformIntegerHyperparameter(name = "conv_3_kernel_size", lower =1 ,upper =16)

    conv_4_type = CSH.Constant(name = "conv_4_type", value = "Conv1D")
    conv_4_padding = CSH.Constant(name = "conv_4_padding",value = "same")
    conv_4_filters = CSH.UniformIntegerHyperparameter(name = "conv_4_filters", lower = 16 ,upper = 128)
    conv_4_BatchNormalization = CSH.UniformIntegerHyperparameter(name = "conv_4_BatchNormalization", lower = 0,upper = 1)
    conv_4_kernel_size = CSH.UniformIntegerHyperparameter(name = "conv_4_kernel_size", lower =1 ,upper =16)

    conv_5_type = CSH.Constant(name = "conv_5_type", value = "Conv1D")
    conv_5_padding = CSH.Constant(name = "conv_5_padding",value = "same")
    conv_5_filters = CSH.UniformIntegerHyperparameter(name = "conv_5_filters", lower = 16 ,upper = 128)
    conv_5_BatchNormalization = CSH.UniformIntegerHyperparameter(name = "conv_5_BatchNormalization", lower = 0,upper = 1)
    conv_5_kernel_size = CSH.UniformIntegerHyperparameter(name = "conv_5_kernel_size", lower =1 ,upper =16)
    
    dense_1_type = CSH.Constant(name = "dense_1_type", value = "Dense")
    dense_1_units = CSH.UniformIntegerHyperparameter(name = "dense_1_units", lower = 16, upper = 228)

    dense_2_type = CSH.Constant(name = "dense_2_type", value = "Dense")
    dense_2_units = CSH.UniformIntegerHyperparameter(name = "dense_2_units", lower = 16, upper = 128)

    dense_3_type = CSH.Constant(name = "dense_3_type", value = "Dense")
    dense_3_units = CSH.UniformIntegerHyperparameter(name = "dense_3_units", lower = 16, upper = 128)
    
    #layer_4_padding = CSH.Constant(name = "padding",value = "same")
    #layer_4_filters = CSH.UniformIntegrerHyperparameter(name = "filters", lower = 16 ,upper = 128)
    #layer_4_BatchNormalization = CSH.UniformIntegrerHyperparameter(name = "BatchNormalization", lower = 0,upper = 1)
    #layer_4_kernel_size = CSH.UniformIntegrerHyperparameter(name = "kernel_size_size", lower =1 ,upper =6)

    ####List of Hyperparameters

    hp_list = [
        conv_1_type,
        conv_1_padding,
        conv_1_filters,
        conv_1_BatchNormalization,
        conv_1_kernel_size,
        conv_2_type,
        conv_2_padding,
        conv_2_filters,
        conv_2_BatchNormalization,
        conv_2_kernel_size,
        conv_3_type,
        conv_3_padding,
        conv_3_filters,
        conv_3_BatchNormalization,
        conv_3_kernel_size,
        conv_4_type,
        conv_4_padding,
        conv_4_filters,
        conv_4_BatchNormalization,
        conv_4_kernel_size,
        conv_5_type,
        conv_5_padding,
        conv_5_filters,
        conv_5_BatchNormalization,
        conv_5_kernel_size,
        window_size,
        dense_1_type,
        dense_1_units,
        dense_2_type,
        dense_2_units,
        dense_3_type,
        dense_3_units,
        epochs,
        batch_size,
        optimiser,
        lr,
        num_conv_layers,
        num_dense_layers
    ]
    cs.add_hyperparameters(hp_list)
    conv_hp_list = [    
       "layer_@_type",
       "layer_@_padding",
       "layer_@_filters",
       "layer_@_BatchNormalization",
       "layer_@_kernel_size"]

    for layer in range(max_conv_layers):
        layer_parameter_list = generate_layer_hp(layer)
        batch_add_cond(cs,layer_parameter_list,layer)

    cs.add_hyperparameters(hp_list)
    dense_hp_list = [    

       "layer_@_type",
       "layer_@_units"]
    for layer in range(max_dense_layers):
        layer_parameter_list = generate_layer_hp(layer)
        batch_add_cond(cs,layer_parameter_list,layer)
    return cs
