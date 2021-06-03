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
    max_dense_layers = 4
    cs = CS.ConfigurationSpace()
    ###Training Configuration###
    epochs = CSH.Constant(name = "epochs",value = 50)
    batch_size = CSH.Constant(name = "batch_size",value = 32)
    num_conv_layers = CSH.UniformIntegerHyperparameter(name = "num_conv_layers",lower =1 ,upper = max_conv_layers)
    num_dense_layers = CSH.UniformIntegerHyperparameter(name = "num_dense_layers",lower =1 ,upper = max_conv_layers)


    ###Optimiser###
    optimiser  = CSH.Constant(name = "optimiser", value = "Adam"		) 
    lr =CSH.UniformFloatHyperparameter(name = "optimiser_lr",			lower = 1e-8,upper = 5e-1 ,log = True ) #lr



    window = CSH.UniformIntegerHyperparameter(name = "window", lower = 1 ,upper = 16)
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

    layer_4_type = CSH.Constant(name = "layer_4_type", value = "Conv1D")
    layer_4_padding = CSH.Constant(name = "layer_4_padding",value = "same")
    layer_4_filters = CSH.UniformIntegerHyperparameter(name = "layer_4_filters", lower = 16 ,upper = 128)
    layer_4_BatchNormalization = CSH.UniformIntegerHyperparameter(name = "layer_4_BatchNormalization", lower = 0,upper = 1)
    layer_4_kernel_size = CSH.UniformIntegerHyperparameter(name = "layer_4_kernel_size", lower =1 ,upper =6)

    layer_5_type = CSH.Constant(name = "layer_5_type", value = "Conv1D")
    layer_5_padding = CSH.Constant(name = "layer_5_padding",value = "same")
    layer_5_filters = CSH.UniformIntegerHyperparameter(name = "layer_5_filters", lower = 16 ,upper = 128)
    layer_5_BatchNormalization = CSH.UniformIntegerHyperparameter(name = "layer_5_BatchNormalization", lower = 0,upper = 1)
    layer_5_kernel_size = CSH.UniformIntegerHyperparameter(name = "layer_5_kernel_size", lower =1 ,upper =6)
    
    layer_1_type = CSH.Constant(name = "layer_4_type", value = "Dense")
    layer_1_units = CSH.UniformIntegerHyperparameter(name = "layer_4_units", lower = 16, upper = 228)

    layer_2_type = CSH.Constant(name = "layer_4_type", value = "Dense")
    layer_2_units = CSH.UniformIntegerHyperparameter(name = "layer_4_units", lower = 16, upper = 128)

    layer_3_type = CSH.Constant(name = "layer_4_type", value = "Dense")
    layer_3_units = CSH.UniformIntegerHyperparameter(name = "layer_4_units", lower = 16, upper = 128)
    
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
        window,
        layer_4_type,
        layer_4_units,
        epochs,
        batch_size,
        optimiser,
        lr,
        num_layers
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
