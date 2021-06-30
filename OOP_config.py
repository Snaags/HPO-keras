import ConfigSpace as CS 
import ConfigSpace.hyperparameters as CSH
import os
import csv
import time 
from ConfigStruct import Parameter, Cumulative_Integer_Struct



def init_config():
    
     
    cs = CS.ConfigurationSpace()

    conv_parameters = [
        Parameter("type",               "Constant", lower_or_constant_value = "Conv1D"),
        Parameter("padding",            "Constant" ,lower_or_constant_value = "same"),
        Parameter("filters",            "Integer", 16, 128),
        Parameter("BatchNormalization", "Integer", 0,1),
        Parameter("kernel_size",        "Integer", 1,16),
        Parameter("kernel_regularizer", "Float", 1e-8,5e-1, log = True),
        Parameter("bias_regularizer",   "Float", 1e-8,5e-1, log = True),
        Parameter("activity_regularizer", "Float", 1e-8,5e-1, log = True)]
     
    Cumulative_Integer_Struct(cs,conv_parameters,"conv", "num_conv_layers","Integer", 1, 5).init() 
    
    dense_parameters = [
        Parameter("type",               "Constant", "Dense"),
        Parameter("units",              "Integer", 1,128),
        Parameter("kernel_regularizer", "Float", 1e-8,5e-1, log = True),
        Parameter("bias_regularizer",   "Float", 1e-8,5e-1, log = True),
        Parameter("activity_regularizer", "Float", 1e-8,5e-1, log = True)]
     
    Cumulative_Integer_Struct(cs,dense_parameters,"dense","num_dense_layers","Integer", 1, 3).init() 

    ###Training Configuration###
    epochs = CSH.Constant(name = "epochs",value = 50)
    batch_size = CSH.Constant(name = "batch_size",value = 32)
    ###Optimiser###
    optimiser  = CSH.Constant(name = "optimiser", value = "Adam"		) 
    lr =CSH.UniformFloatHyperparameter(name = "optimiser_lr",			lower = 1e-8,upper = 5e-1 ,log = True )
    window_size = CSH.UniformIntegerHyperparameter(name = "window_size", lower = 1 ,upper = 100)
    window_size = CSH.Constant(name = "window_size", value = 1		) 
    ###Topology Definition]###

    hp_list = [
        window_size,
        epochs,
        batch_size,
        optimiser,
        lr]
    cs.add_hyperparameters(hp_list)
    return cs
