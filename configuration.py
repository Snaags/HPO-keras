import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import os
import csv
import time

cs = CS.ConfigurationSpace()
max_window_size = 6

batch_size	= 	CSH.UniformIntegerHyperparameter(name = "batch_size", lower = 16, upper = 64) #batch_size
epochs 		=	CSH.UniformIntegerHyperparameter(name = "epochs", lower = 20, upper = 100) #epochs
window_size 		=	CSH.UniformIntegerHyperparameter(name = "window_size", lower = 6, upper = 12) #window_size
window_size = CSH.Constant(name = "window_size", value = 3)
layer_1_kernel_size	=	CSH.UniformIntegerHyperparameter(name = "layer_1_kernel_size", lower = 3, upper = 6) #conv_kernel_size

epochs 		=	CSH.Constant(name = "epochs",value = 32) #epochs
layer_2_kernel_size	=	CSH.UniformIntegerHyperparameter(name = "layer_2_kernel_size", lower = 3, upper = 6) #conv_kernel_size


layer_4_type  =	CSH.Constant(name = "layer_4_type", value = "nn.Linear")
layer_3_kernel_size	=	CSH.UniformIntegerHyperparameter(name = "layer_3_kernel_size", lower = 3, upper = 6) #conv_kernel_size

layer_1_kernel_size	=	CSH.Constant(name = "layer_1_kernel_size", value = 3)
layer_2_kernel_size	=	CSH.Constant(name = "layer_2_kernel_size", value = 3)
layer_3_kernel_size	=	CSH.Constant(name = "layer_3_kernel_size", value = 3)
batch_size =	CSH.Constant(name = "batch_size", value = 32)
#
#

num_layers  =	CSH.Constant(name = "num_layers", value = 4)

#layer_1_type = CSH.CatagoricalHyperparameter(name = "layer_1_type" choices  = [None, "nn.Linear","nn.Conv1d"])
layer_1_type  =	CSH.Constant(name = "layer_1_type", value = "nn.Conv1d")
layer_2_type  =	CSH.Constant(name = "layer_2_type", value = "nn.Conv1d")
layer_3_type  =	CSH.Constant(name = "layer_3_type", value = "nn.Conv1d")

layer_1_out_channels		=	CSH.UniformIntegerHyperparameter(name = "layer_1_out_channels", lower = 40, upper = 80) #conv_layer_out_channels
layer_2_out_channels		=	CSH.UniformIntegerHyperparameter(name = "layer_2_out_channels", lower = 40, upper = 80) #conv_layer_out_channels
layer_3_out_channels		=	CSH.UniformIntegerHyperparameter(name = "layer_3_out_channels", lower = 40, upper = 80) #conv_layer_out_channels
layer_4_out_features		=	CSH.UniformIntegerHyperparameter(name = "layer_4_out_features",lower = 40, upper = 80) #hidden_layer_size


activation_function = CSH.Constant(name = "activation_function", value = "ReLU") #activation_function
optimiser   		=	CSH.Constant(name = "optimiser", value = "Adam"		) #optimizer				
dropout				=	CSH.UniformFloatHyperparameter(name = "dropout",	lower = 0,upper = 0.6 ,log = False ) #dropout
lr 					=	CSH.UniformFloatHyperparameter(name = "optimiser_lr",			lower = 1e-8,upper = 5e-1 ,log = True ) #lr
#lr_decay			=	CSH.UniformFloatHyperparameter(name = "optimiser_lr_decay",	lower = 1e-10,upper = 5e-2 ,log = True ) #lr_decay
#weight_decay		=	CSH.UniformFloatHyperparameter(name = "optimiser_weight_decay",lower = 1e-10,upper = 5e-1 ,log = True ) #weight_decay
#lambd				=	CSH.UniformFloatHyperparameter(name = "optimiser_lambd",		lower = 1e-8,upper = 5e-2 ,log = True ) #lambd
#momentum			=	CSH.UniformFloatHyperparameter(name = "optimiser_momentum",	lower = 1e-8,upper = 5e-2 ,log = True ) #momentum

hp_list = [
	layer_1_out_channels,
	layer_1_kernel_size,
	layer_2_kernel_size,
	layer_2_out_channels,
	layer_3_kernel_size,
	layer_3_out_channels,
	num_layers,
	activation_function,
	optimiser,
	dropout,
	lr,
	#lr_decay,
	#weight_decay,
	#lambd,
	#momentum,
	layer_4_out_features,
	batch_size,
	epochs,
	layer_1_type,
	layer_2_type,
	layer_3_type,
	layer_4_type,
	window_size
	]


cs.add_hyperparameters(hp_list)

"""
for i in range(1,max_window_size+1):
	print(i)
	forbidden_clause_a = CS.ForbiddenEqualsClause(window_size, i)
	forbidden_clause_b = CS.ForbiddenInClause(layer_1_kernel_size, list(range(i,max_window_size)))
	forbidden_clause_c = CS.ForbiddenInClause(layer_1_kernel_size, list(range(i,max_window_size)))
	forbidden_clause_d = CS.ForbiddenInClause(layer_1_kernel_size, list(range(i,max_window_size)))
	forbidden_clause_e = CS.ForbiddenInClause(layer_1_kernel_size, list(range(i,max_window_size)))
	forbidden_clause1 = CS.ForbiddenAndConjunction(forbidden_clause_a, forbidden_clause_b)
	forbidden_clause2 = CS.ForbiddenAndConjunction(forbidden_clause_a, forbidden_clause_c)
	forbidden_clause3 = CS.ForbiddenAndConjunction(forbidden_clause_a, forbidden_clause_d)
	forbidden_clause4 = CS.ForbiddenAndConjunction(forbidden_clause_a, forbidden_clause_e)
	cs.add_forbidden_clause(forbidden_clause1)
	cs.add_forbidden_clause(forbidden_clause2)
	cs.add_forbidden_clause(forbidden_clause3)
	cs.add_forbidden_clause(forbidden_clause4)
"""


"""
for layer_number in range(1,number_of_layers):

	l = CSH.Constant(name = "layer_"+str(layer_number)+"_type", value = "nn.Conv1d")
	cs.add_hyperparameter(l)


	cond = CS.LessThanCondition(l, num_layers, layer_number)
	cs.add_condition(cond)
"""
def load_evaluations(filename):
	x = []

	with open(filename) as csvfile:
	    reader = csv.reader(csvfile, delimiter=',')
	    for i,row in enumerate(reader):
    		hold =[]
	    	hold.append(eval(row[1]))
	    	hold.append(float(row[0]))
	    	x.append(hold)
	return x




scores = []
validation = []
log_file = "HPO_log.txt"
score_file = "TEPS_Random_scores.csv"
config_file = "TEPS_Random_configs.csv"


def save_to_file(csv_file,dict):
	with open(csv_file, 'w') as csvfile:
	    writer = csv.writer(csvfile)
	    for ID in config_ID_dict:
	        writer.writerow([ID,dict[ID]])


from HPO_utils import Dispatcher
from algorithms import GA
import train_function
from matplotlib import pyplot as plt
config_list = cs.sample_configuration(size = 10)
current_iter = 0


ga = GA(pop=10,elite = 0.3)
##Init ID dict
config_ID_dict = {} 
scores_ID_dict = {}
for index, conf in enumerate(config_list):
	config_ID_dict[index] = conf







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
	#config_list = cs.sample_configuration(size = 32)
	config_list = ga.mutate(config_ID_dict,scores_ID_dict)

	##Update Configs dictionary
	start_index = max(scores_ID_dict.keys())  + 1
	for index, conf in enumerate(config_list):
		idx = start_index + index
		config_ID_dict[idx] = conf

