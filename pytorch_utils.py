import torch.nn as nn
import torch
import torch.optim as optim
class Node(nn.Module):
	def __init__(self,layer_type : str, layer_args, model):
		super().__init__()
		self.out_dim = None
		function = eval(layer_type)
		self.type = layer_type
		self.input_size = model.get_input_dim(type_name = self.type)

		self.transition = model.get_conv2lin(self.type)

		if self.type == "nn.Conv1d":
			self.norm = nn.BatchNorm1d(layer_args["out_channels"]).cuda()
			layer_args["padding"] = layer_args["kernel_size"]//2 
		self.sequence_length = model.get_sequence_length()
		self.NN = function(self.input_size,**layer_args).cuda()
		self.act = nn.ReLU().cuda()
		if self.transition == True:
			self.pool = nn.AdaptiveAvgPool1d(1)
	def forward(self, x):

		#Preset Input Layer

		#if self.type == "nn.Linear":

		if self.transition == True:
			x = self.pool(x) 
			x = torch.squeeze(x)
			



		out = self.NN(x)

		if self.type == "nn.Conv1d":
			out = self.norm(out)

		out = self.act(out)
		return out
		

	def get_out_dim(self):
		if self.out_dim == None:
			self.init_output_size()

		return self.out_dim

	def get_type(self):
		return self.type



	def init_output_size(self):

		if self.type == "nn.Conv1d":
			self.out_dim = self.NN(torch.randn(1, self.input_size,self.sequence_length).cuda()).cuda().size()

		elif self.type == "nn.Linear":
			self.out_dim = self.NN(torch.randn(1, self.input_size).cuda()).size()




class Model(nn.Module):
	def __init__(self,output_dim,input_size,seq_len):
		super().__init__()
		self.sequence_length = seq_len

		self.output_dim = output_dim
		self.input_dim = input_size
		self.layers = []
		# Readout layer



	def get_conv2lin(self,l_type):

		if len(self.layers) == 0:
			return False 
		if self.layers[-1].get_type() == "nn.Conv1d" and l_type == "nn.Linear":
			return True
		else:
			return False
	def add_layer(self,node):
		self.layers.append(node)

	def _get_out_dim(self):
		return self.layers[-1].get_out_dim()

	def _get_input_dim_conv_2_lin(self):
		x = self._get_out_dim()
		return torch.prod(torch.tensor(x))

	def get_input_dim(self,type_name):
		if len(self.layers) == 0:
			return self.input_dim
		else:
			return self._get_out_dim()[1]


		"""
		elif self.layers[-1].get_type() == "nn.Conv1d" and type_name == "nn.Conv1d":
			return self._get_out_dim()[1]

		elif self.layers[-1].get_type() == "nn.Conv1d" and type_name == "nn.Linear":
			return self._get_input_dim_conv_2_lin()

		elif self.layers[-1].get_type() == "nn.Linear" and type_name == "nn.Linear":
			return self._get_out_dim()[1]
		"""
			
	def get_sequence_length(self):
		if len(self.layers) == 0:
			return self._get_input_sequence_length()
		elif self.layers[-1].get_type() == "nn.Conv1d":
			return self._get_conv_sequence_length()
		elif self.layers[-1].get_type() == "nn.Linear":
			return 1


	def _get_input_sequence_length(self):
		return self.sequence_length

	def _get_conv_sequence_length(self):
		return self._get_out_dim()[2]


	def build_output_layer(self):
		last_layer_dim = self.get_input_dim("nn.Linear") 
		self.fc = (nn.Linear(last_layer_dim,2))
		self.fcact = nn.Softmax(dim = 0)

	def forward(self, x):

		#Preset Input Layer
		out = x

		for i in self.layers:
			out = i(out)


		#Preset OutputLayer
		out = self.fc(out)


		out = self.fcact(out)
		return out


def build_model(hyperparameter,n_feature,n_classes):
	"""
	This functions builds a the pytorch model based on the ConfigSpace hyperparameters
	Format should be "layer_"+ layer_index +"_" + kwarg/layer type i.e.
			layer_1_nn.Conv2d or layer_3_in_features

	the layer and number is used as a the identifier and is striped from the string and the whats left is eval'd for the function and args
	"""

	#Loops through layers in configuration and creates a kwargs dictionary to be passed to pytorch functions for each layer in the node class
	model = Model(n_classes,n_feature,hyperparameter["layer_1_kernel_size"])

	for layer in range(1,hyperparameter["num_layers"]+1):
		#Get the type of layer
		layer_type = hyperparameter["layer_"+str(layer)+"_type"]

		layer_args = {}

		for i in hyperparameter:

			#Get the arguments for that layer
			if ("layer_"+str(layer) in i) and (i != "layer_"+str(layer)+"_type"):
				#Strip the layer identifier and add to the args dict
				layer_args[i.replace("layer_"+str(layer)+"_",'')] = hyperparameter[i]


		model.add_layer(Node(layer_type,layer_args,model))
	model.build_output_layer()


	return model


def build_optimiser(params,hyperparameter):
	optimiser_args = {"params":params}
	optimiser = eval("optim."+hyperparameter["optimiser"])
	for i in hyperparameter:
		if ("optimiser_" in i) and (i != "optimiser"):
			#Strip the layer identifier and add to the args dict
			optimiser_args[i.replace("optimiser_",'')] = hyperparameter[i]
	return optimiser(**optimiser_args)
