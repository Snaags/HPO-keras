import tensorflow.keras as keras

def format_reg(hyperparameters):
    for i in hyperparameters:
        if "regularizer" in i:
            hyperparameters[i] = keras.regularizers.l2(hyperparameters[i]) 


def build_conv_layer(layer_number : int , previous_layer, hyperparameter_conf):

    layer_type = hyperparameter_conf["conv_"+str(layer_number)+"_type"] 
    if layer_number > 1: 
        previous_type = hyperparameter_conf["conv_"+str(layer_number-1)+"_type"] 
    else:
        previous_type = None
    hyperparameters = hyperparameter_conf
    layer_args = dict() 
    for parameter_name in hyperparameters:
        if "conv_"+str(layer_number) in parameter_name and "_type" not in parameter_name :
            layer_args[parameter_name.replace("conv_"+str(layer_number)+"_",'')] = hyperparameters[parameter_name]
    print(layer_args) 
    function = eval("keras.layers."+layer_type)
    print(previous_layer )
    if layer_args["BatchNormalization"] == 1:
        layer_args.pop("BatchNormalization")
        layer = function(**layer_args)(previous_layer)
        layer = keras.layers.MaxPooling1D(pool_size=2, strides=None, padding="valid",data_format="channels_last")(layer)
        layer = keras.layers.BatchNormalization()(layer)
        layer = keras.layers.ReLU()(layer)
    else:
        layer_args.pop("BatchNormalization")
        layer = function(**layer_args)(previous_layer)
        layer = keras.layers.MaxPooling1D(pool_size=2, strides=None, padding="valid",data_format="channels_last")(layer)
        layer = keras.layers.ReLU()(layer)
    return layer


def build_dense_layer(layer_number : int , previous_layer, hyperparameter_conf):

    layer_type = hyperparameter_conf["dense_"+str(layer_number)+"_type"] 
    if layer_number > 1: 
        previous_type = hyperparameter_conf["dense_"+str(layer_number-1)+"_type"] 
    else:
        previous_type = "Conv1D"
    hyperparameters = hyperparameter_conf
    layer_args = dict() 
    for parameter_name in hyperparameters:
        if "dense_"+str(layer_number) in parameter_name and "_type" not in parameter_name :
            layer_args[parameter_name.replace("dense_"+str(layer_number)+"_",'')] = hyperparameters[parameter_name]
    print(layer_args) 
    function = eval("keras.layers."+layer_type)
    print(previous_layer )
    if previous_type == "Conv1D":     
        gap = keras.layers.GlobalAveragePooling1D()(previous_layer)
        layer = function(**layer_args,activation = "ReLU")(gap)
    else:
        layer = function(**layer_args,activation = "ReLU")(previous_layer)
        
    return layer





def make_model(input_shape, output_size,hyperparameters):
    
    input_layer = keras.layers.Input(input_shape)
    format_reg(hyperparameters)
    layers = [input_layer]#List used to link layers together
    ##Loop Convolutional Layers
    for layer in range(1,hyperparameters["num_conv_layers"]+1):
        print(layer)
        layers.append(build_conv_layer(layer,layers[-1],hyperparameters) )

    #Loop Dense layers
    for layer in range(1,hyperparameters["num_dense_layers"]+1):
        print(layer)
        layers.append(build_dense_layer(layer,layers[-1],hyperparameters) )

    #Add read-out layer 
    layers.append(keras.layers.Dense(output_size,activation = "softmax")(layers[-1]))
    return keras.models.Model(inputs=input_layer, outputs=layers[-1])


 