import tensorflow.keras as keras
def build_layer(layer_number : int , previous_layer, hyperparameter_conf):

    layer_type = hyperparameter_conf["layer_"+str(layer_number)+"_type"] 
    hyperparameters = hyperparameter_conf
    hyperparameters.pop("layer_"+str(layer_number)+"_type")
    layer_args = dict() 
    for parameter_name in hyperparameters:
        if "layer_"+str(layer_number) in parameter_name:
            layer_args[parameter_name.replace("layer_"+str(layer_number)+"_",'')] = hyperparameters[parameter_name]
    print(layer_args) 
    function = eval("keras.layers."+layer_type)
    print(previous_layer )
    if layer_type == "Dense":
            
        gap = keras.layers.GlobalAveragePooling1D()(previous_layer)
        layer = function(**layer_args,activation = "ReLU")(gap)
   
    if layer_type == "Conv1D": 
        if layer_args["BatchNormalization"] == 1:
            layer_args.pop("BatchNormalization")
            layer = function(**layer_args,data_format = "channels_first")(previous_layer)
            layer = keras.layers.BatchNormalization()(layer)
            layer = keras.layers.ReLU()(layer)
        else:
            layer_args.pop("BatchNormalization")
            layer = function(**layer_args)(previous_layer)
            layer = keras.layers.ReLU()(layer)
    ##TODO 
        #Add conditional features of different layer types batch pooling etc 
    return layer
    

def make_model(input_shape, output_size,hyperparameters):
    input_layer = keras.layers.Input(input_shape)
    layers = [input_layer]
    ##Layer 1                
    for layer in range(1,hyperparameters["num_layers"]+1):
        print(layer)
        layers.append(build_layer(layer,layers[-1],hyperparameters) )
    layers.append(keras.layers.Dense(output_size,activation = "softmax")(layers[-1]))
    return keras.models.Model(inputs=input_layer, outputs=layers[-1])
