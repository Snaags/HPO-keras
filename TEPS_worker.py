import numpy as np 
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from window import window_array
"""
## Build a model

We build a Fully Convolutional Neural Network originally proposed in
[this paper](https://arxiv.org/abs/1611.06455).
The implementation is based on the TF 2 version provided
[here](https://github.com/hfawaz/dl-4-tsc/).
The following hyperparameters (kernel_size, filters, the usage of BatchNorm) were found
via random search using [KerasTuner](https://github.com/keras-team/keras-tuner).

"""

def load_dataset():
    path = "/home/snaags/scripts/HPO-keras/datasets/TEPS/"
    x_train = np.load(path+"x_train.npy")
    y_train =    np.load(path+"y_train.npy")
    x_test =np.load(path+"x_test.npy")
    y_test = np.load(path+"y_test.npy")        
    return x_train, y_train,x_test,y_test


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
        layer = function(**layer_args,data_format = "channels_first")(previous_layer)
        layer = keras.layers.BatchNormalization()(layer)
        layer = keras.layers.ReLU()(layer)
    else:
        layer_args.pop("BatchNormalization")
        layer = function(**layer_args)(previous_layer)
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
   
    return layer



def build_layer(layer_number : int , previous_layer, hyperparameter_conf):

    layer_type = hyperparameter_conf["layer_"+str(layer_number)+"_type"] 
    if layer_number > 1: 
        previous_type = hyperparameter_conf["layer_"+str(layer_number-1)+"_type"] 
    else:
        previous_type = None
    hyperparameters = hyperparameter_conf
    layer_args = dict() 
    for parameter_name in hyperparameters:
        if "layer_"+str(layer_number) in parameter_name and "_type" not in parameter_name :
            layer_args[parameter_name.replace("layer_"+str(layer_number)+"_",'')] = hyperparameters[parameter_name]
    print(layer_args) 
    function = eval("keras.layers."+layer_type)
    print(previous_layer )
    if layer_type == "Dense" and previous_type == "Conv1D":
            
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
    for layer in range(1,hyperparameters["num_conv_layers"]+1):
        print(layer)
        layers.append(build_conv_layer(layer,layers[-1],hyperparameters) )
    for layer in range(1,hyperparameters["num_dense_layers"]+1):
        print(layer)
        layers.append(build_dense_layer(layer,layers[-1],hyperparameters) )
    layers.append(keras.layers.Dense(output_size,activation = "softmax")(layers[-1]))
    return keras.models.Model(inputs=input_layer, outputs=layers[-1])


def main(hyperparameter,budget):
    
        

    train_samples = int(900000)
    test_samples = 90000
    num_classes = 21
    x_train,y_train,x_test,y_test = load_dataset()
    x_train,y_train = window_array(x_train[:1000000] , y_train[:1000000], hyperparameter["window_size"])
    x_test,y_test = window_array(x_test[:100000] , y_test[:100000], hyperparameter["window_size"])
    classes = np.unique(y_train)
    print(x_train.shape)
    x_train = np.squeeze(x_train)
    x_test = np.squeeze(x_test)
    x_train,y_train = shuffle(x_train,y_train)
    x_test,y_test = shuffle(x_test,y_test,n_samples = test_samples)
    print(x_train.shape)
    model = make_model(input_shape=x_train.shape[1:],output_size = num_classes,hyperparameters = hyperparameter)
    keras.utils.plot_model(model, show_shapes=True)
        
    """
    ## Train the model
    
    """
    num_classes = 21
    epochs = int(budget)
    batch_size = 256 
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "best_model.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]
    opt = keras.optimizers.Adam(learning_rate = hyperparameter["optimiser_lr"])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    history = model.fit(
        x_train,
        y_train,
        shuffle = True,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.1,
        verbose=1,
    )
    
    """
    ## Evaluate model on test data
    """
    
    model = keras.models.load_model("best_model.h5")
    
    test_loss, test_acc = model.evaluate(x_test, y_test)
    
    print("Test accuracy", test_acc)
    print("Test loss", test_loss)
    
    """
    ## Plot the model's training and validation loss
    """
    

    return test_loss, test_acc 
    """
    We can see how the training accuracy reaches almost 0.95 after 100 epochs.
    However, by observing the validation accuracy we can see how the network still needs
    training until it reaches almost 0.97 for both the validation and the training accuracy
    after 200 epochs. Beyond the 200th epoch, if we continue on training, the validation
    accuracy will start decreasing while the training accuracy will continue on increasing:
    the model starts overfitting.
    """
if __name__ == "__main__":
    hyperparameter = {"batch_size": 32, "epochs": 50, 
    "layer_1_BatchNormalization": 0, "layer_1_filters": 109, "layer_1_kernel_size": 16, "layer_1_padding": "same", "layer_1_type": "Conv1D", 
    "layer_2_BatchNormalization": 1, "layer_2_filters": 93, "layer_2_kernel_size": 8, "layer_2_padding": "same", "layer_2_type": "Conv1D", 
    "layer_3_BatchNormalization": 1, "layer_3_filters": 21, "layer_3_kernel_size": 4, "layer_3_padding": "same", "layer_3_type": "Conv1D", 
    "layer_4_type": "Dense", "layer_4_units": 35, "num_layers": 4,
     "window_size":16, "optimiser": "Adam", "optimiser_lr": 1.2028420169154692e-05}
    main(hyperparameter,20 )
