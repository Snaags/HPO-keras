import numpy as np 
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
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
            layer = function(**layer_args)(previous_layer)
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


def main(hyperparameter,budget):
    
        

    train_samples = int(budget)
    test_samples = 100000
    num_classes = 21
    x_train,y_train,x_test,y_test = load_dataset()
    print(x_train.shape)
    classes = np.unique(y_train)
    x_train,y_train = shuffle(x_train,y_train,n_samples = train_samples)
    x_full,y_full = shuffle(x_test,y_test,n_samples = test_samples)
    x_test,y_test = shuffle(x_test,y_test, n_samples = (25*21))
    print(x_train.shape)
    model = make_model(input_shape=x_train.shape[1:],output_size = num_classes,hyperparameters = hyperparameter)
    keras.utils.plot_model(model, show_shapes=True)
        
    """
    ## Train the model
    
    """
    num_classes = 21
    epochs = 1 
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
    full_loss, full_acc = model.evaluate(x_full, y_full)
    
    print("Test accuracy", test_acc)
    print("Test loss", test_loss)
    
    """
    ## Plot the model's training and validation loss
    """
    

    return test_loss, full_loss 
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
    "layer_1_BatchNormalization": 0, "layer_1_filters": 109, "layer_1_kernel_size": 2, "layer_1_padding": "same", "layer_1_type": "Conv1D", 
    "layer_2_BatchNormalization": 1, "layer_2_filters": 93, "layer_2_kernel_size": 2, "layer_2_padding": "same", "layer_2_type": "Conv1D", 
    "layer_3_BatchNormalization": 1, "layer_3_filters": 21, "layer_3_kernel_size": 6, "layer_3_padding": "same", "layer_3_type": "Conv1D", 
    "layer_4_type": "Dense", "layer_4_units": 35, "num_layers": 4,
     "optimiser": "Adam", "optimiser_lr": 1.2028420169154692e-05}
    main(hyperparameter )
