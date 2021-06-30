import numpy as np 

from tensorflow import keras
import matplotlib.pyplot as plt
from keras_model import make_model
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
    path = "/home/snaags/scripts/HPO-keras/datasets/Ford/processed/"
    x_train = np.load(path+"x_train.npy")
    y_train =    np.load(path+"y_train.npy")
    x_test =np.load(path+"x_test.npy")
    y_test = np.load(path+"y_test.npy")        
    return x_train, y_train,x_test,y_test


 


def main(hyperparameter,budget= 150):


    ###TEMP!!!######



 
    batch_size = 32 
    x_train,y_train,x_test,y_test = load_dataset()
    if hyperparameter["window_size"] > 1:
        x_train,y_train = window_array(x_train, y_train, hyperparameter["window_size"])
        x_test,y_test = window_array(x_test, y_test, hyperparameter["window_size"])
        x_train = np.squeeze(x_train)
        x_test = np.squeeze(x_test)
    else:
        x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
        x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    classes = np.unique(y_train)
    x_train,y_train = shuffle(x_train,y_train)
    x_test,y_test = shuffle(x_test,y_test)
    """
    print(x_train.shape)
    train=keras.preprocessing.timeseries_dataset_from_array(x_train[:int(len(y_train)*0.8),:],y_train[:int(len(y_train)*0.8)],sequence_length =hyperparameter["window"],batch_size = batch_size )
    val = keras.preprocessing.timeseries_dataset_from_array(x_train[int(len(y_train)*0.8):,:],y_train[int(len(y_train)*0.8):],hyperparameter["window"],batch_size = batch_size)
    test = keras.preprocessing.timeseries_dataset_from_array(x_test,y_test,hyperparameter["window"],batch_size = batch_size)
    """
    num_classes = 2
    epochs = int(100)
    ##Just added option to apply featurewise convolution as well as timewise convolution
    model = make_model(input_shape=(x_train.shape[1:]),output_size = num_classes,hyperparameters=hyperparameter, timewise_convolution = True)
    
    keras.utils.plot_model(model, show_shapes=True)
    id = 0
    for i in hyperparameter:
        if type(hyperparameter[i]) == int:
            id += hyperparameter[i]
    file_id = "best_model"+str(id)+".h5"
    if epochs > 1:
        callbacks = [
        keras.callbacks.ModelCheckpoint(
            file_id, save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, verbose=1),
        ]
    else:
        callbacks = []

    opt = keras.optimizers.Adam(learning_rate = hyperparameter["optimiser_lr"])
    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
        
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size = batch_size, 
        epochs=epochs,
        verbose=1,
        shuffle = True,
        callbacks = callbacks,validation_split = 0.2
    )
    
    """
    ## Evaluate model on test data
    """
    
    if epochs > 1:
        model = keras.models.load_model(file_id)
    test_loss, test_acc = model.evaluate(x_test,y_test)
    
    print("Test accuracy", test_acc)
    print("Test loss", test_loss)
    
    """
    ## Plot the model's training and validation loss
    """
    
    
    return  test_acc, test_loss
    """
    We can see how the training accuracy reaches almost 0.95 after 100 epochs.
    However, by observing the validation accuracy we can see how the network still needs
    training until it reaches almost 0.97 for both the validation and the training accuracy
    after 200 epochs. Beyond the 200th epoch, if we continue on training, the validation
    accuracy will start decreasing while the training accuracy will continue on increasing:
    the model starts overfitting.
    """
if __name__ == "__main__":
    '''    hyperparameter = {"batch_size": 32, "epochs": 50, 
    "layer_1_BatchNormalization": 1, "layer_1_filters": 100, "layer_1_kernel_size": 2, "layer_1_padding": "same", "layer_1_type": "Conv1D", 
    "layer_2_BatchNormalization": 1, "layer_2_filters": 80, "layer_2_kernel_size": 2, "layer_2_padding": "same", "layer_2_type": "Conv1D", 
    "layer_3_BatchNormalization": 1, "layer_3_filters": 20, "layer_3_kernel_size": 6, "layer_3_padding": "same", "layer_3_type": "Conv1D", 
    "layer_4_type": "Dense", "layer_4_units": 20, "num_layers": 4,
    "window":1, "optimiser": "Adam", "optimiser_lr": 1.2028420169154692e-05}'''

    hyperparameter = {"batch_size": 32, 
    "conv_1_BatchNormalization": 0, "conv_1_filters": 32, "conv_1_kernel_size": 1, "conv_1_padding": "same", "conv_1_type": "Conv1D", 
    "dense_1_type": "Dense", "dense_1_units": 7,
    "dense_2_type": "Dense", "dense_2_units": 32,
    "dense_3_type": "Dense", "dense_3_units": 16,
     "epochs": 50, "num_conv_layers": 5, "num_dense_layers": 1, "optimiser": "Adam", "optimiser_lr": 0.000053908363691981195, 
     "window_size": 1,
     "conv_2_BatchNormalization": 1, "conv_2_filters": 32, "conv_2_kernel_size": 2, "conv_2_padding": "same", "conv_2_type": "Conv1D", 
     "conv_3_BatchNormalization": 1, "conv_3_filters": 32, "conv_3_kernel_size": 4, "conv_3_padding": "same", "conv_3_type": "Conv1D",
     "conv_4_BatchNormalization": 1, "conv_4_filters": 32, "conv_4_kernel_size": 8, "conv_4_padding": "same", "conv_4_type": "Conv1D",
     "conv_5_BatchNormalization": 1, "conv_5_filters": 32, "conv_5_kernel_size": 16, "conv_5_padding": "same", "conv_5_type": "Conv1D"} 

    hyperparameters ={
  "batch_size": 32,
  "conv_1_BatchNormalization": 0,
  "conv_1_activity_regularizer": 5.2411491637761416e-05,
  "conv_1_bias_regularizer": 1.1823592840766283e-08,
  "conv_1_filters": 88,
  "conv_1_kernel_regularizer": 0.0010205704098090421,
  "conv_1_kernel_size": 6,
  "conv_1_padding": 'same',
  "conv_1_type": 'Conv1D',
  "conv_2_BatchNormalization": 1,
  "conv_2_activity_regularizer": 8.507048520250179e-08,
  "conv_2_bias_regularizer": 1.992171519796033e-06,
  "conv_2_filters": 66,
  "conv_2_kernel_regularizer": 2.6539162874602007e-05,
  "conv_2_kernel_size": 14,
  "conv_2_padding": 'same',
  "conv_2_type": 'Conv1D',
  "dense_1_activity_regularizer": 0.0004536934202033339,
  "dense_1_bias_regularizer": 0.0005522261017267918,
  "dense_1_kernel_regularizer": 0.00046103262364006136,
  "dense_1_type": 'Dense',
  "dense_1_units": 7,
  "dense_2_activity_regularizer": 3.551423061679453e-08,
  "dense_2_bias_regularizer": 6.1259045639414e-08,
  "dense_2_kernel_regularizer": 0.0007199455813671363,
  "dense_2_type": 'Dense',
  "dense_2_units": 124,
  "dense_3_activity_regularizer": 0.21534214512991076,
  "dense_3_bias_regularizer": 4.636632285365818e-08,
  "dense_3_kernel_regularizer": 0.10595695499057485,
  "dense_3_type": 'Dense',
  "dense_3_units": 35,
  "epochs": 50,
  "num_conv_layers": 2,
  "num_dense_layers": 3,
  "optimiser": 'Adam',
  "optimiser_lr": 0.000020987333617918545,
  "window_size": 1}



    main(hyperparameters)
