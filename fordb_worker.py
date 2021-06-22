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
    path = "/home/snaags/scripts/HPO-keras/datasets/Ford/processedb/"
    x_train = np.load(path+"x_train.npy")
    y_train =    np.load(path+"y_train.npy")
    x_test =np.load(path+"x_test.npy")
    y_test = np.load(path+"y_test.npy")        
    return x_train, y_train,x_test,y_test


 


def main(hyperparameter,budget= 150):
    
    batch_size = 64 
    print(hyperparameter)
    x_train,y_train,x_test,y_test = load_dataset()
    x_train,y_train = window_array(x_train, y_train, hyperparameter["window_size"])
    x_test,y_test = window_array(x_test, y_test, hyperparameter["window_size"])
    classes = np.unique(y_train)
    print(x_train.shape)
    x_train = np.squeeze(x_train)
    x_test = np.squeeze(x_test)
    x_train,y_train = shuffle(x_train,y_train)
    x_test,y_test = shuffle(x_test,y_test)
    """
    print(x_train.shape)
    train=keras.preprocessing.timeseries_dataset_from_array(x_train[:int(len(y_train)*0.8),:],y_train[:int(len(y_train)*0.8)],sequence_length =hyperparameter["window"],batch_size = batch_size )
    val = keras.preprocessing.timeseries_dataset_from_array(x_train[int(len(y_train)*0.8):,:],y_train[int(len(y_train)*0.8):],hyperparameter["window"],batch_size = batch_size)
    test = keras.preprocessing.timeseries_dataset_from_array(x_test,y_test,hyperparameter["window"],batch_size = batch_size)
    """
    print("budget: ",budget)
    num_classes = 2
    epochs = int(50)
    model = make_model(input_shape=(x_train.shape[1:]),output_size = num_classes,hyperparameters=hyperparameter)
    
    keras.utils.plot_model(model, show_shapes=True)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "best_model.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, verbose=1),
    ]
    model.compile(
        optimizer="adam",
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
    
    
    model = keras.models.load_model("best_model.h5")
    test_loss, test_acc = model.evaluate(x_test,y_test)
    full_loss, full_acc = model.evaluate(x_test,y_test)
    
    print("Test accuracy", test_acc)
    print("Test loss", test_loss)
    print("full accuracy", full_acc)
    print("full loss", full_loss)
    
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
    '''    hyperparameter = {"batch_size": 32, "epochs": 50, 
    "layer_1_BatchNormalization": 1, "layer_1_filters": 100, "layer_1_kernel_size": 2, "layer_1_padding": "same", "layer_1_type": "Conv1D", 
    "layer_2_BatchNormalization": 1, "layer_2_filters": 80, "layer_2_kernel_size": 2, "layer_2_padding": "same", "layer_2_type": "Conv1D", 
    "layer_3_BatchNormalization": 1, "layer_3_filters": 20, "layer_3_kernel_size": 6, "layer_3_padding": "same", "layer_3_type": "Conv1D", 
    "layer_4_type": "Dense", "layer_4_units": 20, "num_layers": 4,
    "window":1, "optimiser": "Adam", "optimiser_lr": 1.2028420169154692e-05}'''

    hyperparameter = {"batch_size": 32, 
    "conv_1_BatchNormalization": 0, "conv_1_filters": 2, "conv_1_kernel_size": 1, "conv_1_padding": "same", "conv_1_type": "Conv1D", 
    "dense_1_type": "Dense", "dense_1_units": 8,
     "epochs": 50, "num_conv_layers": 5, "num_dense_layers": 1, "optimiser": "Adam", "optimiser_lr": 0.000053908363691981195, 
     "window_size": 2, 
     "conv_2_BatchNormalization": 1, "conv_2_filters": 2, "conv_2_kernel_size": 2, "conv_2_padding": "same", "conv_2_type": "Conv1D", 
     "conv_3_BatchNormalization": 1, "conv_3_filters": 2, "conv_3_kernel_size": 4, "conv_3_padding": "same", "conv_3_type": "Conv1D",
     "conv_4_BatchNormalization": 1, "conv_4_filters": 2, "conv_4_kernel_size": 8, "conv_4_padding": "same", "conv_4_type": "Conv1D",
     "conv_5_BatchNormalization": 1, "conv_5_filters": 2, "conv_5_kernel_size": 16, "conv_5_padding": "same", "conv_5_type": "Conv1D"} 



    main(hyperparameter)
