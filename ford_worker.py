import numpy as np 
from tensorflow import keras
import matplotlib.pyplot as plt
from keras_model import make_model
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
    path = "/home/snaags/scripts/HPO-keras/datasets/Ford/processed/"
    x_train = np.load(path+"x_train.npy")
    y_train =    np.load(path+"y_train.npy")
    x_test =np.load(path+"x_test.npy")
    y_test = np.load(path+"y_test.npy")        
    return x_train, y_train,x_test,y_test


 


def main(hyperparameter,budget= 150):
    
    batch_size = 32 


    x_train,y_train,x_test,y_test = load_dataset()
    """
    print(x_train.shape)
    train=keras.preprocessing.timeseries_dataset_from_array(x_train[:int(len(y_train)*0.8),:],y_train[:int(len(y_train)*0.8)],sequence_length =hyperparameter["window"],batch_size = batch_size )
    val = keras.preprocessing.timeseries_dataset_from_array(x_train[int(len(y_train)*0.8):,:],y_train[int(len(y_train)*0.8):],hyperparameter["window"],batch_size = batch_size)
    test = keras.preprocessing.timeseries_dataset_from_array(x_test,y_test,hyperparameter["window"],batch_size = batch_size)
    """
    print(type(x_train))
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    """
    ## Train the model
    
    """
    print("budget: ",budget)
    num_classes = 2
    epochs = 50
    model = make_model(input_shape=(x_train.shape[1:]),output_size = num_classes,hyperparameters=hyperparameter)
    
    keras.utils.plot_model(model, show_shapes=True)
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "best_model.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
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
    #test_loss, test_acc = model.evaluate(train)
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
    hyperparameter = {"batch_size": 32, "epochs": 50, 
    "layer_1_BatchNormalization": 1, "layer_1_filters": 100, "layer_1_kernel_size": 2, "layer_1_padding": "same", "layer_1_type": "Conv1D", 
    "layer_2_BatchNormalization": 1, "layer_2_filters": 80, "layer_2_kernel_size": 2, "layer_2_padding": "same", "layer_2_type": "Conv1D", 
    "layer_3_BatchNormalization": 1, "layer_3_filters": 20, "layer_3_kernel_size": 6, "layer_3_padding": "same", "layer_3_type": "Conv1D", 
    "layer_4_type": "Dense", "layer_4_units": 20, "num_layers": 4,
    "window":1, "optimiser": "Adam", "optimiser_lr": 1.2028420169154692e-05}

    main(hyperparameter)
