import numpy as np 
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from window import window_array_random
from keras_model import make_model
from sklearn.metrics import confusion_matrix
import pandas as pd

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


def load_dataset_without_fault(fault_nums : list):
    path = "/home/snaags/scripts/HPO-keras/datasets/TEPS/"
    x_train = np.load(path+"x_train.npy")
    y_train =    np.load(path+"y_train.npy")
    x_test =np.load(path+"x_test.npy")
    y_test = np.load(path+"y_test.npy")


    for i in fault_nums:
        idx = np.where(y_train == i)

        y_train = np.delete(y_train,idx,axis = 0)

        x_train = np.delete(x_train,idx,axis = 0)

    for index, i in enumerate(fault_nums):
        i = i - index
        for index, y in enumerate(y_train):
            if y > i:
                y_train[index] = y - 1

    for i in fault_nums:
        idx = np.where(y_test == i)
        y_test = np.delete(y_test,idx[0],axis = 0)
        x_test = np.delete(x_test,idx[0],axis = 0)

    for index, i in enumerate(fault_nums):
        i = i - index
        for index, y in enumerate(y_test):
            if y > i:
                y_test[index] = y - 1

    return x_train, y_train,x_test,y_test

def main(hyperparameter,budget = 10):
    
        

    train_samples = int(900000)
    test_samples = 10000


    x_train,y_train,x_test,y_test = load_dataset()
    num_classes = len(np.unique(y_train))
    classes = np.unique(y_train)

    x_train,y_train = window_array_random(x_train , y_train, hyperparameter["window_size"], 500000)
    x_test,y_test = window_array_random(x_test , y_test, hyperparameter["window_size"],100000)

   
    print("Training classes: ",classes)
    print("Testing classes: ",np.unique(y_test))
    print(x_train.shape)
    x_train = np.squeeze(x_train)
    x_test = np.squeeze(x_test)
    x_train,y_train = shuffle(x_train,y_train)
    x_test,y_test = shuffle(x_test,y_test)
    print(x_train.shape)
    print(hyperparameter)
    model = make_model(input_shape=x_train.shape[1:],output_size = num_classes,hyperparameters = hyperparameter)
    keras.utils.plot_model(model, show_shapes=True)
        
    """
    ## Train the model
    
    """
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
        optimizer=opt,
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

    def convert_labels(y):
        samples = y.shape[0]
        classes = np.unique(y)
        one_hot_arr = np.zeros(shape = (samples,len(classes)))
        print(one_hot_arr.shape)
        for count,label_value in enumerate(y):
            one_hot_arr[count,int(label_value)] = 1


        return one_hot_arr



    def convert_label_max_only(y):
        idx = np.argmax(y,axis = 1)
        print(idx)
        out = np.zeros_like(y)
        for count, i in enumerate(idx):
            out[count, i] = 1
        return idx

    test_loss, test_acc = model.evaluate(x_test, y_test)

    y_pred = model.predict(x_test)
    from sklearn.metrics import ConfusionMatrixDisplay
    #y_1_hot = convert_labels(y_test)
    y_pred = convert_label_max_only(y_pred)
    print(y_pred, y_test)
    confusion_matrix_output = confusion_matrix(y_test,y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix_output)
    disp.plot()
    plt.savefig(str(test_acc)+".png",dpi = 1200)
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
    hyperparameter = {"batch_size": 32, 
    "conv_1_BatchNormalization": 0, "conv_1_filters": 52, "conv_1_kernel_size": 1, "conv_1_padding": "same", "conv_1_type": "Conv1D", 
    "dense_1_type": "Dense", "dense_1_units": 40,
    "dense_2_type": "Dense", "dense_2_units": 30,
    "dense_3_type": "Dense", "dense_3_units": 25,
     "epochs": 50, "num_conv_layers": 3, "num_dense_layers": 2, "optimiser": "Adam", "optimiser_lr": 0.013908363691981195, 
     "window_size": 100, 
     "conv_2_BatchNormalization": 1, "conv_2_filters": 52, "conv_2_kernel_size": 2, "conv_2_padding": "same", "conv_2_type": "Conv1D", 
     "conv_3_BatchNormalization": 1, "conv_3_filters": 52, "conv_3_kernel_size": 4, "conv_3_padding": "same", "conv_3_type": "Conv1D",
     "conv_4_BatchNormalization": 1, "conv_4_filters": 52, "conv_4_kernel_size": 8, "conv_4_padding": "same", "conv_4_type": "Conv1D",
     "conv_5_BatchNormalization": 1, "conv_5_filters": 52, "conv_5_kernel_size": 16, "conv_5_padding": "same", "conv_5_type": "Conv1D"} 

    main(hyperparameter,5 )

