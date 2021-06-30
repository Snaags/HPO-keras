import tensorflow.keras as keras

class layer:
    def __init__(self):
        

class model:
    def __init__(self,input_shape : tuple,output_shape : tuple,hyperparameter : dict):
        self.input_layer = keras.layers.Input(input_shape)
        self.cells = [self.input_layer]
        


