import numpy as np
import math
def window_array(x_arr,y_arr , window_size : int, step_size :int =1):
    """
    splits array into set of sequences of window_size.
    starting at x = 0 then x += step_size 
    dim 0 in x_arr should be the time dim
    """

    if x_arr.shape[0] != y_arr.shape[0]: 
        raise ValueError("X (length "+str(x_arr.shape)+") and Y (length "+str(y_arr.shape)+") arrays different lengths in time domain")

    #Define array sizes
    num_windows = math.ceil(len(x_arr)/step_size) - int((window_size -1)/ step_size )
    x_output = np.empty(( num_windows , window_size , *x_arr.shape[1:] ) , float )
    y_output = np.empty((num_windows),float)
    #Compute X Array
    for c in range(num_windows):
        x_output[c] = x_arr[c*step_size:(c*step_size)+window_size]
        y_output[c] = y_arr[(c*step_size)+window_size-1 ]
    return x_output, y_output
    
if __name__ == "__main__":
    ##Tests 
    size = 8
    arr_1 = np.arange(100)
    arr_2 = np.arange(100)
    out_1,out_2= window_array(arr_1,arr_2,window_size = size, step_size = 6)
    for x,y in zip(out_1,out_2):
        print(x,y)
    print(out_1.shape)
    assert     arr_1[7:15,:,:].all() ==  out_1[1,:,:,:].all()
                    
        
