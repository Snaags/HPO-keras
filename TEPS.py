# %% 
import numpy as np 
import pandas as pd 
import pyreadr
import os
#%%
path = os.getcwd()
#%%
data = pyreadr.read_r(path+"/datasets/TEPS/TEP_Faulty_Testing.RData")
                                                                                                                                                                                                                                                                                                               
# %%
df = pd.DataFrame(data)
# %%
np.asarray(data)
# %%
data.keys()
# %%
arr = data["faulty_testing"]
# %%
type(arr)
# %%
df = arr
# %%
df.head()
# %%
df.hist()
# %%
df["faultNumber"].hist()
# %%
df[df["faultNumber"] == 0]["xmeas_1"].hist()
# %%
df = pd.DataFrame(np.load("datasets/TEPS/Training_0.npy"))
# %%
df.hist()
# %%
def import_data(path):
    ext_path = os.getcwd()
    data = pyreadr.read_r(ext_path+path)
    k = list(data.keys())[0]
    arr = data[k]

    return arr

# %%
train = import_data("/datasets/TEPS/TEP_Faulty_Training.RData")
# %%
train.head()

# %%
train["faultNumber"].hist(bins = 40)
# %%
train.shape
# %%
train
# %%
train[train["faultNumber"] == 1].shape
# %%
arr= np.load("datasets/TEPS/Testing_5.npy")
# %%
arr.shape
# %%
arr
# %%
arr[:,:5]
# %%
arr[-1,:5]
# %%
