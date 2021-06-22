from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler 
from ford_worker import load_dataset
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt 
from sklearn.utils import shuffle


def generate_pca_plot_2d(x,y):
    pca = PCA(n_components = 2 )
    pca.fit(x)
    x = pca.transform(x)
    fig = plt.figure(figsize=(10,10))
    
    # choose projection 3d for creating a 3d graph
    
    # x[:,0]is pc1,x[:,1] is pc2 while x[:,2] is pc3
    cb = plt.scatter(x[:,0],x[:,1], c=y,cmap="tab20c")
    #axis.plot_trisurf(grid_s[:,0],grid_s[:,1],grid_s[:,2])
    plt.colorbar(cb)
    
    plt.show()

def generate_pca_plot_3d(x,y):
    pca = PCA(n_components = 3 )
    pca.fit(x)
    x = pca.transform(x)
    fig = plt.figure(figsize=(10,10))
    
    # choose projection 3d for creating a 3d graph
    axis = fig.add_subplot(111, projection='3d')
    
    # x[:,0]is pc1,x[:,1] is pc2 while x[:,2] is pc3
    cb = axis.scatter(x[:,0],x[:,1],x[:,2], c=y,cmap="tab20c")
    #axis.plot_trisurf(grid_s[:,0],grid_s[:,1],grid_s[:,2])
    axis.set_xlabel("PC1", fontsize=10)
    axis.set_ylabel("PC2", fontsize=10)
    axis.set_zlabel("PC3", fontsize=10)
    plt.colorbar(cb)
    
    plt.show()
    

x_train,y_train, x_test, y_test = load_dataset()
print(np.unique(y_test ))
generate_pca_plot_2d(np.squeeze(x_test),np.squeeze(y_test))
generate_pca_plot(x_train,y_train)
