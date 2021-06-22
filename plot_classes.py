from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from ford_worker import load_dataset
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt 
from sklearn.utils import shuffle
from matplotlib import cm

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
    #axis.plot_trisurf    df_melted = df1.melt(var_name='column')
    while len(df2) < len(df_melted['column']):
        df2 = df2.append(df2)
        print("doubled!")
    df = df_melted.join(df2, lsuffix='_caller', rsuffix='_other')
    # Create the data
    df = df.rename(columns = {0:"classes"})
    (grid_s[:,0],grid_s[:,1],grid_s[:,2])
    axis.set_xlabel("PC1", fontsize=10)
    axis.set_ylabel("PC2", fontsize=10)
    axis.set_zlabel("PC3", fontsize=10)
    plt.colorbar(cb)
    
    plt.show()

def class_hist(x,y):
    cmap = cm.get_cmap("tab20c")
    outlist = [[]]*len(np.unique(y))
    clist = ["y","b","g"]
    for x_i, y_i in zip(x,y):
        outlist[y_i].append(x_i)
    list_of_arr = []
    for i in outlist:
        list_of_arr.append( np.asarray(i))
    print(len(list_of_arr))
    for count, l in enumerate(list_of_arr):
        c = clist[count]
        
        for i in range(l.shape[1]): 
            plt.hist(np.reshape(x[:,i],(-1)),bins = 100 ,alpha = 0.1,color = c) 
        plt.show()    

def plot_ridge_by_class(df):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    # Create the data
    print(df.head())
    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(data = df,row = "0_other", hue="0_other", aspect=15, height=.5, palette=pal)
    
    # Draw the densities in a few steps
    g.map(sns.kdeplot,"0_caller",
          bw_adjust=.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "0_caller", clip_on=False, color="w", lw=2, bw_adjust=.5)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)
    
    
    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
    
    
    g.map(label, "0_caller")
    
    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.25)
    
    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    plt.show()

def plot_ridge_by_feature(df1,df2):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    df_melted = df1.melt(var_name='column')
    while len(df2) < len(df_melted['column']):
        df2 = df2.append(df2,ignore_index = True)
    df = df_melted.join(df2, lsuffix='_caller', rsuffix='_other')
    # Create the data
    df = df.rename(columns = {0:"classes"})
    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(data = df,row = "column",hue = "classes", aspect=15, height=.5, palette = pal)
    
    # Draw the densities in a few steps
    g.map(sns.kdeplot,"value",
          bw_adjust=.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "value", clip_on=False, color="w", lw=2, bw_adjust=.5)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)
    
    
    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, str(x.iloc[0]), fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
    
    
    g.map(label, "column")
    g.set(xlim =(-3,3))     
    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.45)
    
    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    plt.show()

def plot_kde():
    x_train,y_train, x_test, y_test = load_dataset()
    x_train,y_train = shuffle(x_train,y_train, n_samples = 100000)
    x_train = np.squeeze(x_train)
    y_train = np.squeeze(y_train)
    df1 = pd.DataFrame(x_train[:,:1])
    df2 = pd.DataFrame(y_train)
    df = df1.join(df2, lsuffix='_caller', rsuffix='_other')
    #df = pd.concat([df1,df2])
    print(df.head())
    sns.kdeplot(data = df, x= "0_caller",palette = sns.color_palette("husl", 21), hue = "0_other")
    plt.show()

x_train,y_train, x_test, y_test = load_dataset()
#x_train,y_train = shuffle(x_train,y_train, n_samples = 100000)
x_train = np.squeeze(x_train)
y_train = np.squeeze(y_train)
df1 = pd.DataFrame(x_train[:,:50])
df2 = pd.DataFrame(y_train)
df = df1.join(df2, lsuffix='_caller', rsuffix='_other')
plot_ridge_by_feature(df1,df2)
