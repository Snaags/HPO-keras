#%%
import matplotlib.pyplot as plt
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
import os
import numpy as np

def random_model_over_time(runs, id2conf, show=False):
    model_based_runs = list(filter(lambda r: id2conf[r.config_id]['config_info']['model_based_pick'], runs))
    random_runs = list(filter(lambda r: not id2conf[r.config_id]['config_info']['model_based_pick'], runs))
    model_l = []
    random_l = []
    for r in model_based_runs:
        if r.loss is None or not np.isfinite(r.loss):
            continue
        model_l.append(r.loss)
    for r in random_runs:
        if r.loss is None or not np.isfinite(r.loss):
            continue
        random_l.append(r.loss)
 
    plt.scatter(range(len(model_l)),model_l ,alpha = 0.4,c = "r")
    plt.scatter(range(len(random_l)),random_l ,alpha = 0.4,c = "b")
    plt.show()

def performance_histogram_model_vs_random(runs, id2conf, show=False):
    model_based_runs = list(filter(lambda r: id2conf[r.config_id]['config_info']['model_based_pick'], runs))
    random_runs = list(filter(lambda r: not id2conf[r.config_id]['config_info']['model_based_pick'], runs))
    budgets = list(set([r.budget for r in runs]))
    budgets.sort()

    losses = {}
    for b in budgets:
        losses[b] = {'model_based': [], 'random': []}

    for r in model_based_runs:
        if r.loss is None or not np.isfinite(r.loss):
            continue
        losses[r.budget]['model_based'].append(r.loss)

    for r in random_runs:
        if r.loss is None or not np.isfinite(r.loss):
            continue
        losses[r.budget]['random'].append(r.loss)

    fig, axarr = plt.subplots(len(budgets), 2, sharey='row', sharex='all')
    plt.suptitle('Loss of model based configurations vs. random configuration')

    for i,b in enumerate(budgets):
        mbax, rax = axarr[i]
        print(mbax)
        mbax.hist(losses[b]['model_based'], bins = 50,label='Model Based',alpha = 0.5)
        mbax.set_ylabel('frequency')
        plt.xlim(0,0.5)

        
        
        
        mbax.hist(losses[b]['random'], bins = 70,label='Random',alpha = 0.5)

        mbax.legend()
        
        if i == len(budgets)-1:
            mbax.set_xlabel('loss')
    if show:        
        plt.show()
        
    return(fig, axarr)



os.chdir("/home/snaags/scripts/")
def extract(runs,data,name):
    data[name+" ID"] = [] 
    data[name+" loss"] = [] 
    data[name+" info"] = [] 
    data[name+" time_stamps"] = [] 
    for r in runs:
        if r.loss is None:
            continue
        data[name+" ID"].append(r.config_id)
        data[name+" loss"].append(r.loss)
        data[name+" info"].append(r.info)
        data[name+" time_stamps"].append(r.time_stamps)
    return data

def get_min(data, name):
    current_min = 100
    data[name+" min"] = []
    for i in data[name]:
        if i < current_min:
            current_min = i
        data[name+ " min"].append(current_min)

def get_linked(data, name, org, link):

    data[name+" linked"] = []
    for i in data[name]:
        data[name+" linked"].append(data[link][data[org].index(i)])


def get_min_loss(data,name):
    current_best_id = 0
    current_best= 1
    for i in data[name]:
        if i < current_best:
            current_best_id = data["limited ID"][data[name].index(i)]
            current_best = i
    print(current_best)
    return current_best_id


#%%
# load the example run from the log files
full = "/home/snaags/scripts/logs_ford_full_window"
limited = "/home/snaags/scripts/logs_TEPS_BO_FULL"

limited_run_data = hpres.logged_results_to_HBS_result(limited)

full_run_data = hpres.logged_results_to_HBS_result(full)
id2conf = full_run_data.get_id2config_mapping()
# %%
data = dict()
def normalise_mean(x1,x2):
    #normalise x1 and x2 so that all values of x1 are the same
    mean = np.mean(x1)
    outlist1 = []
    outlist2 = []
    for count,i in enumerate(x1):

        outlist1.append(x1[count]+ (mean - i))
        outlist2.append(x2[count]+ (mean - i))

    return outlist2, outlist1
runs = limited_run_data.get_all_runs()
extract(runs,data,"limited")
runs = full_run_data.get_all_runs()
extract(runs,data,"full")


get_min(data,"limited loss")
get_min(data,"limited info")
get_min(data,"full loss")
get_linked(data,"limited loss min", "limited loss","limited info")
from sklearn.preprocessing import StandardScaler

plt.hist(data["limited loss"],label = "Full Test Scores",bins = 100,alpha = 0.6)
#plt.hist(data["limited info"],label = "Reduced Test Scores",bins = 100,alpha = 0.6)
plt.legend()
plt.xlabel("Loss")
plt.ylabel("Occurrence")
plt.show()

arr, arr2 = normalise_mean(data["limited info"], data["limited loss"])
plt.hist(arr, bins = 50,label = "Adjusted Limited Validation Set Loss")
plt.hist(arr2, bins = 50,label = "Adjusted Limited Validation Set Loss")
#plt.vlines(mu,0,30,color= "r",linestyles='dashed',alpha = 0.7, label = "Mean Adjusted Loss")
plt.xlabel("Cross Entropy Loss")
plt.ylabel("Occurances")
plt.show()

mean = np.mean(data["full loss"])
#mean2 = np.mean(data["limited loss"])
#plt.scatter(range(len(data["limited loss"])),data["limited loss"], alpha = 0.4,c = "grey")
plt.scatter(range(len(data["full loss"])),data["full loss"], alpha = 0.4,c = "r")
#plt.scatter(range(len(data["limited info"])),data["limited info"], alpha = 0.4,c = "b")
#plt.plot(range(len(data["limited loss min"])),data["limited loss min"],label="Restricted Validation Set",c = "grey")
plt.plot(range(len(data["full loss min"])),data["full loss min"],label="Large Validation Set Scores",c= "r")
#plt.plot(range(len(data["limited info min"])),data["limited info min"],label="Restricted Validation Set Base True Scores")
#plt.plot(range(len(data["limited loss min linked"])),data["limited loss min linked"],label="Full Validation Set",c = "b")
plt.xlabel("Iteration")
plt.ylabel("Sparse Catagorical Loss")
plt.legend()

plt.xlim(0,300)
plt.grid()
plt.show()
# the number of concurent runs,


hpvis.concurrent_runs_over_time(runs)
performance_histogram_model_vs_random(runs, id2conf)
plt.show()

random_model_over_time(runs,id2conf)
# %%
