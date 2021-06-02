#%%
import matplotlib.pyplot as plt
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
import os
import numpy as np

os.chdir("/home/snaags/scripts/HPO-keras")
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
full = "/home/snaags/scripts/logs_BOHB_full"
limited = "/home/snaags/scripts/HPO-keras/TEPS_full_log"

limited_run_data = hpres.logged_results_to_HBS_result(limited)

full_run_data = hpres.logged_results_to_HBS_result(full)
# %%
data = dict()


runs = limited_run_data.get_all_runs()
extract(runs,data,"limited")
runs = full_run_data.get_all_runs()
extract(runs,data,"full")
get_min(data,"limited loss")
get_min(data,"limited info")
get_min(data,"full loss")
#get_linked(data,"limited loss min", "limited loss","limited info")


plt.scatter(range(len(data["limited loss"])),data["limited loss"], alpha = 0.4,c = "grey")
#plt.scatter(range(len(data["full loss"])),data["full loss"], alpha = 0.4,c = "r")
plt.scatter(range(len(data["limited info"])),data["limited info"], alpha = 0.4,c = "b")
plt.plot(range(len(data["limited loss min"])),data["limited loss min"],label="Restricted Validation Set",c = "grey")
#plt.plot(range(len(data["full loss min"])),data["full loss min"],label="Large Validation Set Scores",c= "r")
#plt.plot(range(len(data["limited info min"])),data["limited info min"],label="Restricted Validation Set Base True Scores")

plt.xlabel("Iteration")
plt.ylabel("Sparse Catagorical Loss")
plt.legend()
plt.ylim(0,0.6)
plt.xlim(0,300)
plt.grid()
plt.show()
exit()
# the number of concurent runs,
hpvis.concurrent_runs_over_time(all_runs)

# and the number of finished runs.
hpvis.finished_runs_over_time(all_runs)

# This one visualizes the spearman rank correlation coefficients of the losses
# between different budgets.
hpvis.correlation_across_budgets(result)

hpvis.performance_histogram_model_vs_random(all_runs, id2conf)

plt.show()
# %%
