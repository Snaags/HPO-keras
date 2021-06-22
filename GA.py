import logging 
logging.basicConfig(level=logging.WARNING)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from hpbandster.optimizers import RandomSearch as RandomSearch
from worker_hp_ford import MyWorker
from algorithms import GA
from multiprocessing import Pool
from TEPS_worker import main as train_function
def dict_loop(dict_, list_):
    for ID, config in enumerate(list_):
        dict_[ID] = config
    return dict_

w = MyWorker(sleep_interval = 0, nameserver='127.0.0.1',run_id='example1')
configspace = w.get_configspace()
###Configuration Settings###
pop_size = 64
elite_size = 8
num_iter = 20


ga = GA(pop_size, elite_size)
population = configspace.sample_configuration(pop_size)
pop_dict = dict()
score_dict = dict()
pop_dict = dict_loop(pop_dict, population) 

for i in range(num_iter):
    pop = []
    for i in population:
        pop.append(i.get_dictionary())
    with Pool(processes = 1) as pool:
        results = pool.map(train_function, pop)
        pool.close()
        pool.join()
    scores = []
    for i in results:
        scores.append(i[0])
    score_dict = dict_loop(score_dict, scores) 
        
     
    population = ga.mutate(pop_dict, score_dict)
    pop_dict = dict_loop(pop_dict, population) 
            
    
                 
