import numpy as np
import pickle

class logger:
    
    def generate_attr_dict(self):
        #remove magic methods from 
        for attr in self.raw_algorithm_attr:
            if attr[:2] != "__":
                self.attr[attr] = eval("self.algorithm_object."+attr) 
            
    def __init__(self,algorithm , filename : str ):
        self.history = {
        "population": []
        "scores"    : []
        "state"     : []
        }
        self.algorithm_object = algorithm
        self.algorithm_state = dict()
        self.raw_algorithm_attr = dir(algorithm)
        self.attr = dict()
        self.current_population = []
        self.current_scores = []
        self.filename = filename 

    def update_current_state(self,algorithm, population, scores):
        self.algorithm_object = algorithm 
        self.current_population = population
        self.current_scores = scores
        for attr in self.attr:
            self.attr[attr] = eval("self.algorithm_object."+attr) 

    def log_history(self):
        self.history["population"].append(self.current_population0
        self.history["scores"].append(self.current_scores)
        self.history["state"].append(self.algorithm_state)
 
    def save_state_to_file(self):
        
    def load_state_from_file():

    def generate_plots():
        """
        data analysis should be incorped into this class
    
        """
        pass
