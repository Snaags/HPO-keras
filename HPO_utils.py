import random
import math
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import os


def Dispatcher(Configurations: list, num_workers: int, train_func,validation_flag = False) -> dict:
		"""
			returns a dictionary "ID":[hyperparameters: dict, Model_weights: dict, Training Loss: list[float], Training Time: int]	
		"""
		with Pool(processes=num_workers) as pool:
			results = pool.map(train_func.main,Configurations)
			pool.close()
			pool.join()


		return results

