!pip install tabpfn
!pip install naiveautoml
import sys
import numpy as np
from pathlib import Path
import pandas as pd
import torch
import openml
import os
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
from tabpfn.scripts.decision_boundary import DecisionBoundaryDisplay

from naiveautoml import NaiveAutoML

from py_experimenter.experimenter import PyExperimenter
from py_experimenter.result_processor import ResultProcessor

import openml
import json

def get_dataset(openmlid):
    ds = openml.datasets.get_dataset(openmlid)
    print("dataset info loaded")
    df = ds.get_data()[0]
    num_rows = len(df)
    
    print("Data in memory, now creating X and y")
        
    # prepare label column as numpy array
    X = np.array(df.drop(columns=[ds.default_target_attribute]).values)
    y = np.array(df[ds.default_target_attribute].values)
    if y.dtype != int:
        y_int = np.zeros(len(y)).astype(int)
        vals = np.unique(y)
        for i, val in enumerate(vals):
            mask = y == val
            y_int[mask] = i
        y = y_int
    
    print(f"Data read. Shape is {X.shape}.")
    return X, y



def run_experiment(keyfields: dict, result_processor: ResultProcessor, custom_config):
    
    # Extracting given parameters
    openmlid = keyfields['openmlid']
    seed = keyfields['seed']
    learner = keyfields["algorithm"]
    
    X, y = get_dataset(openmlid)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 1000 if X.shape[0] >= 1200 else 0.8, random_state = seed)
    
    
    classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=4) if learner == "tabpfn" else NaiveAutoML(max_hpo_iterations = 10, scoring="accuracy")
    
    print(f"training {learner}.")
    start = time.time()
    classifier.fit(X_train, y_train)
    end = time.time()
    print(f"Done. Collecting predictions.")
    
    y_hat = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_hat)
    
    
    
    # Write intermediate results to database    
    resultfields = {
        "test_accuracy": np.round(acc, 4),
        "fitting_time": int(1000 * (end - start)),
    }
    if learner == "naiveautoml":
        times = [np.round(e["time"], 4) for e in classifier.history]
        scores = [np.round(e["score_internal"], 4) for e in classifier.history]
        resultfields["details"] = json.dumps([times, scores])
    
    result_processor.process_results(resultfields)
    

if __name__ == '__main__':
    job_name = sys.argv[1]
    experimenter = PyExperimenter(config_file="config/experiments.cfg", name = job_name)
    experimenter.execute(run_experiment, max_experiments=-1, random_order=True)