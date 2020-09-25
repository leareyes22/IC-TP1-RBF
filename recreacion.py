import pandas as pd
import numpy as np

# Carga de datos
train_input = np.asarray(pd.read_csv('train_data_input.csv', sep=',', header=None))
train_output = np.asarray(pd.read_csv('train_data_output.csv', sep=',', header=None))

test_input = np.asarray(pd.read_csv('test_data_input.csv', sep=',', header=None))

#TODO la listita de indices