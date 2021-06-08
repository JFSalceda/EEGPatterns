import numpy as np
import pandas
from scipy import signal


num_states = 4;
num_iterations =2000;
sample_split = 0.7;
sample_percent_size = 0.9;
sample_size = 1000
fs = 10e3
amp = 2 * np.sqrt(2)

raw_data = pandas.read_csv('D:\Programs\THS-MS\Complete Dataset\Complete Raw Dataset\c9.csv');

#sampled_data = raw_data.sample(frac=sample_percent_size);


signals_only = raw_data.iloc[:,5:6];
signals_only = signals_only.to_numpy().flatten();


#short term fourier transform
f, t, Zxx = signal.stft(signals_only, nperseg=256, noverlap=128, axis=0)