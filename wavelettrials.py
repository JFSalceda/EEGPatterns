import pywt
import numpy as np
import pandas
import csv
from scipy import signal
import matplotlib.pyplot as plt

num_states = 4;
num_iterations =2000;
sample_split = 0.7;
sample_percent_size = 0.9;
sample_size = 128
fs = 10e3
amp = 2 * np.sqrt(2)

aplusix_data = pandas.read_csv('D:\Programs\THS-MS\Complete Dataset\Aplusix_CompleteData_label.csv');

raw_data = pandas.read_csv('D:\Programs\THS-MS\Complete Dataset\Complete Raw Dataset\c9.csv');

sampled_data = aplusix_data.sample(frac=sample_percent_size);


signals_only = raw_data.iloc[:,6:7];
channel_name = signals_only.columns[0];
signals_only = signals_only.to_numpy().flatten();

sample_signals = signals_only[0:256]

wavlist = pywt.wavelist(kind='continuous')

#wwavelet transforms
scales = np.arange(1,128)

frequencies = pywt.scale2frequency('cmor1.5-1.0', scales)


(cA, cD) = pywt.dwt(signals_only, 'sym20');
dec = pywt.wavedec(signals_only, 'sym20', level=6);

(cA_sample, cD_sample) = pywt.dwt(sample_signals, 'sym20');
dec_sample = pywt.wavedec(sample_signals, 'sym20', level=5);
#(coef, freqs) = pywt.cwt(signals_only,scales,'cmor1.5-1.0');


delta_magnitude = [abs(ele) for ele in dec_sample[0]]; 
delta_power = [ele*ele for ele in dec_sample[0]];
delta_peakMagnitude = max(delta_magnitude); 
delta_meanPower = np.mean(delta_power);

theta_magnitude = [abs(ele) for ele in dec_sample[1]] 
theta_power = [ele*ele for ele in dec_sample[1]] 
theta_peakMagnitude = max(theta_magnitude); 
theta_meanPower = np.mean(theta_power);

alpha_magnitude = [abs(ele) for ele in dec_sample[2]] 
alpha_power = [ele*ele for ele in dec_sample[2]] 
alpha_peakMagnitude = max(alpha_magnitude); 
alpha_meanPower = np.mean(alpha_power);

beta_magnitude = [abs(ele) for ele in dec_sample[3]] 
beta_power = [ele*ele for ele in dec_sample[3]]
beta_peakMagnitude = max(beta_magnitude); 
beta_meanPower = np.mean(beta_power); 

gamma_magnitude = [abs(ele) for ele in dec_sample[4]] 
gamma_power = [ele*ele for ele in dec_sample[4]]
gamma_peakMagnitude = max(gamma_magnitude); 
gamma_meanPower = np.mean(gamma_power); 


plt.plot(np.arange(4), dec_sample[0])
plt.title('WAVEDEC cA')
plt.show()

plt.plot(np.arange(65), dec_sample[6])
plt.title('WAVEDEC cD1')
plt.show()

plt.plot(np.arange(34), dec_sample[5])
plt.title('WAVEDEC cD2')
plt.show()

plt.plot(np.arange(18), dec_sample[4])
plt.title('WAVEDEC cD3')
plt.show()

plt.plot(np.arange(10), dec_sample[3])
plt.title('WAVEDEC cD4')
plt.show()

plt.plot(np.arange(6), dec_sample[2])
plt.title('WAVEDEC cD5')
plt.show()

plt.plot(np.arange(4), dec_sample[1])
plt.title('WAVEDEC cD6')
plt.show()

plt.plot(np.arange(3), dec_sample[8])
plt.title('WAVEDEC cD7')
plt.show()

plt.plot(np.arange(128), sample_signals)
plt.title('SIGNALS')
plt.show()


plt.plot(np.arange(31269), cA)
plt.title('DWT cA')
plt.show()

plt.plot(np.arange(31269), cD)
plt.title('DWT cD')
plt.show()

plt.plot(np.arange(62532),signals_only)
plt.title('Original')
plt.show()