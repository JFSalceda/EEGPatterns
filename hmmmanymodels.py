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
sample_size = 1000
fs = 10e3
amp = 2 * np.sqrt(2)

aplusix_data = pandas.read_csv('D:\Programs\THS-MS\Complete Dataset\Aplusix_CompleteData_label.csv');

raw_data = pandas.read_csv('D:\Programs\THS-MS\Complete Dataset\Complete Raw Dataset\c9.csv');

sampled_data = aplusix_data.sample(frac=sample_percent_size);


signals_only = raw_data.iloc[:,6:7];
signals_only = signals_only.to_numpy().flatten();


#wwavelet transforms
frequencies = pywt.scale2frequency('cmor1.5-1.0', [1, 2, 3, 4])

(cA, cD) = pywt.dwt(signals_only, 'db1');
(coef, freqs) = pywt.cwt(signals_only,frequencies,'gaus1');


#short term fourier transform
f, t, Zxx = signal.stft(signals_only, nperseg=256, axis=0)


fs = 10e3
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
mod = 500*np.cos(2*np.pi*0.25*time)
carrier = amp * np.sin(2*np.pi*3e3*time + mod)
noise = np.random.normal(scale=np.sqrt(noise_power),
                        size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noise


#STFT plot
plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp)
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

'''
high_frustration = sampled_data[aplusix_data.FrustratedLabel == 'H'];
#sampled_high_frustration = high_frustration.sample(frac=sample_percent_size);

high_frustration_train = high_frustration.sample(frac=sample_split);

high_frustration_train_x = high_frustration_train.iloc[:,3:128];
high_frustration_train_x = high_frustration_train_x.to_numpy();

high_frustration_train_y = high_frustration_train.iloc[:,135:136];
high_frustration_train_y = high_frustration_train_y.to_numpy(dtype='str');


high_frustration_test = high_frustration.drop(high_frustration_train.index);

high_frustration_test_x = high_frustration_test.iloc[:,3:128];
high_frustration_test_x = high_frustration_test_x.to_numpy();

high_frustration_test_y = high_frustration_test.iloc[:,135:136];
high_frustration_test_y = high_frustration_test_y.to_numpy(dtype='str');



low_frustration = sampled_data[aplusix_data.FrustratedLabel == 'L'];
#sampled_low_frustration = low_frustration.sample(frac=sample_percent_size);

low_frustration_train = low_frustration.sample(frac=sample_split);

low_frustration_train_x = low_frustration_train.iloc[:,3:128];
low_frustration_train_x = low_frustration_train_x.to_numpy();

low_frustration_train_y = low_frustration_train.iloc[:,135:136];
low_frustration_train_y = low_frustration_train_y.to_numpy(dtype='str');


low_frustration_test = low_frustration.drop(low_frustration_train.index)

low_frustration_test_x = low_frustration_test.iloc[:,3:128];
low_frustration_test_x = low_frustration_test_x.to_numpy();

low_frustration_test_y = low_frustration_test.iloc[:,135:136];
low_frustration_test_y = low_frustration_test_y.to_numpy(dtype='str');



himodel = hlhmm.GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=num_iterations)
himodel.fit(high_frustration_train_x);

himodel_results = [];
himodel_state_probs = np.empty([0,num_states])
himodel_state_likelihoods = np.empty([0,num_states])

for i in range(len(high_frustration_test_x)): 
    observation = high_frustration_test_x[i:i+1,:]
    hiZ2 = himodel.predict(observation);
    hiZ3, hiZ4 = himodel.score_samples(observation);
    hiZ5 = himodel._compute_log_likelihood(observation)
    row = [hiZ2[0], hiZ3];
    himodel_results.append(row);
    himodel_state_probs = np.vstack((himodel_state_probs, hiZ4))
    himodel_state_likelihoods = np.vstack((himodel_state_likelihoods, hiZ5))
    
himodel_lodata_results = [];
himodel_lodata_state_probs = np.empty([0,num_states])
himodel_lodata_state_likelihoods = np.empty([0,num_states])

for i in range(len(low_frustration_test_x)): 
    observation = low_frustration_test_x[i:i+1,:]
    hiZ2 = himodel.predict(observation);
    hiZ3, hiZ4 = himodel.score_samples(observation);
    hiZ5 = himodel._compute_log_likelihood(observation)
    row = [hiZ2[0], hiZ3];
    himodel_lodata_results.append(row);
    himodel_lodata_state_probs = np.vstack((himodel_lodata_state_probs, hiZ4))
    himodel_lodata_state_likelihoods = np.vstack((himodel_lodata_state_likelihoods, hiZ5))



lomodel = hlhmm.GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=num_iterations)
lomodel.fit(low_frustration_train_x)

lomodel_results = [];
lomodel_state_probs = np.empty([0,num_states])
lomodel_state_likelihoods = np.empty([0,num_states])

for i in range(len(low_frustration_test_x)): 
    observation = low_frustration_test_x[i:i+1,:]
    loZ2 = lomodel.predict(observation)
    loZ3, loZ4 = lomodel.score_samples(observation)
    loZ5 = lomodel._compute_log_likelihood(observation)
    row = [loZ2[0], loZ3];
    lomodel_results.append(row);
    lomodel_state_probs = np.vstack((lomodel_state_probs, loZ4))
    lomodel_state_likelihoods = np.vstack((lomodel_state_likelihoods, loZ5))
    
lomodel_hidata_results = [];
lomodel_hidata_state_probs = np.empty([0,num_states])
lomodel_hidata_state_likelihoods = np.empty([0,num_states])

for i in range(len(high_frustration_test_x)): 
    observation = high_frustration_test_x[i:i+1,:]
    loZ2 = lomodel.predict(observation)
    loZ3, loZ4 = lomodel.score_samples(observation)
    loZ5 = lomodel._compute_log_likelihood(observation)
    row = [loZ2[0], loZ3];
    lomodel_hidata_results.append(row);
    lomodel_hidata_state_probs = np.vstack((lomodel_hidata_state_probs, loZ4))
    lomodel_hidata_state_likelihoods = np.vstack((lomodel_hidata_state_likelihoods, loZ5))


hidata_classification = []

for i in range(len(high_frustration_test_x)):
    if(himodel_results[i][1] >= lomodel_hidata_results[i][1]):
        hidata_classification.append('H');
    else:
        hidata_classification.append('L');
      
himodel_accuracy = hidata_classification.count('H')/len(high_frustration_test_x);

hiZ2 = himodel.predict(high_frustration_test_x);
hiZ3, hiZ4 = himodel.score_samples(high_frustration_test_x);
hiZ5 = himodel._compute_log_likelihood(high_frustration_test_x)


lodata_classification = []

for i in range(len(low_frustration_test_x)):
    if(lomodel_results[i][1] >= himodel_lodata_results[i][1]):
        lodata_classification.append('L');
    else:
        lodata_classification.append('H');

lomodel_accuracy = lodata_classification.count('L')/len(low_frustration_test_x);

loZ2 = lomodel.predict(low_frustration_test_x)
loZ3, loZ4 = lomodel.score_samples(low_frustration_test_x)
loZ5 = lomodel._compute_log_likelihood(low_frustration_test_x)

true_high = hidata_classification.count('H');
false_high = lodata_classification.count('H');

true_low = lodata_classification.count('L');
false_low = hidata_classification.count('L');

CM_recall = true_high/(true_high + false_low);
CM_specificity = true_low/(true_low + false_high);
CM_precision = true_high/(true_high + false_high);
CM_fmeasure = 2*((CM_precision * CM_recall)/(CM_precision + CM_recall))
CM_accuracy = (true_high + true_low) / (true_high + false_high + true_low + false_low)



import shelve


filename='D:\Programs\THS-MS\python files\shelve.out';
my_shelf = shelve.open(filename,'n') # 'n' for new

for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        #
        # __builtins__, my_shelf, and imported modules can not be shelved.
        #
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()

my_shelf = shelve.open('D:\Programs\THS-MS\python files\shelve.out');
for key in my_shelf:
    globals()[key]=my_shelf[key]
my_shelf.close()
'''
