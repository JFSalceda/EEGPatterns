from hmmlearn import hmm as hlhmm
from seqlearn import hmm as sqhmm
import numpy as np
import pandas
import csv

def main():
    aplusix_data = np.genfromtxt('D:\Programs\THS-MS\Complete Dataset\Aplusix_CompleteData_label.csv', 
                              delimiter=',',
                              skip_header=1,
                              usecols=np.arange(3,128))
    
    aplusix_user_info = pandas.read_csv('D:\Programs\THS-MS\Complete Dataset\Aplusix_CompleteData_label.csv',
                         usecols=(0,1,2))
    aplusix_user_info = aplusix_user_info.to_numpy(dtype='str')
    
    
    aplusix_emotion_label = pandas.read_csv('D:\Programs\THS-MS\Complete Dataset\Aplusix_CompleteData_label.csv',
                         usecols=(133,134,135,136))
    aplusix_emotion_label = aplusix_emotion_label.to_numpy(dtype='str')
    
    
    
    aplusix_emotion_values = pandas.read_csv('D:\Programs\THS-MS\Complete Dataset\Aplusix_CompleteData_label.csv',
                         usecols=(129,130,131,132))
    aplusix_emotion_values = aplusix_emotion_values.to_numpy()
    
    mini_aplusix_y = aplusix_emotion_label[0:100,0:1]
    
    mini_aplusix_x = aplusix_data[0:100,:]
    
    mini_aplusix_data = tsutil.to_time_series([mini_aplusix_x])
    mini_splusix_dataset = tsutil.to_time_series_dataset(mini_aplusix_data)
    
    lomodel = hlhmm.GaussianHMM(n_components=8, covariance_type="diag", n_iter=2000)
    lomodel.fit(mini_aplusix_x[0:37,:])
    loZ2 = lomodel.predict(mini_aplusix_x)
    loZ3, loZ4 = himodel.score_samples(mini_aplusix_x)
    lomodel.score(mini_aplusix_x)
    
    himodel = hlhmm.GaussianHMM(n_components=8, covariance_type="diag", n_iter=2000)
    himodel.fit(mini_aplusix_x[38:100,:])
    hiZ2 = himodel.predict(mini_aplusix_x)
    hiZ3, hiZ4 = himodel.score_samples(mini_aplusix_x)

if __name__ == __main__:
    main()