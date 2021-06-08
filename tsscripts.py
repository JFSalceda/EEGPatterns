# -*- coding: utf-8 -*-
from tslearn.datasets import UCR_UEA_datasets
import tslearn.piecewise as piecewise
from keras.optimizers import Adagrad
import tslearn.shapelets as shapelets
import tslearn.utils as tsutil
import numpy as np
import pandas
import csv
from hmmlearn import hmm as hlhmm
from seqlearn import hmm as sqhmm

aplusix_data = np.genfromtxt('D:\Programs\THS-MS\Complete Dataset\Aplusix_CompleteData.csv', 
                          delimiter=',',
                          skip_header=1,
                          usecols=np.arange(3,128))

aplusix_user_info = pandas.read_csv('D:\Programs\THS-MS\Complete Dataset\Aplusix_CompleteData.csv',
                     usecols=(0,1,2))
aplusix_user_info = aplusix_user_info.to_numpy(dtype='str')


aplusix_emotion_label = pandas.read_csv('D:\Programs\THS-MS\Complete Dataset\Aplusix_CompleteData.csv',
                     usecols=(133,134,135,136))
aplusix_emotion_label = aplusix_emotion_label.to_numpy(dtype='str')



aplusix_emotion_values = pandas.read_csv('D:\Programs\THS-MS\Complete Dataset\Aplusix_CompleteData.csv',
                     usecols=(129,130,131,132))
aplusix_emotion_values = aplusix_emotion_values.to_numpy()

mini_aplusix_y = aplusix_emotion_values[:,0:1]

mini_aplusix_x = aplusix_data[34:85,:]

mini_aplusix_data = tsutil.to_time_series([mini_aplusix_x])
mini_splusix_dataset = tsutil.to_time_series_dataset(mini_aplusix_data)


clf = shapelets.ShapeletModel(n_shapelets_per_size={10: 5}, max_iter=1, verbose_level=0)
clf.fit(mini_aplusix_data, mini_aplusix_y)
shapelets_data = clf.transform(mini_aplusix_data)


sax = piecewise.SymbolicAggregateApproximation(n_segments=10, alphabet_size_avg=5)
sax_data = sax.transform(mini_aplusix_data)
shrunk_sax_data = sax_data[0]

print(sax_data[0])

remodel = hlhmm.GaussianHMM(n_components=16, covariance_type="diag", n_iter=1000)
remodel.fit(mini_aplusix_x)
Z2 = remodel.predict(mini_aplusix_x)
Z3 = remodel.predict_proba(mini_aplusix_x)
remodel.score(mini_aplusix_x)

seqmodel = hmm.MultinomialHMM()
seqmodel.fit(aplusix_data, mini_aplusix_y, 33977)
z4 = seqmodel.predict(aplusix_data)
seqmodel.score(aplusix_data,mini_aplusix_y)
