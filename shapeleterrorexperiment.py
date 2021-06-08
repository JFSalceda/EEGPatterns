# -*- coding: utf-8 -*-
"""
Created on Thu May 13 22:21:15 2021

@author: franc
"""

import os
import pywt
import numpy as np
import pandas
import csv
from sklearn.preprocessing import StandardScaler
from pyts.transformation import ShapeletTransform
from sklearn.tree import DecisionTreeClassifier

root = 'D:\\Programs\\THS-MS\\Complete Dataset\\Complete Raw Dataset\\'
relaxed_path = 'D:\\Programs\\THS-MS\\Complete Dataset\\Relaxed Dataset\\'
output_path = 'D:\\Programs\\THS-MS\\Complete Dataset\\Extracted files\\'


standardized_dataset = pandas.read_csv(output_path + 'StandardizedDataset.csv');

users = standardized_dataset['User'].unique();

users = [str(user) for user in users]

sample_data = []


for user_names in users:
   sample_data.append(standardized_dataset.loc[standardized_dataset['User'] == user_names].iloc[:,1].to_numpy())

        
max_length = len(max(sample_data, key=len))

reshaped_samples = np.zeros((len(users), max_length));
user_ids = [];
segmented_subsequences = np.empty((0,4), int);   

for i in range(0, len(sample_data)):
    reshaped_samples[i][0:len(sample_data[i])] = sample_data[i]
    
    index = 0;
    current_sequence = sample_data[i];
            
    while index < len(current_sequence):
        
        if(index + 4 > len(current_sequence)):
            end_index = len(current_sequence) - 1;
        else:
            end_index = index + 4;
        
        potential_subsequence = np.zeros(4);
        potential_subsequence[0:end_index-index] = current_sequence[index:end_index];
                    
        segmented_subsequences = np.append(segmented_subsequences, np.array([potential_subsequence]), axis=0);
        
        index = end_index + 1;
        
        user_ids.append(users[i]);

    
window_sizes = np.array(range(1,5))/max_length

st = ShapeletTransform(n_shapelets=3, verbose=100, n_jobs=-1, criterion='anova', window_sizes=window_sizes)
st.fit(reshaped_samples, users)
    
sax_string = st.transform(segmented_subsequences);

test = standardized_dataset.iloc[0:3,2:83].to_numpy()
test_users = standardized_dataset.iloc[0:3,0].to_numpy()

st.fit(test,users)


# Decision trees trial


X = standardized_dataset.iloc[:,1:85].to_numpy()
y = standardized_dataset.iloc[:,88:89].to_numpy()

tree = DecisionTreeClassifier().fit(X, y)
featutes4 = tree.feature_importances_


#    users = users.flatten()
    
#     reshaped_samples = np.append(reshaped_samples, reshaped_samples, axis=0)
#     users = np.append(users, users)
    
#     sax_string = st.transform(segmented_subsequences);

    
#     test = reshaped_samples.T
    
#     test = np.append(test, test, axis=0)

# test = standardized_dataset.iloc[0:5,1:83].to_numpy()
# test = reshaped_samples[:,0:303]
# test = np.append(test, np.array([segmented_subsequences[265,:]]), axis=0);


#  test=np.array(test)
#  users =
# users = standardized_dataset['User'][0:5].to_numpy()
