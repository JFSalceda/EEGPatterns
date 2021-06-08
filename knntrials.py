import tslearn.piecewise as piecewise
import numpy as np
import pandas
import math
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def get_dominant_emotion(emotion_values, emotion_labels, emotion_columns):
    dynamic_emotions = []
        
    for i in range(emotion_values.shape[0]):
        values_row = emotion_values[i,:].flatten()
        labels_row = emotion_labels[i,:].flatten()
        
        if('H' in labels_row):
            m = np.amax(values_row)
            m_index = np.argwhere(values_row == m).flatten()
            
            if(m_index.size > 1):
                dynamic_emotions = np.append(dynamic_emotions, 'M')
            else:
                dynamic_emotions = np.append(dynamic_emotions, emotion_columns[m_index[0]][0][0])
        else:
            dynamic_emotions = np.append(dynamic_emotions, 'L')
        
    return dynamic_emotions   

def get_sax_emotions(dominant_emotions, num_sax_segments):
    sax_emotions = []
    segment_size = dominant_emotions.shape[0] // num_sax_segments
    
    for i in range(0, num_sax_segments):
        sax_emotions = np.append(sax_emotions, dominant_emotions[i*segment_size])
    
    return sax_emotions

aplusix_data = pandas.read_csv('D:\Programs\THS-MS\Complete Dataset\Aplusix_CompleteData_label.csv');

sample_split = 0.7;
sample_percent_size = 0.35;
rows = len(aplusix_data)
sample_size = math.floor(0.20*rows);

num_sax_segments = 2000
num_sax_alphabet = 4

sample_index = random.randint(sample_size, rows-sample_size)


sampled_data = aplusix_data.iloc[sample_index:sample_index+sample_size,:];
#sampled_data = aplusix_data.iloc[0:471,:]

sampled_emotion_values = sampled_data.iloc[:,129:133]
sampled_emotion_labels = sampled_data.iloc[:,133:138]

sample_dominant_emotions = get_dominant_emotion(sampled_emotion_values.to_numpy(), sampled_emotion_labels.to_numpy(), sampled_emotion_values.columns)

sampled_data_x = sampled_data.iloc[:,3:128]
sampled_data_x = sampled_data_x.to_numpy();
sampled_data_x = sampled_data_x.reshape(1,len(sampled_data),125)

sampled_data_y = sample_dominant_emotions

sampled_data_y = sampled_data_y.reshape(1,len(sampled_data_y))


split_size = math.floor(0.7*num_sax_segments)

split_indeces = random.sample(range(num_sax_segments), split_size)

'''
#sampled_data_train = aplusix_data.iloc[split_index:split_index+split_size,:];
sampled_data_train = sax_data[0].sample(frac=sample_split);
'''

sampled_data_train_x = np.take(sampled_data_x[0], split_indeces, axis = 0)
sampled_data_train_y = np.take(sampled_data_y, split_indeces)

sampled_data_test_x = np.delete(sampled_data_x[0], split_indeces, axis = 0)
sampled_data_test_y = np.delete(sampled_data_y, split_indeces)


classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(sampled_data_train_x, sampled_data_train_y)

y_pred = classifier.predict(sampled_data_test_x)
    
print(confusion_matrix(sampled_data_test_y, y_pred))
print(classification_report(sampled_data_test_y, y_pred))