import pandas as pd
import numpy as np
import tslearn.piecewise as piecewise
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

def student_fold_knn(input_data, dominant_emotions):
    classificaion_results = []
    accuracies = 0.0
    
    for i in range(1, num_students):
        test_indeces = np.where(input_data[:,0] == i)
        
        test_student_x = input_data[test_indeces]
        test_student_x = np.delete(test_student_x, 0, axis = 1)
        
        test_student_y = dominant_emotions[test_indeces]
        test_student_y = np.delete(test_student_y, 0, axis = 1)
        
        train_data_x = np.delete(input_data, test_indeces, axis = 0)
        train_data_x = np.delete(train_data_x, 0, axis = 1)
        
        train_data_y = np.delete(dominant_emotions, test_indeces, axis = 0)
        train_data_y = np.delete(train_data_y, 0, axis = 1) 
                
        print('Train for student', i)
        
        classifier = KNeighborsClassifier(n_neighbors=10)
        classifier.fit(train_data_x, train_data_y)
        
        y_pred = classifier.predict(test_student_x)
            
        print(confusion_matrix(test_student_y, y_pred))
        
        report = classification_report(test_student_y, y_pred, output_dict = True)
        
        classificaion_results.append(report)
        accuracies += report["accuracy"]
    
    return classificaion_results, (accuracies/num_students)

def sax_knn(aplusix_data, dominant_emotions, num_students, num_sax_segments, num_sax_alphabet):
    sax_dataset = np.empty((0,126))
    sax_labels = np.empty((0,2))
    
    sax_results = []
    accuracies = 0.0
    
    for i in range(1, num_students):
        curr_student_data = aplusix_data.loc[aplusix_data['User'] == i]
        curr_student_labels = dominant_emotions[curr_student_data.index]
        
        curr_student_data = curr_student_data.iloc[:,3:128]
        curr_student_data = curr_student_data.to_numpy()
        curr_student_data = curr_student_data.reshape(1,len(curr_student_data),125)    
    
        curr_student_labels = get_sax_emotions(curr_student_labels, num_sax_segments)
    
        sax = piecewise.SymbolicAggregateApproximation(n_segments=num_sax_segments, alphabet_size_avg=num_sax_alphabet)
        sax_data = sax.transform(curr_student_data)
        
        student_id = np.full((num_sax_segments,1), i)
        
        sax_data = sax_data[0]
        sax_data = np.concatenate((student_id, sax_data), axis=1)
        
        curr_student_labels = np.concatenate((student_id, curr_student_labels[:, np.newaxis]), axis=1)
        
        sax_dataset = np.append(sax_dataset, sax_data, axis=0)
        sax_labels = np.append(sax_labels, curr_student_labels, axis=0)
                
    return student_fold_knn(sax_dataset, sax_labels)
        

num_students = 50
num_sax_segments = 200
num_sax_alphabet = 6

aplusix_data = pd.read_csv('D:\Programs\THS-MS\Complete Dataset\Aplusix_CompleteData_label.csv');

emotion_values = aplusix_data.iloc[:,129:133]
emotion_labels = aplusix_data.iloc[:,133:138]

dominant_emotions = get_dominant_emotion(emotion_values.to_numpy(), emotion_labels.to_numpy(), emotion_values.columns)

sax_results, overall_sax_accuracy = sax_knn(aplusix_data, dominant_emotions, num_students, num_sax_segments, num_sax_alphabet);

student_data = aplusix_data.iloc[:,3:128]
student_data = aplusix_data.to_numpy()
knn_results, overall_accuracy = student_fold_knn(student_data, dominant_emotions);

    
    


