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
  
subsequence_length = 4;
shapelet_label = ['A', 'B', 'C', 'D', 'E'];

#dirs = os.listdir(root);
# 
#for directory in dirs:
#    if directory.endswith(".csv"):
#        print(directory)
#
#pretest = [('1','2'),('3','4'),('5','6')]
#
#test = np.array(pretest).T.tolist()
# 
#path = root
#
#filenames = []
#
#for directory in os.listdir(path):
#    if directory.endswith(".csv"):
#        filenames.append(os.path.join(path, directory))
#            
#for path in filenames:
#    raw_data = pandas.read_csv(path);
#    path_elements = path.split('\\');
#    user_name = path_elements[len(path_elements)-1].split('.')[0];
#
#file_data = raw_data;
#
#features = [];
#    
#index = 0;
#end_index = 0;
#num_rows = len(file_data.index);
#
#while index < num_rows:
#    
#    thetaPM = [];
#    thetaAP = [];
#    alphaPM = [];
#    alphaAP = [];
#    betaPM = [];
#    betaAP = [];
#    channel_names = [];
#    
#    features = [];
#
#    end_index = index + 255;
#    
#    if(num_rows <= end_index):
#        end_index = num_rows - 1;
#    
#    confidence_rating = file_data.iloc[index:end_index,32:33].mode().iat[0,0]   
#    excitement_rating = file_data.iloc[index:end_index,33:34].mode().iat[0,0] 
#    interest_rating = file_data.iloc[index:end_index,34:35].mode().iat[0,0]  
#    frustration_rating = file_data.iloc[index:end_index,35:37].mode().iat[0,0] 
#        
#    print(index, ' ', end_index);
#    
#    for i in range(5,19):
#        
#        data_segment = file_data.iloc[index:end_index,i:(i+1)];
#        channel_names.append(data_segment.columns[0]);
#                
#        data_segment = data_segment.to_numpy().flatten();   
#        
#        dec_sample = pywt.wavedec(data_segment, 'sym20', level=5);
#        
#        theta_magnitude = [abs(ele) for ele in dec_sample[1]]; 
#        theta_power = [ele*ele for ele in dec_sample[1]];
#        thetaPM.append(max(theta_magnitude)); 
#        thetaAP.append(np.mean(theta_power));
#        
#        alpha_magnitude = [abs(ele) for ele in dec_sample[2]]; 
#        alpha_power = [ele*ele for ele in dec_sample[2]]; 
#        alphaPM.append(max(alpha_magnitude));
#        alphaAP.append(np.mean(alpha_power));
#        
#        beta_magnitude = [abs(ele) for ele in dec_sample[3]]; 
#        beta_power = [ele*ele for ele in dec_sample[3]];
#        betaPM.append(max(beta_magnitude)); 
#        betaAP.append(np.mean(beta_power)); 
#        
#    for j in range(14):
#        features.append([channel_names[j] + "_thetaPM", thetaPM[j]]);
#        features.append([channel_names[j] + "_thetaAP", thetaAP[j]]);
#        features.append([channel_names[j] + "_alphaPM", alphaPM[j]]);
#        features.append([channel_names[j] + "_alphaAP", alphaAP[j]]);
#        features.append([channel_names[j] + "_betaPM", betaPM[j]]);
#        features.append([channel_names[j] + "_betaAP", betaAP[j]]);
#    
#    features.append(["BoredLabel", "H"]) if excitement_rating <= 50 else features.append(["BoredLabel", "L"])
#    features.append(["ConfusedLabel", "H"]) if confidence_rating <= 50 else features.append(["ConfusedLabel", "L"])
#    features.append(["InterestedLabel", "H"]) if interest_rating >= 50 else features.append(["InterestedLabel", "L"])
#    features.append(["FrustratedLabel", "H"]) if frustration_rating >= 50 else features.append(["FrustratedLabel", "L"])
#    
#    features = np.array(features).T;
#       
#    with open(output_path + user_name + '_features.csv', 'a', newline='') as f:
#        file_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL);        
#        if(f.tell() == 0):
#            file_writer.writerow(features[0]);
#            
#        file_writer.writerow(features[1]);
#            
#    index = end_index + 1;

def walkAndAnnotateDatasets(basedir):
    
    user_data = []
    user_annotations = []
    
    for path, subdirs, files in os.walk(basedir):
        for name in files:
            if "-annotation.csv" in name:
                user_annotations.append(os.path.join(path, name))
            else:
                user_data.append(os.path.join(path, name))
            print(name)
    
    for i in range(0, len(user_data)):
        data_file = user_data[i]
        annotation_file = user_annotations[i];
        
        user_label = data_file.replace(basedir, '').split('\\')[0]
        
        annotation_data = pandas.read_csv(annotation_file);
        user_eeg = pandas.read_csv(data_file);
        
        start_time = 0
        end_time = 0
        
        user_eeg['INTEREST'] = 0;
        user_eeg['CONFUSION'] = 0;
        user_eeg['FRUSTRATION'] = 0;
        user_eeg['BOREDOM'] = 0;
        
        for j in range(0, len(annotation_data)):
            if(j > 0):
                start_time = annotation_data['Start time'][j];
            
            start_index = user_eeg[user_eeg['Timestamp'].gt(start_time)].index[0]
            
            if(j < len(annotation_data)-1):
                end_time = annotation_data['End time'][j+1];
                end_index = user_eeg[user_eeg['Timestamp'].gt(end_time)].index[0]-1
            else:
                end_index = len(user_eeg) - 1;
               
            user_eeg.loc[start_index:end_index,'INTEREST'] = annotation_data['Interested'][j];
            user_eeg.loc[start_index:end_index,'CONFUSION'] = annotation_data['Confused'][j];
            user_eeg.loc[start_index:end_index,'FRUSTRATION'] = annotation_data['Frustrated'][j];
            user_eeg.loc[start_index:end_index,'BOREDOM'] = annotation_data['Bored'][j];
                   
        user_eeg.to_csv(os.path.join(basedir,user_label+'.csv'), index=False);
        
    
def getDatasetFilenames(path):
        
    filenames = []
    
    for directory in os.listdir(path):
        if directory.endswith(".csv"):
            filenames.append(os.path.join(path, directory))
            
    return filenames

def walkAndPreProcessDirectory(files):
    
    for path in files:
        raw_data = pandas.read_csv(path);
        path_elements = path.split('\\');
        user_name = path_elements[len(path_elements)-1].split('.')[0];
        preProcessDataset(raw_data, user_name)            

def preProcessDataset(file_data, user):
    
    features = [];
        
    index = 0;
    end_index = 0;
    num_rows = len(file_data.index);
    
    while index < num_rows:
        
        thetaPM = [];
        thetaAP = [];
        alphaPM = [];
        alphaAP = [];
        betaPM = [];
        betaAP = [];
        channel_names = [];
        features = [];
    
        end_index = index + 255;
        
        if(num_rows <= end_index):
            end_index = num_rows - 1;
        
        interest_rating = file_data.iloc[index:end_index,32:33].mode().iat[0,0]  
        confusion_rating = file_data.iloc[index:end_index,33:34].mode().iat[0,0] 
        frustration_rating = file_data.iloc[index:end_index,34:35].mode().iat[0,0]  
        boredom_rating = file_data.iloc[index:end_index,35:36].mode().iat[0,0] 
        
#        print(index, ' ', end_index);
        
        for i in range(5,19):
            
            data_segment = file_data.iloc[index:end_index,i:(i+1)];
            channel_names.append(data_segment.columns[0]);
            
            data_segment = data_segment.to_numpy().flatten();   
            
            dec_sample = pywt.wavedec(data_segment, 'sym20', level=5);
            
            theta_magnitude = [abs(ele) for ele in dec_sample[1]]; 
            theta_power = [ele*ele for ele in dec_sample[1]];
            thetaPM.append(max(theta_magnitude)); 
            thetaAP.append(np.mean(theta_power));
            
            alpha_magnitude = [abs(ele) for ele in dec_sample[2]]; 
            alpha_power = [ele*ele for ele in dec_sample[2]]; 
            alphaPM.append(max(alpha_magnitude));
            alphaAP.append(np.mean(alpha_power));
            
            beta_magnitude = [abs(ele) for ele in dec_sample[3]]; 
            beta_power = [ele*ele for ele in dec_sample[3]];
            betaPM.append(max(beta_magnitude)); 
            betaAP.append(np.mean(beta_power)); 
            
        for j in range(14):
            features.append([channel_names[j] + "_THETA_PM", thetaPM[j]]);
            features.append([channel_names[j] + "_THETA_AP", thetaAP[j]]);
            features.append([channel_names[j] + "_ALPHA_PM", alphaPM[j]]);
            features.append([channel_names[j] + "_ALPHA_AP", alphaAP[j]]);
            features.append([channel_names[j] + "_BETA_PM", betaPM[j]]);
            features.append([channel_names[j] + "_BETA_AP", betaAP[j]]);
        
        features.append(["Interested", "H"]) if interest_rating >= 50 else features.append(["Interested", "L"])
        features.append(["Confused", "H"]) if confusion_rating >= 50 else features.append(["Confused", "L"])
        features.append(["Frustrated", "H"]) if frustration_rating >= 50 else features.append(["Frustrated", "L"])
        features.append(["Bored", "H"]) if boredom_rating >= 50 else features.append(["Bored", "L"])
        
        features = np.array(features).T;
           
        with open(output_path + 'User Features\\' + user + '_features.csv', 'a', newline='') as f:
            file_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL);        
            if(f.tell() == 0):
                file_writer.writerow(features[0]);
                
            file_writer.writerow(features[1]);
                
        index = end_index + 1;

def preProcessBaselineDataset(directory_path):
       
    for directory in os.listdir(directory_path):
        if directory.endswith(".csv"):
            filename = directory_path + "/" + directory;
            
            file_data = pandas.read_csv(filename);
    
            path_elements = filename.split('/');
            user = path_elements[len(path_elements)-1].split('.')[0];
    
            features = [];
                        
            thetaPM = [];
            thetaAP = [];
            alphaPM = [];
            alphaAP = [];
            betaPM = [];
            betaAP = [];
            channel_names = [];
        
            for i in range(5,19):
                
                data_segment = file_data.iloc[:,i:(i+1)];
                channel_names.append(data_segment.columns[0]);
                
                data_segment = data_segment.to_numpy().flatten();   
                
                dec_sample = pywt.wavedec(data_segment, 'sym20', level=5);
                
                theta_magnitude = [abs(ele) for ele in dec_sample[1]]; 
                theta_power = [ele*ele for ele in dec_sample[1]];
                thetaPM.append(max(theta_magnitude)); 
                thetaAP.append(np.mean(theta_power));
                
                alpha_magnitude = [abs(ele) for ele in dec_sample[2]]; 
                alpha_power = [ele*ele for ele in dec_sample[2]]; 
                alphaPM.append(max(alpha_magnitude));
                alphaAP.append(np.mean(alpha_power));
                
                beta_magnitude = [abs(ele) for ele in dec_sample[3]]; 
                beta_power = [ele*ele for ele in dec_sample[3]];
                betaPM.append(max(beta_magnitude)); 
                betaAP.append(np.mean(beta_power)); 
            
            features.append(["User", user])
                
            for j in range(14):
                features.append([channel_names[j] + "_THETA_PM", thetaPM[j]]);
                features.append([channel_names[j] + "_THETA_AP", thetaAP[j]]);
                features.append([channel_names[j] + "_ALPHA_PM", alphaPM[j]]);
                features.append([channel_names[j] + "_ALPHA_AP", alphaAP[j]]);
                features.append([channel_names[j] + "_BETA_PM", betaPM[j]]);
                features.append([channel_names[j] + "_BETA_AP", betaAP[j]]);
            
            features = np.array(features).T;
               
            with open(output_path + 'relaxed_features.csv', 'a', newline='') as f:
                file_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL);        
                if(f.tell() == 0):
                    file_writer.writerow(features[0]);
                    
                file_writer.writerow(features[1]);

def subtractBaselineFeaturesandMerge():
    
    baseline_features = pandas.read_csv(output_path + 'relaxed_features.csv');
        
    for i in range(0, len(baseline_features.index)):
        
        feature_row = baseline_features.iloc[i:(i+1),:];
        curr_user = feature_row.iloc[0]['User'];
        
        feature_row = feature_row.loc[:, feature_row.columns != 'User'].to_numpy().flatten();
        
        user_data = pandas.read_csv(output_path + 'User Features\\' + curr_user + '_features.csv') 
    
        subtracted = user_data.iloc[:,0:84].to_numpy();
        subtracted = subtracted - feature_row;
                
        with open(output_path + 'CompleteDataset.csv', 'a', newline='') as f:
            file_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL);        
            if(f.tell() == 0):                
                feature_columns = user_data.columns.values.flatten();
                feature_columns = np.insert(feature_columns, 0, "User");
                file_writer.writerow(feature_columns);
            
            for j in range(0, len(subtracted)):
                labels = user_data.iloc[j,84:89].to_numpy().flatten();
                to_write = np.append(np.append(curr_user, subtracted[j]), labels);
                file_writer.writerow(to_write);                      
    
def standardizeDataset():
    full_dataset = pandas.read_csv(output_path + 'CompleteDataset.csv')
    
    features = full_dataset.iloc[:,1:85].to_numpy();
    
    scaler = StandardScaler();
    scaler.fit(features);
    
    print(scaler.mean_);
    print(scaler.var_);
    
    standardized = scaler.transform(features);

    standardized[standardized > 3] = 3;
    standardized[standardized < -3] = -3;

    with open(output_path + 'StandardizedDataset.csv', 'a', newline='') as f:
        file_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL);        
        if(f.tell() == 0):                
            feature_columns = full_dataset.columns.values.flatten();
            file_writer.writerow(feature_columns);

        for j in range(0, len(standardized)):
            labels = full_dataset.iloc[j,85:90].to_numpy().flatten();
            to_write = np.append(np.append(full_dataset.iloc[j]['User'], standardized[j]), labels);
            file_writer.writerow(to_write);  

#    column_scaler = StandardScaler();
#    column_scaler.fit(full_dataset[:,83:84]);
#    transformed_column = column_scaler.transform(full_dataset[:,83:84]);
##                write the dataset
##    
  
def selectFeaturesPerEmotion(dataset, emotion_index):
    
    
    
    X = standardized_dataset.iloc[:,1:85].to_numpy();
    y = standardized_dataset.iloc[:,87].to_numpy();
    
    tree = DecisionTreeClassifier().fit(X, y);
    features = tree.feature_importances_;
    indices = np.where(features > 0)[0];        
    
    return indices;
    
def transformToStringPatterns():
    standardized_dataset = pandas.read_csv(output_path + 'StandardizedDataset.csv');
    
    for emotion_index in range(85, 89):
        
        current_emotion = standardized_dataset.columns[emotion_index];
        feature_indices = selectFeaturesPerEmotion(standardized_dataset, emotion_index);
                
        users = standardized_dataset['User'].unique();
        
        users = [str(user) for user in users]
        
        emotion_labels = standardized_dataset.iloc[:,emotion_index].to_numpy();
        
        shapelet_strings = pandas.DataFrame(columns=['User', 'Feature','Shapelet_String']);
    
        for i in range(1, len(feature_indices)):
            sample_data = [];
            for user_name in users:
               sample_data.append(standardized_dataset.loc[standardized_dataset['User'] == user_name].iloc[:,i].to_numpy())
               
            feature_name = standardized_dataset.columns[i]
            
            feature_strings = getShapeletStringForFeature(sample_data, users, feature_name, emotion_labels);
            
            shapelet_strings = shapelet_strings.append(feature_strings, ignore_index=True);
            
        shapelet_strings.to_csv (output_path + current_emotion +'_ShapeletStrings.csv', index = False, header=True)
    
          
def getShapeletStringForFeature(sample_data, users, feature_name, emotion_labels):   
    
    reshaped_samples = np.zeros((len(users), len(max(sample_data, key=len))));
    user_ids = [];
    segmented_subsequences = np.empty((0,subsequence_length), int);   
    
    for i in range(0, len(sample_data)):
        reshaped_samples[i][0:len(sample_data[i])] = sample_data[i]
        
        index = 0;
        current_sequence = sample_data[i];
                
        while index < len(current_sequence):
            
            if(index + subsequence_length > len(current_sequence)):
                end_index = len(current_sequence);
            else:
                end_index = index + subsequence_length;
            
            potential_subsequence = np.zeros(subsequence_length);
            potential_subsequence[0:end_index-index] = current_sequence[index:end_index];
                        
            segmented_subsequences = np.append(segmented_subsequences, np.array([potential_subsequence]), axis=0);
            
            index = end_index;
            
            user_ids.append(users[i]);
    
    max_length = len(max(sample_data, key=len));
    
    window_sizes = np.array(range(1,subsequence_length+1))/max_length
    
    print("Extracting shapelets...");
    
    st = ShapeletTransform(n_shapelets=5, verbose=100, n_jobs=-1, criterion='anova', window_sizes=window_sizes)
    st.fit(reshaped_samples, users)
        
    print("Calculating Shapelet Distances...")
    
    shapelet_distances = st.transform(segmented_subsequences);

    shapelet_pattern = [];

    for i in range(len(shapelet_distances)):
        shapelet_pattern.append(shapelet_label[np.argmin(shapelet_distances[i], axis=0)])
    
    user = 'c10'
    
    shapelet_strings = pandas.DataFrame(columns=['User', 'Feature','Shapelet_String']);
    for user in users:
        user_indices = [i for i in range(len(user_ids)) if user_ids[i] == user]
        
        shapelet_strings = shapelet_strings.append({
            'User' : user,
            'Feature' : feature_name,
            'Shapelet_String' : ''.join([shapelet_pattern[i] for i in user_indices])}, ignore_index=True);
     
    return shapelet_strings;        

def main():
    walkAndAnnotateDatasets(root);
    files = getDatasetFilenames(root);
    walkAndPreProcessDirectory(files);
    preProcessBaselineDataset(relaxed_path);
    subtractBaselineFeaturesandMerge();
    standardizeDataset();
    transformToStringPatterns();

if __name__ == "__main__":
    main()