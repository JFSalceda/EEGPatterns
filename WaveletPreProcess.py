import os
import pywt
import numpy as np
import pandas

root = 'D:/Programs/THS-MS/Complete Dataset/Complete Raw Dataset';

  
def getDatasetFilenames(path):
    
    filenames = []
    
    for directory in os.listdir(path):
        if directory.endswith(".csv"):
            filenames.append(path + "/" + directory)
            
    return filenames
    
def preProcessDatasets(filepaths):
    
    for path in filepaths:
        raw_data = pandas.read_csv(path);
        signals_only = raw_data.iloc[:,6:7];
        signals_only = signals_only.to_numpy().flatten();
        
        index = 0;
        
        while index < signals_only.size:            
            if(signals_only.size <= (index + 256)):
                end_index = signals_only.size-1;
            else:
                end_index = index + 256;
            
            data_segment = signals_only[index:end_index];            
            
            dec_sample = pywt.wavedec(data_segment, 'sym20', level=5);
            
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
            
            print(index,end_index,theta_peakMagnitude,theta_meanPower,alpha_peakMagnitude,alpha_meanPower,+beta_peakMagnitude,beta_meanPower);
            
            index = end_index+1;

def main():
    files = getDatasetFilenames(root)
    preProcessDatasets(files)

if __name__ == "__main__":
    main()