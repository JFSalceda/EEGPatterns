import os
import numpy as np
import pandas
import random
from pyts.transformation import ShapeletTransform

'''
import tslearn.piecewise as piecewise
from keras.optimizers import Adagrad
import tslearn.shapelets as shapelets
import tslearn.utils as tsutil
import csv
'''

root = 'D:\\Programs\\THS-MS\\Complete Dataset\\Extracted files';
file = 'StandardizedDataset.csv';

aplusix_data = pandas.read_csv(os.path.join(root, file));

sample_split = 0.3;
sample_percent_size = 0.15;
sample_size = 100;

sample_index_start = random.randint(0, (len(aplusix_data.index)-sample_size));
sample_index_end = sample_index_start + sample_size;

data_start_index = 1;
data_end_index = 85;

label_start_index = 85;
label_end_index = 86;

shapelet_label = ['A', 'B', 'C', 'D', 'E']


sample_data_x = aplusix_data.iloc[sample_index_start:sample_index_end, data_start_index:data_end_index].to_numpy();
sample_data_y = aplusix_data.iloc[sample_index_start:sample_index_end, label_start_index:label_end_index].to_numpy(dtype='str').flatten();


#pyts shapelet transform

st = ShapeletTransform(n_shapelets=5, verbose=5, n_jobs=-1)
st.fit(sample_data_x, sample_data_y)

full_data_x = aplusix_data.iloc[:,data_start_index:data_end_index].to_numpy();
full_data_y = aplusix_data.iloc[:,label_start_index:label_end_index].to_numpy().flatten();

#sampled_data = aplusix_data.sample(frac=sample_percent_size);
sampled_data = aplusix_data.sample(n=sample_size);


high_frustration = sampled_data[aplusix_data.FrustratedLabel == 'H'];
#sampled_high_frustration = high_frustration.sample(frac=sample_percent_size);

high_frustration_train = high_frustration.sample(frac=sample_split);

high_frustration_train_x = high_frustration_train.iloc[:,data_start_index:data_end_index];
high_frustration_train_x = high_frustration_train_x.to_numpy();

high_frustration_train_y = high_frustration_train.iloc[:,label_start_index:label_end_index];
high_frustration_train_y = high_frustration_train_y.to_numpy(dtype='str');


high_frustration_test = high_frustration.drop(high_frustration_train.index);

high_frustration_test_x = high_frustration_test.iloc[:,data_start_index:data_end_index];
high_frustration_test_x = high_frustration_test_x.to_numpy();

high_frustration_test_y = high_frustration_test.iloc[:,label_start_index:label_end_index];
high_frustration_test_y = high_frustration_test_y.to_numpy(dtype='str');



low_frustration = sampled_data[aplusix_data.FrustratedLabel == 'L'];
#sampled_low_frustration = low_frustration.sample(frac=sample_percent_size);

low_frustration_train = low_frustration.sample(frac=sample_split);

low_frustration_train_x = low_frustration_train.iloc[:,data_start_index:data_end_index];
low_frustration_train_x = low_frustration_train_x.to_numpy();

low_frustration_train_y = low_frustration_train.iloc[:,label_start_index:label_end_index];
low_frustration_train_y = low_frustration_train_y.to_numpy(dtype='str');


low_frustration_test = low_frustration.drop(low_frustration_train.index)

low_frustration_test_x = low_frustration_test.iloc[:,data_start_index:data_end_index];
low_frustration_test_x = low_frustration_test_x.to_numpy();

low_frustration_test_y = low_frustration_test.iloc[:,label_start_index:label_end_index];
low_frustration_test_y = low_frustration_test_y.to_numpy(dtype='str');


train_samples_x = np.append(high_frustration_train_x, low_frustration_train_x, axis=0);
train_samples_y = np.append(high_frustration_train_y, low_frustration_train_y, axis=0);

#train_sample_data_x = train_samples_x.reshape(len(train_samples_x),1,125)
#train_sample_data_y = np.array([1 if x == "H" else 2 for x in train_samples_y]);
#train_sample_data_y = train_sample_data_y.reshape(len(train_sample_data_y),1)


test_samples_x = np.append(high_frustration_test_x, low_frustration_test_x, axis=0);
test_samples_y = np.append(high_frustration_test_y, low_frustration_test_y, axis=0);

#test_sample_data_x = test_samples_x.reshape(len(test_samples_x),1,125)
#test_sample_data_y = np.array([1 if x == "H" else 2 for x in test_samples_y]);
#test_sample_data_y = test_samples_y.reshape(len(test_samples_y),1)



st.fit(full_data_x, full_data_y)

shapelet_distances = st.fit_transform(full_data_x, full_data_y)

shapelet_distances = st.fit_transform(sample_data_x, sample_data_y)

single_column = sample_data_x[1,:].reshape(-1, 1)
single_window = st.transform(single_column)


##Training Shapelets
#
#clf = shapelets.ShapeletModel(n_shapelets_per_size={1: 10}, max_iter=1, verbose_level=1)
#clf.fit(train_sample_data_x, train_sample_data_y)
#train_shapelet_distances = clf.transform(test_sample_data_x)

shapelet_pattern = [];

for i in range(len(shapelet_distances)):
    shapelet_pattern.append(shapelet_label[np.argmin(shapelet_distances[i], axis=0)])

