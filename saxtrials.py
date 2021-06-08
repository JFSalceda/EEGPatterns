import tslearn.piecewise as piecewise
from keras.optimizers import Adagrad
import tslearn.shapelets as shapelets
import tslearn.utils as tsutil
import numpy as np
import pandas
import csv
from pyts.transformation import ShapeletTransform
from pyts.approximation import SymbolicAggregateApproximation
from pyts.approximation import PiecewiseAggregateApproximation
import shelve
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

sample_split = 0.3;
sample_percent_size = 0.15;
sample_size = 100;

aplusix_data = pandas.read_csv('D:\Programs\THS-MS\Complete Dataset\Extracted files\StandardizedDataset.csv');

sampled_data = aplusix_data.sample(frac=sample_percent_size);
#sampled_data = aplusix_data.iloc[0:471,:]

sampled_data_x = sampled_data.iloc[:,1:85];
sampled_data_x = sampled_data_x.to_numpy();
#sampled_data_x = sampled_data_x.reshape(1,len(sampled_data_x),125)

sampled_data_y = sampled_data.iloc[:,86:87];
sampled_data_y = sampled_data_y.to_numpy(dtype='str');

sax = piecewise.SymbolicAggregateApproximation(n_segments=100, alphabet_size_avg=5)
sax_data = sax.transform(sampled_data_x)
shrunk_sax_data = sax_data[0]

transformer = PiecewiseAggregateApproximation(window_size=2)
paa = transformer.transform(sampled_data_x)

sax = SymbolicAggregateApproximation(n_bins=5, strategy='normal')
X_sax = sax.fit_transform(sampled_data_x)
X_sax1 = sax.transform(sampled_data_x)


Xo = [[0, 2, 3, 4, 3, 2, 1],
     [0, 1, 3, 4, 3, 4, 5],
     [2, 1, 0, 2, 1, 5, 4],
     [1, 2, 2, 1, 0, 3, 5]]


# Parameters
n_samples, n_timestamps = 100, 48

# Toy dataset
rng = np.random.RandomState(41)
X = rng.randn(n_samples, n_timestamps)

# PAA transformation
window_size = 6
paa = PiecewiseAggregateApproximation(window_size=window_size)
X_paa = paa.transform(X)

'''

fake_time = range(len(sampled_data_x))

plt.plot(fake_time,sampled_data_x[:,0:1])
plt.show()

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
    globals()[key] = my_shelf[key]
my_shelf.close()


import pickle

f = open('store.pckl', 'wb')
pickle.dump(obj, f)
f.close()

f = open('store.pckl', 'rb')
obj = pickle.load(f)
f.close()
'''