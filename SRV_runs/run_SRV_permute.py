import sys, pickle #, cPickle
print(sys.executable)
import sklearn.preprocessing as pre, scipy, numpy as np, matplotlib.pyplot as plt, glob, pyemma as py, sys, os
import pandas as pd, seaborn as sns, argparse
from sklearn.model_selection import train_test_split

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]= "2"

from temp_tf_load import *
sys.path.append('../hde/')
print(sys.path)
from hde import HDE, analysis
sys.path.remove('../hde/')
import warnings
warnings.filterwarnings('ignore')

skip_t = 1      # freq of skipping during training
plot_num = 5000  # approximate number of points saved in dataframe and plottded

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)      #name of data densor saved in dna_outputs
parser.add_argument('--prefix', type=str)    #name of prefix based on specific run
parser.add_argument('--batch_size', type=int)
parser.add_argument('--lag_time', type=int)
parser.add_argument('--n_sm', type=int)
parser.add_argument('--n_epochs', type=int)
parser.add_argument('--n_feat', type=int)     #number of features per frame (190)
parser.add_argument('--n_plot', type=int)     #number of plot points to save to df 
parser.add_argument('--reversible', type=bool)    # if true, obeys detail balance
parser.add_argument('--trim_start', type=int)    # index where trim starts
parser.add_argument('--trim_end', type=int)    # index where trim ends
parser.add_argument('--n_traj', type=int)    # index where trim ends
parser.add_argument('--equ_train', type=str)    # equ testing string path
parser.add_argument('--l_rate', type=float)    # equ testing string path
args = parser.parse_args()

npy_name = args.name
prefix = args.prefix
batch_size = args.batch_size
lag_time = args.lag_time
n_sm = args.n_sm
n_epochs = args.n_epochs
n_feat = args.n_feat
plot_num = args.n_plot
reversible = args.reversible
trim_start = args.trim_start
trim_end = args.trim_end
n_traj = args.n_traj 
equ_train = args.equ_train
l_rate = args.l_rate

load_path = '/home/mikejones/scratch-midway2/srv/dna_data/' + npy_name
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#pwd_features = np.load(load_path)
data = np.load(load_path, allow_pickle=True)[:n_traj, trim_start:trim_end]

## load interdistances with averaged permutations
n_bp = 10
n_feat = n_bp*(1+n_bp)//2
data_p = np.zeros((np.shape(data)[0], np.shape(data)[1], n_feat))

cnt = 0
for i in range(n_bp):
    for j in range(i+1):
        
        # if indices are the same, jut add value to array
        if i==j:
            ii = i*n_bp + i
            data_p[:, :, cnt] = data[:, :, ii]
            
        else:
            #find compatible indices: eg. 90 = 09
            ij = i*n_bp + j
            ji = j*n_bp + i

            # take the average of two permutable reciprocal distances
            data_p[:, :, cnt] = 1/((1/data[:, :, ij] + 1/data[:, :, ji])/2)
        
        # update cnt index for inter_p
        cnt+=1

# sets output paths for pkl files
pkl_path = './dataframe_outputs/' + prefix + npy_name.replace('.npy', '.pkl')
pkl_hde_path = './hde_outputs/' + prefix + npy_name.replace('.npy', '.pkl')
pkl_history_path = './history_out/' + prefix + npy_name.replace('.npy', '.pkl')

# scales all features to the range 0 and 1
scaler = pre.MinMaxScaler(feature_range=(0, 1))
scaler.fit(np.concatenate(data_p))
pwd_features_s = [scaler.transform(item) for item in data_p]
print(np.shape(pwd_features_s))

# delete tot save mem
del data, data_p

## establish hyperparameters for srv training
hde = HDE(n_feat, n_components=n_sm, lag_time=lag_time // skip_t, dropout_rate=0, batch_size=batch_size, n_epochs=n_epochs, 
          validation_split=0.2, batch_normalization=True, learning_rate = l_rate, reversible=reversible)

## format data for fitting
hde.r_degree = 2
hde.n_epochs = n_epochs 

## if not equilibrium validation set, validate using training data with validation_split 
if 'None' in equ_train:
    hde.fit(pwd_features_s)
    
## else load and train on equilibrium data
else:
    equ_data = np.load('/home/mikejones/scratch-midway2/srv/dna_data/' + equ_train)
    equ_data_s = scaler.transform(equ_data)
    print(np.shape(equ_data_s))
    hde.fit(pwd_features_s, y=equ_data_s)

#use history to save training and val loss over epochs
history = hde.history
pickle.dump([history.history['loss'], history.history['val_loss']], open(pkl_history_path, 'wb'))

# set callbacks to None and save hde weights
hde.callbacks = None
hde.history = None
pickle.dump(hde, open(pkl_hde_path, 'wb'), protocol=4)