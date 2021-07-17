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
parser.add_argument('--reversible', type=bool)    # if true, obeys detail balance
parser.add_argument('--l_rate', type=float)    # learning rate
parser.add_argument('--k', type=int)           # k folds for cv
parser.add_argument('--val_split', type=float)    # validation split 
args = parser.parse_args()

npy_name = args.name
prefix = args.prefix
batch_size = args.batch_size
lag_time = args.lag_time
n_sm = args.n_sm
n_epochs = args.n_epochs
reversible = args.reversible
l_rate = args.l_rate
k_folds = args.k
val_split = args.val_split

load_path = npy_name
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#pwd_features = np.load(load_path)
data = np.load(load_path, allow_pickle=True)

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
            data_p[:, :, cnt] = 1/data[:, :, ii]
            
        else:
            #find compatible indices: eg. 90 = 09
            ij = i*n_bp + j
            ji = j*n_bp + i

            # take reciprocal of average distances
            data_p[:, :, cnt] = 1/ ((data[:, :, ij] + data[:, :, ji])/2)
        
        cnt+=1

# scales all features to the range 0 and 1
scaler = pre.MinMaxScaler(feature_range=(0, 1))
scaler.fit(np.concatenate(data_p))
pwd_features_s = [scaler.transform(item) for item in data_p]
print(np.shape(pwd_features_s))

# delete tot save mem
del data, data_p

from keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min', restore_best_weights=False)


for d_i in range(k_folds): #range(len(pwd_features_s)):
    
    data_train, data_test = train_test_split(pwd_features_s, test_size=val_split)
    print(np.shape(data_test), np.shape(data_train))

    # sets output paths for pkl file  
    prefix_k = f'sm-{n_sm}_k-{d_i}_lag-{lag_time}_ep-{n_epochs}'
    pkl_hde_path = './srv_out/' + prefix_k + npy_name.split('/')[-1].replace('.npy', '.pkl')
    pkl_history_path = './history_out/' + prefix_k + npy_name.split('/')[-1].replace('.npy', '.pkl')
    pkl_score_path = './history_out/' + prefix_k + npy_name.split('/')[-1].replace('.npy', '_score.pkl')
    
    print(pkl_hde_path)
    print(pkl_score_path)

    ## establish hyperparameters for srv training
    hde = HDE(n_feat, n_components=n_sm, lag_time=lag_time, dropout_rate=0, 
              batch_size=batch_size, n_epochs=n_epochs, validation_split=0, 
              batch_normalization=True, learning_rate = l_rate,
              reversible=reversible) #callbacks=[earlyStopping])

    ## format data for fitting
    hde.r_degree = 2
    hde.n_epochs = n_epochs 

    #hde.fit(data_train)
    #hde.fit(data_train, y=data_test)
    hde.fit(data_train, y=data_train) # y is not being used for validation but must have some data

    #use history to save training and val loss over epochs
    history = hde.history
    #pickle.dump([history.history['loss'], history.history['val_loss']], open(pkl_history_path, 'wb'))
        
    # calculate scores on training and test data
    train_list, test_list = [], []
    for k in range(1, n_sm+1):
        train_score = hde.score(data_train, score_k = k)
        test_score = hde.score(data_test, score_k = k)
        train_list.append(train_score)
        test_list.append(test_score)
              
    pickle.dump({'train':train_list, 'test':test_list, 'times':hde.timescales_}, open(pkl_score_path, 'wb'))

    # set callbacks to None and save hde weights
    hde.callbacks = None
    hde.history = None
    pickle.dump(hde, open(pkl_hde_path, 'wb'), protocol=4)