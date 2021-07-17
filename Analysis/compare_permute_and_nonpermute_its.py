import sys, pickle #, cPickle
print(sys.executable)
import sklearn.preprocessing as pre, scipy, numpy as np, matplotlib.pyplot as plt, glob, pyemma as py, sys, os, subprocess
import pandas as pd, seaborn as sns, argparse
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

## load coordinates
n_traj, n_frames, n_feat = 100, 100001, 190

data_list = ['../../../dna_data/AT-all_306K_112-165001-190.npy', 
             '../../../dna_data/GC-core_321K_112-165001-190.npy',
             '../../../dna_data/GC-mid-fix_319K_112-150001-190.npy',
             '../../../dna_data/GC-end-fix_313K_112-150001-190.npy']

data_names = ['AT-all', 'GC-core', 'GC-mix', 'GC-end']

## Compute score with cross-validation

def score_cv(data, dim, lag, test_size=0.5, k_folds=10):
    scores = np.zeros(k_folds)
    
    # temporarily suppress very short-lived progress bars
    # with py.util.contexts.settings(show_progress_bars=False):
    for k in range(k_folds):

        train_data, test_data = train_test_split(data, test_size=0.5)
        print(np.shape(train_data), np.shape(test_data))

        vamp = py.coordinates.vamp(train_data, lag=lag, dim=dim, ncov_max=2)
        scores[k] = vamp.score(test_data)
        print(k, scores[k])

    print(scores)
    return scores

## converts data into inverse over averaged permutable distances

test_size = 0.5
lag = 20
dim = 10
k_folds = 5

sns.set_style('white')
sns.set_palette('muted')
plt.rcParams['font.size'] = 12

labels= ['190 inter and \nintramolecular', '190 inverse inter and \nintramolecular', 
         '100 intermolecular', '100 inverse intermolecular', 
         '55 permutation-free', '55 inverse permutation-free']

fig, axes = plt.subplots(4, figsize=(8, 10), sharex=True, sharey=True)

for y, data_name in enumerate(data_list):

    data = np.load(data_name)[:n_traj, -n_frames:, :n_feat]
    
    n_bp = 10
    unique_i = n_bp*(n_bp+1)//2
    inter_p = np.zeros((np.shape(data)[0], np.shape(data)[1], unique_i))
    name = data_names[y]

    cnt = 0
    for i in range(n_bp):
        for j in range(i+1):

            # if indices are the same, jut add value to array
            if i==j:
                ii = i*n_bp + i
                inter_p[:, :, cnt] = data[:, :, ii]

            else:
                #find compatible indices: eg. 90 = 09
                ij = i*n_bp + j
                ji = j*n_bp + i

                # take the average of two permutable reciprocal distances
                inter_p[:, :, cnt] = 1/((1/data[:, :, ij] + 1/data[:, :, ji])/2)

            # update cnt index for inter_p
            cnt+=1

    scores, errors = [], []

    for coords in [data, data[:, :, :100], inter_p]:

        v = score_cv([1/traj for traj in coords], dim=dim, lag=lag, k_folds=5)
        scores.append(v.mean())
        errors.append(v.std())

        inv_v = score_cv([traj for traj in coords], dim=dim, lag=lag, k_folds=5)
        scores.append(inv_v.mean())
        errors.append(inv_v.std())
        
    # save data
    save_dir = 'nonpermute_scores/'
    np.save(open(f'{save_dir}{name}_scores.npy', 'wb'), np.array(scores))
    np.save(open(f'{save_dir}{name}_errors.npy', 'wb'), np.array(errors))
    
    # plot each sequence
    ax=axes[y]
    ax.bar(labels, scores, yerr=errors, color=sns.color_palette("muted"))
    ax.set_title('pairwise distances')
    ax.set_ylabel("VAMP2 score")
    ax.set_xticklabels(labels, rotation=90)

plt.savefig(f'../../paper_figs/all_permute_vamps', dpi=300, bbox_inches='tight')

