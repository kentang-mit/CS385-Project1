#from tsnecuda import TSNE
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import h5py
import os
import numpy as np
import pickle
from data_processing.gen_bbox import base_dir

# singular value decomp.
def SVD(x):
    mat = x.T@x
    eigenvalue, eigenvec = np.linalg.eig(mat)
    eigenvec = eigenvec[:K, :]
    return x@eigenvec.T


def visualize_tsne(data, K=100, path='visualizations/tsne/tsne_hog_pca'):
    data_tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=1, random_state=1, method='barnes_hut', angle=0.5).fit_transform(data)
    colors = np.zeros((data.shape[0], 3))
    colors[:data.shape[0]//2, 1] = 1
    colors[data.shape[0]//2:, 0] = 1
    plt.axis('off')
    plt.xticks([]) 
    plt.yticks([])
    plt.title('%d-PCA'%K)
    plt.scatter(data_tsne[:,0], data_tsne[:,1], c=colors)
    plt.savefig('%s%d.png'%(path, K))
    plt.cla()
        
if __name__ == '__main__':
    if not os.path.exists('visualizations/tsne'):
        os.makedirs('visualizations/tsne')
    
    f = h5py.File(os.path.join('data_processing', "fddb_hog_train.h5"), 'r')
    data = f['data'][...]
    label = f['label'][...]
    pos_data = data[label==1, ...]
    neg_data = data[label!=1, ...]
    data_ = np.vstack([pos_data[:500], neg_data[:500]])
    
    data = data_.copy()
    data_tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=1, random_state=1, method='barnes_hut', angle=0.5).fit_transform(data)
    colors = np.zeros((data.shape[0], 3))
    colors[:data.shape[0]//2, 1] = 1
    colors[data.shape[0]//2:, 0] = 1
    plt.axis('off')
    plt.xticks([]) 
    plt.yticks([])
    plt.title('w/o PCA')
    plt.scatter(data_tsne[:,0], data_tsne[:,1], c=colors)
    plt.savefig('visualizations/tsne/tsne_hog.png')
    plt.cla()
    
    for K in [20,30,50,100]:
        data = SVD(data_, K)
        visualize_tsne(data, K)