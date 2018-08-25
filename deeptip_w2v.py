in_seed = 3545
import numpy as np
np.random.seed(in_seed)
import tensorflow as tf 
tf.set_random_seed(in_seed)
import string,re
import pickle,types
from sklearn.metrics import f1_score
from tensorflow.contrib import learn
from sklearn.metrics import roc_curve, auc
from util import remove_puncs,process_raw_txt
from sklearn.metrics import confusion_matrix
import os
from cnn_optimal import text_cnn
os.environ["CUDA_VISIBLE_DEVICES"]="2"
home = '/data/yong/api_mining/'
if __name__ == '__main__':
    img_rows = 100
    filter_sizes = [3,4,5]
    num_filters = 128
    batch_size = 128 #
    epochs = 50
    l2_reg_lambda = 3.0
    level='sent'
    mode = 'static'
    word_vectors = 'nonrand'
    process_mode=''
    print('--{}--{}--{}--{}'.format(level,mode,word_vectors,process_mode))
    with open(home+'datasets/w2v_model/wv_{}.pickle'.format(process_mode), 'rb') as handle:
        vocab = pickle.load(handle)
    embeddings_index = {}
    for word in vocab:
        embeddings_index[word] = vocab[word]
    for word in embeddings_index:
        img_cols = len(embeddings_index[word])
        break
    res = text_cnn(
                embeddings_index=embeddings_index,
                img_cols=img_cols,
                img_rows=img_rows,
                filter_sizes=filter_sizes,
                num_filters=num_filters,
                batch_size=batch_size,
                epochs=epochs,
                l2_reg_lambda=l2_reg_lambda,
                in_seed=in_seed,
                mode=mode,
                word_vectors=word_vectors,
                process_mode=process_mode,
                level=level,
                save_fv=True,
                )




"""
this is optimal parameters for paragraph level


img_rows = 100
    filter_sizes = [3,4,5]
    num_filters = 128
    batch_size = 128 #
    epochs = 50
    l2_reg_lambda = 3.0
    level='para'
    mode = 'static'
    word_vectors = 'nonrand'
    process_mode=''
    print('--{}--{}--{}--{}'.format(level,mode,word_vectors,process_mode))
    with open(home+'datasets/w2v_model/wv_{}.pickle'.format(process_mode), 'rb') as handle:
        vocab = pickle.load(handle)
    embeddings_index = {}
    for word in vocab:
        embeddings_index[word] = vocab[word]
    for word in embeddings_index:
        img_cols = len(embeddings_index[word])
        break
    res = text_cnn(
                embeddings_index=embeddings_index,
                img_cols=img_cols,
                img_rows=img_rows,
                filter_sizes=filter_sizes,
                num_filters=num_filters,
                batch_size=batch_size,
                epochs=epochs,
                l2_reg_lambda=l2_reg_lambda,
                in_seed=in_seed,
                mode=mode,
                word_vectors=word_vectors,
                process_mode=process_mode,
                level=level,
                save_fv=False,
                )

"""