in_seed = 1
import numpy as np
np.random.seed(in_seed)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from os import system as command_call
# command_call('rm -rf metrics/')
# command_call('mkdir metrics/')

home = '/data/yong/api_mining/'

names = [
        "Logistic",
        "Nearest-Neighbors", 
        "Decision-Tree", 
        "Random-Forest", 
        "Naive-Bayes", 
        # "RBF-SVM"
        ]

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GaussianNB(),
    # SVC(probability=True),
    ]

def make_ds(cv,data):
    train_x,test_x,train_y,test_y,train_ids,test_ids = [],[],[],[],[],[]
    for piece in data:
        if piece['split']==cv:
            test_x.append(piece['fset'])
            test_y.append(piece['y'])
            test_ids.append(piece['id'])
        else:
            train_x.append(piece['fset'])
            train_y.append(piece['y'])
            train_ids.append(piece['id'])
    return train_x,test_x,train_y,test_y,train_ids,test_ids

results = ['Method\tLevel\tAUC\tPrecision\tRecall\tF1-Score\tAccuracy\tExclude']
for level in ['sent','para']:
    for process_mode in ['']:
        with open(home+'datasets/feature_set/{}-{}-trip-entirefset.pickle'.format(level,process_mode),'rb') as f:
            raw_data = pickle.load(f)
        for name, clf in zip(names, classifiers):
            print('---{}---{}---{}---'.format(name,level,process_mode))
            for ex_f in range(0,18):
                cnn_actuals = np.array([])
                cnn_probs = np.array([])
                cnn_preds = np.array([])
                cnn_ids = np.array([])
                print(ex_f)
                for k in range(10):
                    train_x,test_x,train_y,test_y,train_ids,test_ids = make_ds(k,raw_data)
                    train_x = np.array(train_x)
                    test_x = np.array(test_x)
                    # exit(test_x.shape)
                    # train_x = train_x[:,26:27]
                    # test_x = test_x[:,26:27]
                    # train_x = np.delete(train_x,np.s_[28:812],1)
                    # test_x = np.delete(test_x,np.s_[28:812],1)
                    # exit(test_x.shape)
                    if ex_f == 1: #exclude tgrams
                        train_x = train_x[:,3:]
                        test_x = test_x[:,3:]
                    elif ex_f == 2: #exclude posgram
                        train_x = np.delete(train_x,np.s_[3:6],1)
                        test_x = np.delete(test_x,np.s_[3:6],1)
                    elif ex_f == 3: #exclude template
                        train_x = np.delete(train_x,np.s_[6:10],1)
                        test_x = np.delete(test_x,np.s_[6:10],1)
                    elif ex_f == 4: #exclude word CNN
                        train_x = np.delete(train_x,np.s_[428:812],1)
                        test_x = np.delete(test_x,np.s_[428:812],1)
                        train_x = np.delete(train_x,np.s_[27:28],1)
                        test_x = np.delete(test_x,np.s_[27:28],1)
                    elif ex_f == 5: #exclude one hot
                        train_x = np.delete(train_x,np.s_[26:27],1)
                        test_x = np.delete(test_x,np.s_[26:27],1)
                    elif ex_f == 6: #exclude centroids
                        train_x = np.delete(train_x,np.s_[28:428],1)
                        test_x = np.delete(test_x,np.s_[28:428],1)
                    elif ex_f == 7: #exclude uniw2v
                        train_x = np.delete(train_x,np.s_[28:228],1)
                        test_x = np.delete(test_x,np.s_[28:228],1)
                    elif ex_f == 8: #exclude idfw2v
                        train_x = np.delete(train_x,np.s_[228:428],1)
                        test_x = np.delete(test_x,np.s_[228:428],1)
                    elif ex_f == 9: #exclude sise
                        train_x = np.delete(train_x,np.s_[11:26],1)
                        test_x = np.delete(test_x,np.s_[11:26],1)

                    elif ex_f == 10: #only tgrams
                        train_x = train_x[:,0:3]
                        test_x = test_x[:,0:3]
                    elif ex_f == 11: #only posgrams
                        train_x = train_x[:,3:6]
                        test_x = test_x[:,3:6]
                    elif ex_f == 12: #only template
                        train_x = train_x[:,6:10]
                        test_x = test_x[:,6:10]
                    elif ex_f == 13: #only onehot
                        train_x = train_x[:,26:27]
                        test_x = test_x[:,26:27]
                    elif ex_f == 14: #only uniw2v
                        train_x = train_x[:,28:228]
                        test_x = test_x[:,28:228]
                    elif ex_f == 15: #only idfw2v
                        train_x = train_x[:,228:428]
                        test_x = test_x[:,228:428]
                    elif ex_f == 16: #only uniw2v + idfw2v
                        train_x = train_x[:,28:428]
                        test_x = test_x[:,28:428]
                    elif ex_f == 17: #only word CNN
                        train_x = np.delete(train_x,np.s_[28:428],1)
                        test_x = np.delete(test_x,np.s_[28:428],1)
                        train_x = train_x[:,27:]
                        test_x = test_x[:,27:]
                    data_size = len(train_x)
                    rand_state = np.random.RandomState(in_seed)
                    shuffle_indices = rand_state.permutation(np.arange(data_size))
                    train_x = np.array(train_x)[shuffle_indices]
                    train_y = np.array(train_y)[shuffle_indices]

                    data_size = len(test_x)
                    rand_state = np.random.RandomState(in_seed)
                    shuffle_indices = rand_state.permutation(np.arange(data_size))
                    test_x = np.array(test_x)[shuffle_indices]
                    test_y = np.array(test_y)[shuffle_indices]
                    test_ids = np.array(test_ids)[shuffle_indices]
                    
                    clf.fit(train_x, train_y)
                    probs = clf.predict_proba(test_x)[:,1]
                    preds = clf.predict(test_x)
                    cnn_preds = np.append(cnn_preds,preds)
                    cnn_actuals = np.append(cnn_actuals,test_y)
                    cnn_probs = np.append(cnn_probs,probs)
                    cnn_ids = np.append(cnn_ids,test_ids)

                fpr, tpr, thresholds = roc_curve(cnn_actuals, cnn_probs)
                roc_auc = round(auc(fpr, tpr),3)
                print('auc: ', roc_auc)
                cnf_matrix = confusion_matrix(cnn_actuals, cnn_preds)
                print(cnf_matrix)
                pre = round(cnf_matrix[1][1]/(cnf_matrix[1][1]+cnf_matrix[0][1]),3)
                recall = round(cnf_matrix[1][1]/(cnf_matrix[1][1]+cnf_matrix[1][0]),3)
                f1 = round(2*pre*recall/(pre+recall),3)
                acc = round((cnf_matrix[1][1]+cnf_matrix[0][0])/(cnf_matrix[0][0]+cnf_matrix[0][1]+cnf_matrix[1][0]+cnf_matrix[1][1]),3)
                print('precision: ',pre)
                print('recall: ',recall)
                print('F1 score: ',f1)
                print('Accuracy: ',acc)
                results.append(name+'\t'+level+'\t'+str(roc_auc)+'\t'+str(pre)+'\t'+str(recall)+'\t'+\
                    str(f1)+'\t'+str(acc)+'\t'+str(ex_f))
            results.append('\n')
f=open(home+'metrics/classifier-metrics.txt','w')
f.write('\n'.join(results))
f.close()

                # s = []
                # for i in range(len(cnn_preds)):
                #     s.append(str(int(cnn_preds[i]))+' '+cnn_ids[i])
                # f=open(home+'overall-{}_paris_cls_id.txt'.format(level),'w')
                # f.write('\n'.join(s))
                # f.close()
    

















