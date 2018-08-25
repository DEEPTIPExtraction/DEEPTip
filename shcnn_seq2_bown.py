in_seed = 3545
import numpy as np
np.random.seed(in_seed)
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="4"
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from os import system as command_call
import types,string,re
from util import build_data_cv,make_ds,random_shuffle

home = '/data/yong/api_mining/'
level='para'

cv = 10
if False:
    command_call('rm -rf shcnn_seq2_bown/data/')
    command_call('mkdir shcnn_seq2_bown/data/')
    raw_data = build_data_cv(level=level,home=home)
    cnn_ids = np.array([])
    for k in range(cv):
        train_x,test_x,train_y,test_y,train_ids,test_ids = make_ds(k,raw_data)
        train_list=list(zip(train_x,train_y,train_ids))
        train_x,train_y,train_ids = zip(*random_shuffle(train_list,rand_seed=in_seed))
        test_list=list(zip(test_x,test_y,test_ids))
        test_x,test_y,test_ids = zip(*random_shuffle(test_list,rand_seed=in_seed))
        cnn_ids = np.append(cnn_ids,test_ids)
        ##training data
        f = open('shcnn_seq2_bown/data/cv{}-train.txt.tok'.format(k),'w',encoding='utf-8')
        f.write('\n'.join(train_x)+'\n')
        f.close()
        f = open('shcnn_seq2_bown/data/cv{}-train.cat'.format(k),'w',encoding='utf-8')
        f.write('\n'.join(train_y)+'\n')
        f.close()
        ##test data
        f = open('shcnn_seq2_bown/data/cv{}-test.txt.tok'.format(k),'w',encoding='utf-8')
        f.write('\n'.join(test_x)+'\n')
        f.close()
        f = open('shcnn_seq2_bown/data/cv{}-test.cat'.format(k),'w',encoding='utf-8')
        f.write('\n'.join(test_y)+'\n')
        f.close()
        ##dataset text
        citation = np.concatenate((train_x, test_x), axis=0)
        f = open('shcnn_seq2_bown/data/allds.txt.tok'.format(k),'w',encoding='utf-8')
        f.write('\n'.join(citation)+'\n')
        f.close()

command_call('rm -rf shcnn_seq2_bown/model/')
command_call('mkdir shcnn_seq2_bown/model/')
command_call('rm -rf shcnn_seq2_bown/out/')
command_call('mkdir shcnn_seq2_bown/out/')
command_call('rm -rf shcnn_seq2_bown/temp/')
command_call('mkdir shcnn_seq2_bown/temp/')

for nd in [200,500,800,1000]:
    for nodes0 in [5,15,20,25,30]:
        for step_size in [0.15,0.25,0.35]:
            for mini_batch_size in [32,64,128]:
                psz0=3
                psz1=4
                reg=1e-4
                z='s2bn'
                options="LowerCase UTF8"
                cnn_preds = np.array([])
                cnn_actuals = np.array([])
                cnn_probs = np.array([])
                epoch=180
                for k in range(cv):
                    #step1 generate vocanulary for NB weights
                    voc123='shcnn_seq2_bown/temp/cv{}-{}_trn-123gram.vocab'.format(k,z)
                    command_call('rm -f {}'.format(voc123))
                    for nn in [1,2,3]:
                        vocab_fn='shcnn_seq2_bown/temp/cv{}-{}_trn-{}gram.vocab'.format(k,z,nn)
                        command_call('bin/./prepText gen_vocab input_fn=shcnn_seq2_bown/data/allds.txt.tok \
                            vocab_fn={} {} WriteCount n={}'.format(vocab_fn,options,nn))
                        command_call('cat {} >> {}'.format(vocab_fn,voc123))

                    #Step 2-1. Generate NB-weights
                    command_call('bin/./prepText gen_nbw {} nbw_fn=shcnn_seq2_bown/temp/cv{}-{}.nbw3.dmat \
                        vocab_fn={} train_fn=shcnn_seq2_bown/data/cv{}-train \
                        label_dic_fn=labels.catdic'.format(options,\
                            k,z,voc123,k))
                    #Step 2-2.  Generate NB-weighted bag-of-ngram files
                    for s in ['train','test']:
                        command_call('bin/./prepText gen_nbwfeat {} vocab_fn={} \
                            input_fn=shcnn_seq2_bown/data/cv{}-{} \
                            output_fn_stem=shcnn_seq2_bown/temp/cv{}-{}_{}-nbw3 x_ext=.xsmatcvar \
                            label_dic_fn=labels.catdic nbw_fn=shcnn_seq2_bown/temp/cv{}-{}.nbw3.dmat'\
                            .format(options,voc123,k,s,k,z,s,k,z))

                    #Step 3.  Generate vocabulty for CNN
                    max_num=25000
                    vocab_fn='shcnn_seq2_bown/temp/cv{}-{}_trn-1gram.{}.vocab'.format(k,z,max_num)
                    command_call('bin/./prepText gen_vocab input_fn=shcnn_seq2_bown/data/allds.txt.tok \
                        vocab_fn={} max_vocab_size={} {} WriteCount'\
                        .format(vocab_fn,max_num,options))

                    #Step 4. Generate region files (${tmpdir}/*.xsmatbcvar) and target files (${tmpdir}/*.y) for training and testing CNN.  
                    # We generate region vectors of the convolution layer and write them to a file, instead of making them 
                    # on the fly during training/testing.
                    for pch_sz in [psz0,psz1]:
                        for s in ['train','test']:
                            rnm='shcnn_seq2_bown/temp/cv{}-{}_{}-p{}'.format(k,z,s,pch_sz)
                            command_call('bin/./prepText gen_regions {} region_fn_stem={} \
                                input_fn=shcnn_seq2_bown/data/cv{}-{} \
                                vocab_fn={} label_dic_fn=labels.catdic \
                                x_ext=.xsmatcvar patch_size={} padding={}'.
                                format(options,rnm,k,s,vocab_fn,pch_sz,pch_sz-1))

                    #Step 5. Training
                    
                    nodes1=nd 
                    nodes2=nd
                    command_call('bin/./reNet -1 train datatype=sparse data_dir=shcnn_seq2_bown/temp \
                        trnname=cv{}-{}_train- NoTest \
                        dsno0=nbw3 dsno1=p{} dsno2=p{} \
                        x_ext={} \
                        num_epochs={} ss_scheduler=Few ss_decay=0.1 ss_decay_at={} \
                        loss=Square reg_L2={} top_reg_L2=1e-4 step_size={} top_dropout=0.5 \
                        momentum=0.9 mini_batch_size={} random_seed=3545 \
                        layers=3 conn=0-top,1-top,2-top ConcatConn \
                        0layer_type=Weight+ 1layer_type=Weight+ 2layer_type=Weight+ \
                        0nodes={}  0dsno=0 \
                        1nodes={}  1dsno=1 \
                        2nodes={}  2dsno=2 \
                        activ_type=Rect pooling_type=Max num_pooling=1 resnorm_type=Text \
                        save_after=10 save_interval=5 save_fn=shcnn_seq2_bown/model/cv{}-{}-{}-{}-{}-shcnn.mod'.\
                        format(k,z,psz0,psz1,'.xsmatcvar',epoch,
                            int(epoch*0.8),reg,step_size,mini_batch_size,
                            nodes0,nodes1,nodes2,k,nd,nodes0,step_size,mini_batch_size))

results = []
for nd in [200,500,800,1000]:
    for nodes0 in [5,15,20,25,30]:
        for step_size in [0.15,0.25,0.35]:
            for mini_batch_size in [32,64,128]:
                for epoch in list(range(15,epoch+2,5)):
                    cnn_preds = np.array([])
                    cnn_actuals = np.array([])
                    cnn_probs = np.array([])
                    for k in range(cv):
                        #Prediction
                        command_call('bin/./reNet -1 predict model_fn=shcnn_seq2_bown/model/cv{}-{}-{}-{}-{}-shcnn.mod.epo{}.ReNet \
                            prediction_fn=shcnn_seq2_bown/out/cv{}-test.pred.txt WriteText \
                            datatype=sparse data_dir=shcnn_seq2_bown/temp \
                            tstname=cv{}-{}_test- x_ext=.xsmatcvar \
                            dsno0=nbw3 dsno1=p{} dsno2=p{} \
                            test_mini_batch_size={}'.\
                            format(k,nd,nodes0,step_size,mini_batch_size,epoch,k,k,z,psz0,psz1,mini_batch_size))

                        def softmax(x):
                          e_x = np.exp(x - np.max(x))
                          return e_x / e_x.sum()

                        preds = []
                        p = []
                        with open('shcnn_seq2_bown/out/cv{}-test.pred.txt'.format(k),'r',encoding='utf-8') as ins:
                            for line in ins:
                                temp = line.replace('\n','').split()
                                temp = np.array(temp).astype(np.float32)
                                preds.append(np.argmax(np.array(temp)))
                                p.append(softmax(temp)[1])
                        actuals = []
                        with open('shcnn_seq2_bown/temp/cv{}-{}_test-nbw3.y'.format(k,z),'r',encoding='utf-8') as ins:
                            for line in ins:
                                if line == 'sparse 2\n':
                                    continue
                                actuals.append(int(line.replace('\n','')))
                        cnn_preds = np.append(cnn_preds,preds)
                        cnn_actuals = np.append(cnn_actuals,actuals)
                        cnn_probs = np.append(cnn_probs,p)
                        if k == cv-1:
                            fpr, tpr, thresholds = roc_curve(cnn_actuals, cnn_probs)
                            roc_auc = round(auc(fpr, tpr),3)
                            print('-'*25)
                            print('epoch is: ',epoch)
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
                            results.append("{} {} {} {} {} {} {} {} {} {}".\
                                format(nd,nodes0,step_size,mini_batch_size,epoch,roc_auc,pre,recall,f1,acc))

f=open('shcnn_seq2_bown/part1-explore-paras-perform.txt','w')
f.write('\n'.join(results))
f.close()        














