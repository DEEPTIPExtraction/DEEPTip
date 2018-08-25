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
from util import build_data_cv,make_ds,random_shuffle,build_unlabeled_data

batch_size = 256

num_epochs = 200
global_dim = 500
global_un_epc = 10
global_rs = 4
f_pch_sz=3
assigned_cnn_model_epochs = 100
top_num=15

home = '/data/yong/api_mining/'
level='sent'
cv = 10
command_call('rm -rf shcnn_3unsemb/for_sup/')
command_call('mkdir shcnn_3unsemb/for_sup/')
command_call('rm -rf shcnn_3unsemb/for_semi/')
command_call('mkdir shcnn_3unsemb/for_semi/')
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
    f = open('shcnn_3unsemb/for_sup/cv{}-train.txt.tok'.format(k),'w',encoding='utf-8')
    f.write('\n'.join(train_x)+'\n')
    f.close()
    f = open('shcnn_3unsemb/for_sup/cv{}-train.cat'.format(k),'w',encoding='utf-8')
    f.write('\n'.join(train_y)+'\n')
    f.close()
    ##test data
    f = open('shcnn_3unsemb/for_sup/cv{}-test.txt.tok'.format(k),'w',encoding='utf-8')
    f.write('\n'.join(test_x)+'\n')
    f.close()
    f = open('shcnn_3unsemb/for_sup/cv{}-test.cat'.format(k),'w',encoding='utf-8')
    f.write('\n'.join(test_y)+'\n')
    f.close()
    #unlabeled data
    unlabeled_data = build_unlabeled_data(level=level,home=home)
    citation = np.concatenate((unlabeled_data,train_x), axis=0)
    f = open('shcnn_3unsemb/for_semi/cv{}-unlab.txt.tok'.format(k),'w',encoding='utf-8')
    f.write('\n'.join(citation)+'\n')
    f.close()

command_call('rm -rf shcnn_3unsemb/saved_rg/')
command_call('mkdir shcnn_3unsemb/saved_rg/')
command_call('rm -rf shcnn_3unsemb/saved_model/')
command_call('mkdir shcnn_3unsemb/saved_model/')
command_call('rm -rf shcnn_3unsemb/scores/')
command_call('mkdir shcnn_3unsemb/scores/')

paras_str = "{}".format(global_rs)

command_call('rm -rf shcnn_3unsemb/out/')
command_call('rm -rf shcnn_3unsemb/data/')
command_call('rm -rf shcnn_3unsemb/temp/')
command_call('mkdir shcnn_3unsemb/out/')
command_call('mkdir shcnn_3unsemb/data/')
command_call('mkdir shcnn_3unsemb/temp/')

cnn_preds = np.array([])
cnn_actuals = np.array([])
#starting business
for k in range(cv):
    rs1 = global_rs
    dim1 = global_dim
    epo1 = global_un_epc
    decay_epo1 = int(0.8*epo1)
    #---  vocabulary for Y (target)
    command_call('bin/./prepText gen_vocab input_fn=shcnn_3unsemb/for_semi/cv{}-unlab.txt.tok \
        max_vocab_size=50000 \
        vocab_fn=shcnn_3unsemb/data/cv{}-minstop-uns.vocab WriteCount stopword_fn=minstop.txt UTF8 LowerCase'.format(k,k))
    #---  vocabulary for X (regions)
    command_call('bin/./prepText gen_vocab input_fn=shcnn_3unsemb/for_sup/cv{}-train.txt.tok \
        max_vocab_size=50000 \
        vocab_fn=shcnn_3unsemb/data/cv{}-trn.vocab WriteCount UTF8 LowerCase'.format(k,k))

    #generate region file and target file
    unsnm = "cv{}-uns-p{}".format(k,rs1)
    command_call('bin/./prepText gen_regions_unsup x_ext=.xsmatbc y_ext=.ysmatbc x_type=Bow \
        input_fn=shcnn_3unsemb/for_semi/cv{}-unlab.txt.tok \
        x_vocab_fn=shcnn_3unsemb/data/cv{}-trn.vocab \
        y_vocab_fn=shcnn_3unsemb/data/cv{}-minstop-uns.vocab \
        region_fn_stem=shcnn_3unsemb/temp/{} \
        UTF8 LowerCase \
        patch_size={} patch_stride=1 padding={} dist={}'.format(k,k,k,unsnm,rs1,rs1-1,rs1))
    #unsupervised embedding traing
    command_call('bin/./reNet -1 train data_dir=shcnn_3unsemb/temp trnname={} \
        0save_layer_fn=shcnn_3unsemb/out/{}.dim{} \
        datatype=sparse x_ext=.xsmatbc y_ext=.ysmatbc NoCusparseIndex \
        step_size=0.5 ss_scheduler=Few ss_decay=0.1 ss_decay_at={} num_epochs={} \
        NoTest Regression loss=Square mini_batch_size=100 momentum=0.9 reg_L2=0 random_seed=3545 \
        zero_Y_weight=0.2 zero_Y_ratio=5 \
        layers=1 0layer_type=Weight+ 0nodes={} 0activ_type=Rect 0resnorm_type=Text inc=500000'.\
        format(unsnm,unsnm,dim1,decay_epo1,epo1,dim1))

    dim2 = global_dim
    epo2 = global_un_epc
    decay_epo2 = int(0.8*epo2)
    rs2 = global_rs
    #---  vocabulary for Y (target)
    command_call('bin/./prepText gen_vocab input_fn=shcnn_3unsemb/for_semi/cv{}-unlab.txt.tok \
        max_vocab_size=50000 \
        vocab_fn=shcnn_3unsemb/data/cv{}-minstop-uns.vocab WriteCount stopword_fn=minstop.txt UTF8 LowerCase'.format(k,k))
    #---  vocabulary for X (regions)
    fns = ""
    for nn in [1,2,3]:
        command_call('bin/./prepText gen_vocab input_fn=shcnn_3unsemb/for_sup/cv{}-train.txt.tok \
            vocab_fn=shcnn_3unsemb/data/cv{}-trn-{}gram.vocab WriteCount UTF8 LowerCase n={}'.format(k,k,nn,nn))
        fns = fns + "shcnn_3unsemb/data/cv{}-trn-{}gram.vocab".format(k,nn)+" "
    #merging grams
    command_call('perl merge_dic.pl 50000 {} > {}'.format(fns,\
        "shcnn_3unsemb/data/cv{}-trn-allgrams.vocab".format(k)))
    #generate region file and target file
    unsnm = "cv{}-unsx3-p{}".format(k,rs2)
    command_call("bin/./prepText gen_regions_unsup x_ext=.xsmatbc y_ext=.ysmatbc \
         x_type=Bow input_fn=shcnn_3unsemb/for_semi/cv{}-unlab.txt.tok \
         x_vocab_fn=shcnn_3unsemb/data/cv{}-trn-allgrams.vocab \
         y_vocab_fn=shcnn_3unsemb/data/cv{}-minstop-uns.vocab \
         region_fn_stem=shcnn_3unsemb/temp/{} UTF8 LowerCase \
         patch_size={} patch_stride=1 padding={} dist={}".format(k,k,k,unsnm,rs2,rs2-1,rs2))
    command_call('bin/./reNet -1 train data_dir=shcnn_3unsemb/temp trnname={} \
        0save_layer_fn=shcnn_3unsemb/out/{}.dim{} \
        datatype=sparse x_ext=.xsmatbc y_ext=.ysmatbc NoCusparseIndex \
        step_size=0.5 ss_scheduler=Few ss_decay=0.1 ss_decay_at={} num_epochs={} \
        NoTest Regression loss=Square mini_batch_size=100 momentum=0.9 reg_L2=0 random_seed=3545 \
        zero_Y_weight=0.2 zero_Y_ratio=5 \
        layers=1 0layer_type=Weight+ 0nodes={} 0activ_type=Rect 0resnorm_type=Text inc=500000'.\
        format(unsnm,unsnm,dim2,decay_epo2,epo2,dim2))

    rs3 = global_rs
    dim3 = global_dim
    epo3 = global_un_epc
    decay_epo3 = int(0.8*epo3)
    assigned_cnn_model_bs = 256
    
    #genrating region file for X 
    command_call('bin/./prepText gen_vocab input_fn=shcnn_3unsemb/for_sup/cv{}-train.txt.tok \
        max_vocab_size=50000 \
        vocab_fn=shcnn_3unsemb/data/cv{}-for-parsup.vocab WriteCount UTF8 LowerCase'.format(k,k))
 
    command_call('python3 mod_for_semi.py {} {} {} {}'.format(f_pch_sz,assigned_cnn_model_bs,\
        assigned_cnn_model_epochs,k))

    rnm = "shcnn_3unsemb/temp/cv{}-uns-p{}".format(k,f_pch_sz)
    command_call('bin/./prepText gen_regions region_fn_stem={} \
        input_fn=shcnn_3unsemb/for_semi/cv{}-unlab \
        vocab_fn=shcnn_3unsemb/data/cv{}-for-parsup.vocab NoSkip RegionOnly UTF8 LowerCase \
        patch_size={} patch_stride=1 padding={}'.format(rnm,k,k,f_pch_sz,f_pch_sz-1))
    # Apply a supervised model to unlabeled data to obtain embedded regions. 
    command_call('bin/./reNet -1 write_embedded 0DisablePooling test_mini_batch_size=100 \
        datatype=sparse tstname={} \
        model_fn=shcnn_3unsemb/for_semi/cv{}-for-parsup-p{}.supmod.epo{}.ReNet \
        num_top={} embed_fn={}.emb.smats'.format(\
            rnm,k,f_pch_sz,assigned_cnn_model_epochs,\
            top_num,rnm))
    # Generate input files for embedding training with partially-supervised target
    unsnm="cv{}-parsup-p{}p{}".format(k,f_pch_sz,rs3)
    command_call('bin/./prepText gen_regions_parsup x_ext=.xsmatbc y_ext=.ysmatc x_type=Bow \
        input_fn=shcnn_3unsemb/for_semi/cv{}-unlab.txt.tok \
        scale_y=1 \
        x_vocab_fn=shcnn_3unsemb/data/cv{}-for-parsup.vocab \
        region_fn_stem=shcnn_3unsemb/temp/{} \
        UTF8 LowerCase \
        patch_size={} patch_stride=1 padding={} \
        f_patch_size={} f_patch_stride=1 f_padding={} \
        dist={} num_top={} embed_fn={}.emb.smats'.format(\
            k,k,unsnm,\
            rs3,rs3-1,\
            f_pch_sz,f_pch_sz-1,\
            rs3,top_num,rnm))
    command_call('bin/./reNet -1 train data_dir=shcnn_3unsemb/temp trnname={} \
        0save_layer_fn=shcnn_3unsemb/out/{}-dim{} \
        NoCusparseIndex zero_Y_weight=0.2 zero_Y_ratio=5 \
        NoTest Regression loss=Square random_seed=3545 \
        ss_scheduler=Few ss_decay=0.1 step_size=0.5 ss_decay_at={} num_epochs={} \
        datatype=sparse x_ext=.xsmatbc y_ext=.ysmatc \
        mini_batch_size=100 momentum=0.9 reg_L2=0 \
        layers=1 0nodes={} 0activ_type=Rect 0resnorm_type=Text inc=500000'.format(unsnm,\
            unsnm,dim3,\
            decay_epo3,epo3,\
            dim3))
    rs = global_rs
    s_fn0 = "shcnn_3unsemb/out/cv{}-uns-p{}.dim{}.epo{}.ReLayer0".format(k,rs1,dim1,epo1)
    s_fn1 = "shcnn_3unsemb/out/cv{}-parsup-p{}p{}-dim{}.epo{}.ReLayer0".format(k,f_pch_sz,rs3,dim3,epo3)
    s_fn2 = "shcnn_3unsemb/out/cv{}-unsx3-p{}.dim{}.epo{}.ReLayer0".format(k,rs2,dim2,epo2)
    #Step 1. Generate input files. 
    xvoc1="shcnn_3unsemb/temp/cv{}-trn.vocab".format(k)
    command_call('bin/./reNet -1 write_word_mapping layer0_fn={} \
        layer_type=Weight+ word_map_fn={}'.format(s_fn0,xvoc1))
    xvoc3="shcnn_3unsemb/temp/cv{}-trn-123gram.vocab".format(k)
    command_call('bin/./reNet -1 write_word_mapping layer0_fn={} \
        layer_type=Weight+ word_map_fn={}'.format(s_fn2,xvoc3))
    ss = 'train'
    #dataset#0: for the main layer
    rnm="shcnn_3unsemb/temp/cv{}-{}-p{}seq".format(k,ss,rs)
    command_call("bin/./prepText gen_regions NoSkip UTF8 LowerCase region_fn_stem={} \
        input_fn=shcnn_3unsemb/for_sup/cv{}-{} vocab_fn={} label_dic_fn=labels.catdic \
        patch_size={} padding={}".format(rnm,k,ss,xvoc1,rs,rs-1))
    #dataset#1: for the side layer bow
    rnm="shcnn_3unsemb/temp/cv{}-{}-p{}bow".format(k,ss,rs)
    command_call("bin/./prepText gen_regions NoSkip UTF8 LowerCase Bow RegionOnly region_fn_stem={} \
        input_fn=shcnn_3unsemb/for_sup/cv{}-{} vocab_fn={} \
        patch_size={} padding={}".format(rnm,k,ss,xvoc1,rs,rs-1))
    #dataset#2: for the side layer bag-of-{1,2,3}-grams
    rnm="shcnn_3unsemb/temp/cv{}-{}-p{}x3bow".format(k,ss,rs)
    command_call("bin/./prepText gen_regions NoSkip UTF8 LowerCase Bow RegionOnly region_fn_stem={} \
        input_fn=shcnn_3unsemb/for_sup/cv{}-{} vocab_fn={} \
        patch_size={} padding={}".format(rnm,k,ss,xvoc3,rs,rs-1))
    ss = 'test'
    #dataset#0: for the main layer
    rnm="shcnn_3unsemb/saved_rg/cv{}-{}-p{}seq".format(k,ss,rs)
    command_call("bin/./prepText gen_regions NoSkip UTF8 LowerCase region_fn_stem={} \
        input_fn=shcnn_3unsemb/for_sup/cv{}-{} vocab_fn={} label_dic_fn=labels.catdic \
        patch_size={} padding={}".format(rnm,k,ss,xvoc1,rs,rs-1))
    #dataset#1: for the side layer bow
    rnm="shcnn_3unsemb/saved_rg/cv{}-{}-p{}bow".format(k,ss,rs)
    command_call("bin/./prepText gen_regions NoSkip UTF8 LowerCase Bow RegionOnly region_fn_stem={} \
        input_fn=shcnn_3unsemb/for_sup/cv{}-{} vocab_fn={} \
        patch_size={} padding={}".format(rnm,k,ss,xvoc1,rs,rs-1))
    #dataset#2: for the side layer bag-of-{1,2,3}-grams
    rnm="shcnn_3unsemb/saved_rg/cv{}-{}-p{}x3bow".format(k,ss,rs)
    command_call("bin/./prepText gen_regions NoSkip UTF8 LowerCase Bow RegionOnly region_fn_stem={} \
        input_fn=shcnn_3unsemb/for_sup/cv{}-{} vocab_fn={} \
        patch_size={} padding={}".format(rnm,k,ss,xvoc3,rs,rs-1))
    #Step 2. Training. 
    command_call('bin/./reNet -1 train V2 datatype=sparse \
        data_dir=shcnn_3unsemb/temp trnname=cv{}-train-p{} NoTest \
        dsno0=seq dsno1=bow dsno2=x3bow \
        reg_L2=0 top_reg_L2=1e-3 top_dropout=0.5 step_size=0.25 \
        loss=Square mini_batch_size={} momentum=0.9 random_seed=3545 \
        num_epochs={} ss_scheduler=Few ss_decay=0.1 ss_decay_at={} \
        layers=2 num_sides=3 \
        0layer_type=WeightS+ 0nodes=1000 0activ_type=Rect \
        1layer_type=Pooling 1pooling_type=Max 1num_pooling=1 1resnorm_type=Text  \
        0side0_layer_type=Weight+ 0side0_layer_fn={} 0side0_dsno=1 \
        0side1_layer_type=Weight+ 0side1_layer_fn={} 0side1_dsno=1 \
        0side2_layer_type=Weight+ 0side2_layer_fn={} 0side2_dsno=2 \
        save_after=10 save_interval=5 save_fn=shcnn_3unsemb/saved_model/cv{}-fcnn-{}.mod'.format(\
            k,rs,batch_size,num_epochs,int(0.8*num_epochs),\
            s_fn0,s_fn1,s_fn2,\
            k,paras_str))
results=[]
for epoch in list(range(15,num_epochs+2,5)):
    cnn_preds = np.array([])
    cnn_actuals = np.array([])
    cnn_probs = np.array([])
    for k in range(cv):
        #Step 3. Prediction
        command_call('bin/./reNet -1 predict model_fn=shcnn_3unsemb/saved_model/cv{}-fcnn-{}.mod.epo{}.ReNet \
            prediction_fn=shcnn_3unsemb/out/cv{}-test.pred.txt WriteText \
            datatype=sparse data_dir=shcnn_3unsemb/saved_rg tstname=cv{}-test-p{} \
            dsno0=seq dsno1=bow dsno2=x3bow \
            test_mini_batch_size={}'.\
            format(k,paras_str,epoch,k,k,global_rs,batch_size))
        def softmax(x):
          e_x = np.exp(x - np.max(x))
          return e_x / e_x.sum()

        preds = []
        p = []
        with open('shcnn_3unsemb/out/cv{}-test.pred.txt'.format(k),'r',encoding='utf-8') as ins:
            for line in ins:
                temp = line.replace('\n','').split()
                temp = np.array(temp).astype(np.float32)
                preds.append(np.argmax(np.array(temp)))
                p.append(softmax(temp)[1])
        actuals = []
        with open("shcnn_3unsemb/saved_rg/cv{}-test-p{}seq.y".format(k,global_rs),'r',encoding='utf-8') as ins:
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
            print('precision: ',pre)
            print('recall: ',recall)
            print('F1 score: ',f1)
            results.append("{} {} {} {} {}".format(epoch,roc_auc,pre,recall,f1))
f=open('shcnn_3unsemb/perform.txt','w')
f.write('\n'.join(results))
f.close()
















































