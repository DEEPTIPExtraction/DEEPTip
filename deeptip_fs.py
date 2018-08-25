in_seed = 3545
import numpy as np
np.random.seed(in_seed)
from file_process import TextLoader
from sql import db_query,fetch
from bs4 import BeautifulSoup as BS
import types,re,os,string,math
from datetime import datetime
import sklearn.feature_extraction.text
import pickle
from nltk import word_tokenize
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from util import remove_puncs,process_raw_txt
home = '/home/yong/test/'
home = '/data/yong/api_mining/'

def main(level='para',process_mode='',mode = 'static',word_vectors = 'nonrand'):
    print('---{}---{}---'.format(level,process_mode))
    f=open(home+'onehot-{}_paris_cls_id.txt'.format(level),'r')
    outs = f.readlines()
    f.close()
    one_hot_paris_cls_id = {line.replace('\n','').split(' ')[1]:line.replace('\n','').split(' ')[0] for line in outs}
    f=open(home+'normal-{}-{}-{}-{}_paris_cls_id.txt'.format(level,mode,word_vectors,process_mode),'r')
    outs = f.readlines()
    f.close()
    normal_paris_cls_id = {line.replace('\n','').split(' ')[1]:line.replace('\n','').split(' ')[0] for line in outs}

    with open(home+'datasets/feature_map/{}-{}-{}-{}-fmap.pickle'.format(level,mode,word_vectors,process_mode), 'rb') as handle:
        cnn_fmap = pickle.load(handle)

    with open(home+'datasets/feature_set/{}-{}-sise-entirefset.pickle'.format(level,process_mode),'rb') as handle:
        temp = pickle.load(handle)
    sise_fs = {}
    for entity in temp:
        sise_fs[entity['id']] = entity['fset']
    with open(home+'datasets/w2v_model/wv_{}.pickle'.format(process_mode), 'rb') as handle:
        vocab = pickle.load(handle)
    embeddings_index = {}
    for word in vocab:
        embeddings_index[word] = vocab[word]
    for word in embeddings_index:
        img_cols = len(embeddings_index[word])
        break

    tokenize = lambda doc: doc.lower().split(" ")
    if level == 'para':
        sql = "SELECT tipcands FROM answers";
        res = fetch(sql)
        docs = []
        for item in res:
            item = item['tipcands']
            item = item.replace('\n','')
            item = item.replace('\r','')
            item = item.replace('\r\n','')
            item = item.strip()
            item = re.sub('(<p>|</p>)','',item)
            item = re.sub('<[^>]*>', ' ', item)
            item = process_raw_txt(item,mode=process_mode)
            item = remove_puncs(item)
            docs.append(item)
        sklearn_tfidf = TfidfVectorizer(min_df=0,tokenizer=tokenize)
        sklearn_tfidf.fit(docs)
    else:
        sql = "SELECT tipcands FROM answers";
        res = fetch(sql)
        docs = []
        for item in res:
            item = item['tipcands']
            item = item.replace('\n','')
            item = item.replace('\r','')
            item = item.replace('\r\n','')
            item = item.strip()
            item = re.sub('(<p>|</p>)','',item)
            item = re.sub('<[^>]*>', ' ', item)
            sents = sent_tokenize(item)
            for sent in sents:
                sent = process_raw_txt(sent,mode=process_mode)
                sent = remove_puncs(sent)
                docs.append(sent)
        sklearn_tfidf = TfidfVectorizer(min_df=0,tokenizer=tokenize)
        sklearn_tfidf.fit(docs)
    idf = sklearn_tfidf.idf_
    vocabulary = sklearn_tfidf.vocabulary_

    templates = []
    path = home +'datasets/ngrams/'
    for (path,dirs,files) in os.walk(path):
        for file in files:
            fp = open(path+file)
            output = fp.readlines()
            fp.close()
            templates = templates + [item.split('\t')[0].replace('\n','') for item in output ]

    f=open(home+'datasets/{}_tip.pos'.format(level),'r',encoding='utf-8')
    texts = f.readlines()
    f.close()
    pos_ids = [line.replace('\n','').split('\t')[0] for line in texts]
     
    f=open(home+'datasets/{}_tip.neg'.format(level),'r',encoding='utf-8')
    texts = f.readlines()
    f.close()
    neg_ids = [line.replace('\n','').split('\t')[0] for line in texts]
     
    f=open(home+'datasets/{}_tip.ds'.format(level),'r',encoding='utf-8')
    texts = f.readlines()
    f.close()
    texts = [line.replace('\n','').split('\t') for line in texts]

    id2text = {idx:[remove_puncs(txt)] for idx,txt in texts}
    id2text_with_puncs = {idx:txt for idx,txt in texts}
    train_set = [remove_puncs(item[1]) for item in texts]
    unigram_vectorizer = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(1,1),min_df=5)
    unigram_vectorizer.fit(train_set)
    bigram_vectorizer = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(2,2),min_df=5)
    bigram_vectorizer.fit(train_set)
    trigram_vectorizer = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(3,3),min_df=5)
    trigram_vectorizer.fit(train_set)

    postagged_texts=[]
    for _,txt in texts:
        postagged_texts.append(' '.join([pair[1] for pair in \
                nltk.pos_tag(word_tokenize(remove_puncs(txt)))]).lower())
    pos_unigram_vectorizer = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(1,1),min_df=5)
    pos_unigram_vectorizer.fit(postagged_texts)
    pos_bigram_vectorizer = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(2,2),min_df=5)
    pos_bigram_vectorizer.fit(postagged_texts)
    pos_trigram_vectorizer = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(3,3),min_df=5)
    pos_trigram_vectorizer.fit(postagged_texts)

    matrix = []
    for idx,txt in enumerate(texts):
        if level == 'para':
            sql = "SELECT * FROM paragraphs where `ID`=%s";
        elif level == 'sent':
            sql = "SELECT * FROM sentences where `ID`=%s";
        try:
            item = fetch(sql,txt[0])[0]
        except Exception as e:
            print(txt)

            print(fetch(sql,txt[0]))
            raise e
        temp = []
        temp.append(int(txt[0]))
        #TGrams
        ngram = unigram_vectorizer.transform(id2text[txt[0]])
        temp.append(sum(ngram.tocoo().data))
        ngram = bigram_vectorizer.transform(id2text[txt[0]])
        temp.append(sum(ngram.tocoo().data))
        ngram = trigram_vectorizer.transform(id2text[txt[0]])
        temp.append(sum(ngram.tocoo().data))
        #POSGrams
        tags = [' '.join([pair[1] for pair in nltk.pos_tag(word_tokenize(' '.join(id2text[txt[0]])))]).lower()]
        ngram = pos_unigram_vectorizer.transform(tags)
        temp.append(sum(ngram.tocoo().data))
        ngram = pos_bigram_vectorizer.transform(tags)
        temp.append(sum(ngram.tocoo().data))
        ngram = pos_trigram_vectorizer.transform(tags)
        temp.append(sum(ngram.tocoo().data))
        #Surface
        tokens = id2text[txt[0]][0].split()
        temp.append(len(tokens))
        #template
        ids = 0
        num_t = 0
        wc_pos = 0
        maximum = 0
        for jdx,t in enumerate(templates):
            no = t.replace(' ','\\s')
            no = no.replace('*','[\w|\']+')
            regexp = re.compile(no)
            if regexp.search(item['tipcands']):
                ids = ids + (jdx+1)
                num_t = num_t + 1
                wc_pos = wc_pos + (t.split(' ').index('*')+1)
                if len(t.split(' ')) > maximum:
                    maximum = len(t.split(' '))
        temp.append(ids)
        temp.append(num_t)
        temp.append(wc_pos)
        temp.append(maximum)
        temp = temp + list(sise_fs[txt[0]])
        temp.append(int(one_hot_paris_cls_id[txt[0]]))
        temp.append(int(normal_paris_cls_id[txt[0]]))
        #W2V
        cnt = 0
        cnt1 = 0
        uniw2v = np.zeros(img_cols)
        idfw2v = np.zeros(img_cols)
        for token in tokens:
            if token in embeddings_index:
                uniw2v = np.add(uniw2v,embeddings_index[token])
                if token in vocabulary:
                    idfw2v = np.add(idfw2v,idf[vocabulary[token]]*embeddings_index[token])
                    cnt1 = cnt1 + idf[vocabulary[token]]
                cnt = cnt + 1
        if cnt != 0:
            uniw2v = 1/cnt * uniw2v
            idfw2v = 1/cnt1 * idfw2v
        temp = temp + list(uniw2v) + list(idfw2v) + list(cnn_fmap[txt[0]])

        matrix.append(temp)
        
    print(np.array(matrix).shape)

    features = {}
    for f in matrix:
        ids = str(int(f[0]))
        docs = id2text[ids]
        features[ids] = [f[1:],docs]
     
    cv = 10
    def build_data_cv(cv=cv):
        """
        Loads data and split into 10 folds.
        """
        with open(home+'datasets/cv-{}-{}-dataset.pickle'.format(level,process_mode),'rb') as f:
            revs = pickle.load(f)
        temp = []
        for item in revs:
            ids = item['id']
            fset = features[ids][0]
            txt  = features[ids][1]
            datum  = {"y":item['y'], 
                      # "text": txt,
                      "id":ids,
                      "fset": fset,
                      "split": item['split']}
            temp.append(datum)
        return temp
    
    entirefset = build_data_cv()
    with open(home+'datasets/feature_set/{}-{}-trip-entirefset.pickle'.format(level,process_mode),'wb') as f:
        pickle.dump(entirefset,f)

if __name__ == '__main__':
    for level in ['sent','para']:
        for process_mode in ['']:
            main(level=level,process_mode=process_mode)
















































