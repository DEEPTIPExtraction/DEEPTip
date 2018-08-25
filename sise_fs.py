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
import collections
home = '/data/yong/api_mining/'


c = '(?:(?!\n<p>|\n<ul>|\n<hr>|\n<pre|\n<ol>|\n<h1>|\n<h2>).)'
def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude

def main(level='para',process_mode=''):
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
        parent_id = item['parent_id']
        temp = []
        temp.append(int(txt[0]))
        temp.append(item['score']) #answer score
        if item['LastEditDate'] == '':
            adate = item['CreationDate']
        else:
            adate = item['LastEditDate']
        sql = "SELECT * FROM threads where `id` = %s"
        q = fetch(sql,item['parent_id'])
        if q[0]['LastEditDate'] == '':
            qdate = q[0]['CreationDate']
        else:
            qdate = q[0]['LastEditDate']
        adate = datetime.strptime(adate.split('.')[0],"%Y-%m-%dT%H:%M:%S")
        qdate = datetime.strptime(qdate.split('.')[0],"%Y-%m-%dT%H:%M:%S")
        diff = adate-qdate
        adiff = diff.total_seconds()/3600.0
        temp.append(adiff)#answer time difference to question
        now = datetime.strptime(datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),"%Y-%m-%dT%H:%M:%S")
        aage = (now-adate).total_seconds()/3600.0
        temp.append(aage)#answer age
        temp.append(q[0]['score'])#question score
        temp.append(q[0]['FavoriteCount'])#question favorites
        sql = "SELECT * FROM users where `ID` = %s"
        r = fetch(sql,item['OwnerUserId'])
        if len(r) > 0:
            temp.append(r[0]['Reputation'])#question use reputations
        else:
            temp.append(0)
        temp.append(q[0]['ViewCount'])#question views
        temp.append((now-qdate).total_seconds()/3600.0)#question age
        tokens = id2text[txt[0]][0].split()
        temp.append(len(tokens))#number of tokens in sentence
        sql = "SELECT * FROM answer_body where `ID` = %s"
        body = fetch(sql,item['answer_id'])[0]['body']
        cnt = 0
        try:
            m = re.findall(r'(?:<blockquote>(?:(?!\n<p>|\n<ul>|\n<a href|\n<ol>|\n<h1>|\n<h2>).)*</blockquote>)',body,re.DOTALL)
            for s in m:
                body = body.replace(s,'')
            lis = re.findall(r'(?:<pre%s*><code>%s*</code></pre>)|(?:<p>%s*</p>)|(?:<ol>%s*</ol>)|(?:<ul>%s*</ul>)'%(c,c,c,c,c),body,re.DOTALL)
            l = len(lis)
            i = 0
            paras = []
            while i < l:
                if not lis[i].startswith('<pre'):
                    code = []
                    trigger = False
                    for j in range(i+1,l):
                        trigger = True
                        if lis[j].startswith('<pre'):
                            code.append(lis[j])
                        else:
                            break
                    m = '\n$$$$$\n'.join(code)
                    paras.append([lis[i],m])
                    if trigger:
                        i = j
                    else:
                        i = i + 1
                else:
                    i = i + 1
            
            for unit in paras:
                para = unit[0]
                para = para.replace('\n','')
                para = para.replace('\r','')
                para = para.replace('\r\n','')
                para = para.strip()
                para = re.sub('(<p>|</p>)','',para)
                para = re.sub('<[^>]*>', ' ', para)
                cnt = cnt + len(para.split())
        except Exception as e:
            raise e
        temp.append(cnt)#answer size
        question =[]
        sql = "SELECT * FROM threads WHERE `id` =%s"
        res = fetch(sql,parent_id)[0]
        try:
            body = res['Body'].lower()
            m = re.findall(r'(?:<blockquote>(?:(?!\n<p>|\n<ul>|\n<a href|\n<ol>|\n<h1>|\n<h2>).)*</blockquote>)',body,re.DOTALL)
            for s in m:
                body = body.replace(s,'')
            lis = re.findall(r'(?:<pre%s*><code>%s*</code></pre>)|(?:<p>%s*</p>)|(?:<ol>%s*</ol>)|(?:<ul>%s*</ul>)'%(c,c,c,c,c),body,re.DOTALL)
            l = len(lis)
            i = 0
            paras = []
            while i < l:
                if not lis[i].startswith('<pre'):
                    code = []
                    trigger = False
                    for j in range(i+1,l):
                        trigger = True
                        if lis[j].startswith('<pre'):
                            code.append(lis[j])
                        else:
                            break
                    m = '\n$$$$$\n'.join(code)
                    paras.append([lis[i],m])
                    if trigger:
                        i = j
                    else:
                        i = i + 1
                else:
                    i = i + 1
            for unit in paras:
                para = unit[0]
                para = para.replace('\n','')
                para = para.replace('\r','')
                para = para.replace('\r\n','')
                para = para.strip()
                para = re.sub('(<p>|</p>)','',para)
                para = re.sub('<[^>]*>', ' ', para)
                # para = ' '.join(para.split())
                sents = sent_tokenize(para)
                sents = [remove_puncs(sent)\
                    for sent in sents if len(sent.split()) > 5]
                if len(sents)>0:
                    question = question + sents
                title = res['Title'].lower()
                question.append(remove_puncs(title))
        except Exception as e:
            raise e
        doc = item['tipcands']
        if level=='para':
            para = item['tipcands']
            para = para.replace('\n','')
            para = para.replace('\r','')
            para = para.replace('\r\n','')
            para = para.strip()
            para = re.sub('(<p>|</p>)','',para)
            para = re.sub('<[^>]*>', ' ', para)
            para = ' '.join(para.split())
            sents = sent_tokenize(para)
            sents = [remove_puncs(sent)\
             for sent in sents if len(sent.split()) > 5]
            s = 0.0
            for doc0 in question:
                for doc1 in sents:
                    vec0 = sklearn_tfidf.transform([doc0]).toarray()[0]
                    vec1 = sklearn_tfidf.transform([doc1]).toarray()[0]
                    s = s + cosine_similarity(vec0,vec1)
            if s != 0.0:
                s = s/(len(question)*len(sents))
            temp.append(s)
        else:
            doc1 = remove_puncs(doc)
            s = 0.0
            for doc0 in question:
                vec0 = sklearn_tfidf.transform([doc0]).toarray()[0]
                vec1 = sklearn_tfidf.transform([doc1]).toarray()[0]
                s = s + cosine_similarity(vec0,vec1)
            if s != 0.0:
                s = s/len(question)
            temp.append(s)
        
        postags = [pair[1] for pair in nltk.pos_tag(word_tokenize(' '.join(id2text[txt[0]])))]
        counter=collections.Counter(postags)
        if counter["NN"] > 0:
            temp.append(counter["NN"])#number of nouns
        else:
            temp.append(0)
        if postags[0] == "NN":
            temp.append(1)#sentence starts with noun
        else:
            temp.append(0)

        codes = re.findall(r'(?:<code>(?:(?!<code>).)*</code>)',doc,re.DOTALL)
        num = sum([ len(code.replace('<code>','').replace('</code>','')) for code in codes])
        temp.append(num)#number of characters that are code
        counter=collections.Counter(tokens)
        temp.append(counter['be'])
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
        process_mode = ''
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
    with open(home+'datasets/feature_set/{}-{}-sise-entirefset.pickle'.format(level,process_mode),'wb') as f:
        pickle.dump(entirefset,f)

if __name__ == '__main__':
    for level in ['para','sent']:
        for process_mode in ['']:
            main(level=level,process_mode=process_mode)






















