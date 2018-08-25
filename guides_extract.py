import re
import types,os
from bs4 import BeautifulSoup as BS
from file_process import TextLoader
from multiprocessing import Pool
import itertools,multiprocessing
from os import system as command_call
from sql import db_query,fetch

home = '/home/yong/api_mining/'

core_used = int(multiprocessing.cpu_count()/2)
command_call('rm -rf /home/yong/api_mining/datasets/TGuides/')
command_call('mkdir /home/yong/api_mining/datasets/TGuides/')

def partitions(pmids, n):
    "Partitions the pmids into n subsets"
    nodes_iter = iter(pmids)
    while True:
        partition = tuple(itertools.islice(nodes_iter,n))
        if not partition:
            return
        yield partition

posts = TextLoader(home+'raw_data/Posts.xml')
keywords = posts.read(home+'keywords.ls')

def filter_threads(key,str_body):
    is_found = False
    # re_title = [\
    # r'(?i).*\b%s\b.*' % key,\
    # r'(?i).*\b(a |an )%s\b.*' % key]
    # if str_title != '':
    #     for i in range(len(re_title)):
    #         regexp = re.compile(re_title[i])
    #         if regexp.search(str_title):
    #             is_found = True
    #             return is_found
    re_body = [\
    r'.*(^|[a-z]+ |[\.!?\(<\=]\s{0,}|\={0,1}\s{0,}\$[\w\d_]*\-\>)%s(\s{0,}[\(\[][\w\s$,\'\"\(\)]*[\)\]]\s{1,}|[>\)\.,!?$]+| [a-z]+).*' % key,\
    r'.*<code>.*\b%s\b.*</code>.*' % key,\
    r'.*<a.*href.*%s\.php.*>.*</a>.*' % key]
    for i in range(len(re_body)):
        regexp = re.compile(re_body[i])
        if regexp.search(str_body):
            is_found = True
            break
    return is_found

def fun(ids):
    ids = set(ids)
    for line in posts:
        soup = BS(line, "lxml")
        pairs = [tag.attrs for tag in soup.findAll('row')][0]
        try:
            if pairs['posttypeid'] == '2' and int(pairs['id']) in ids:
                is_stop_searching_kw = False
                for key in keywords:
                    body = pairs['body'].lower()
                    if filter_threads(key,body):
                        m = re.findall(r'(?:<blockquote>(?:(?!\n<p>|\n<ul>|\n<a href|\n<ol>|\n<h1>|\n<h2>).)*</blockquote>)',body,re.DOTALL)
                        for s in m:
                            body = body.replace(s,'')
                        paras = re.findall('<p>.*</p>',body)
                        paras = paras + re.findall('<li>.*</li>',body)
                        for p in paras:
                            txt = re.sub('(<p>|</p>)','',p)
                            txt = re.sub('<[^>]*>', '', txt)#extract content of html tags 
                            if len(txt.split()) < 10:
                                continue
                            re_str = r'.*(^|[a-z]+ |[\.!?\(<\=]\s{0,}|\={0,1}\s{0,}\$[\w\d_]*\-\>)%s(\s{0,}[\(\[][\w\s$,\'\"\(\)]*[\)\]]\s{1,}|[>\)\.,!?$]+| [a-z]+).*' % key
                            regexp = re.compile(re_str)
                            if regexp.search(txt):
                                pro_id = multiprocessing.current_process().name.replace('ForkPoolWorker-',"")
                                f = open(home+'datasets/TGuides/TGuides%s.txt'%pro_id,'a',encoding = 'utf-8')
                                f.write(txt+'\n')
                                f.close()
                                is_stop_searching_kw = True
                    if is_stop_searching_kw:
                        is_stop_searching_kw = False
                        break
        except Exception as e:
            print(e)
            pass

def influence_pool(g_tuple):
    return fun(*g_tuple)

def influence_parallel(ids,processes = core_used):
    p = Pool(processes=processes)
    part_generator = len(p._pool)

    node_partitions = list(partitions(ids, int(len(ids)/part_generator)))
    
    num_partitions = len(node_partitions)
 
    p.map( influence_pool,zip(node_partitions ))

if __name__ == "__main__":
    # rest = posts.read(home+'datasets/acceptedanswerid.txt')
    # rest = [i.split('\t')[1] for i in rest]
    # rest = set(rest)
    # print(len(rest))
    # fun(rest)
    sql = "SELECT `acceptedanswerid` FROM kws2TID"
    f = fetch(sql)
    # threadsid = [ ids['parentid'] for ids in f]
    # threadsid = set(threadsid)
    acceptedids = [ ids['acceptedanswerid'] for ids in f if ids['acceptedanswerid'] != -1]
    # acceptedids = set(acceptedids)
    print(len(acceptedids))
    # fun(acceptedids)
    influence_parallel(acceptedids)
    













