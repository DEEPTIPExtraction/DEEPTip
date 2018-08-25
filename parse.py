import re
import types
from bs4 import BeautifulSoup as BS
from os import system as command_call
from file_process import TextLoader
from multiprocessing import Pool
import itertools,multiprocessing
from sql import db_query

lang = 'php'
core_used = multiprocessing.cpu_count()-1
home = '/home/yong/api_mining/'
def filter_threads(key,str_body,str_title):
    is_found = False
    re_title = [\
    r'(?i).*\b%s\b.*' % key,\
    r'(?i).*\b(a |an )%s\b.*' % key]
    if str_title != '':
        for i in range(len(re_title)):
            regexp = re.compile(re_title[i])
            if regexp.search(str_title):
                is_found = True
                return is_found
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

sentences = TextLoader(home +'raw_data/php_thread_Posts.txt')     
keywords = sentences.read(home +'keywords.ls')
threadsid = sentences.read(home+'datasets/threadsid.txt')
threadsid = set(threadsid)
posts = sentences.read_iter()

def partitions(pmids, n):
    "Partitions the pmids into n subsets"
    nodes_iter = iter(pmids)
    while True:
        partition = tuple(itertools.islice(nodes_iter,n))
        if not partition:
            return
        yield partition

def influence_pool(g_tuple):
    return f(*g_tuple)

def influence_parallel(ids,processes = core_used):
    p = Pool(processes=processes)
    part_generator = len(p._pool)

    node_partitions = list(partitions(ids, int(len(ids)/part_generator)))
    
    num_partitions = len(node_partitions)
 
    p.map( influence_pool,zip(node_partitions ))

def f(threadsid):
    for line in posts:
        soup = BS(line, "lxml")
        pairs = [tag.attrs for tag in soup.findAll('row')][0]
        if pairs['id'] in threadsid :
            title = ''
            if 'title' in pairs:#at times, post does not have title. weird
                title = pairs['title'].lower()
            body = pairs['body'].lower()
            k = []
            for key in keywords:
                if filter_threads(key,title,body):
                    k.append(key)
            if not(len(k) > 0):
                continue
            else:
                answer_id = -1
                if 'acceptedanswerid' in pairs:
                    answer_id = int(pairs['acceptedanswerid'])
                idx = int(pairs['id'])
                sql = " INSERT INTO `kws2TID` (`parentid`,`keywords`,`acceptedanswerid`)VALUES(%s,%s,%s)"
                db_query(sql,
                        idx,
                        ','.join(k),
                        answer_id
                        )
            # for key in keywords:
            #     if filter_threads(key,title,body):

                    # if 'acceptedanswerid' in pairs:
                    #     f = open(home +'datasets/acceptedanswerid.txt','a')
                    #     f.write(pairs['id']+'\t'+pairs['acceptedanswerid']+'\n')
                    #     f.close()

                    # f = open(home +'datasets/parentid.txt','a')
                    # f.write(pairs['id']+'\n')
                    # f.close()
                    # break
if __name__ == "__main__":
    # f(threadsid)
    influence_parallel(threadsid)



















