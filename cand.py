import re,os
import types
from bs4 import BeautifulSoup as BS
from file_process import TextLoader
from multiprocessing import Pool
import itertools,multiprocessing
from os import system as command_call
import string
from sql import db_query,fetch
home = '/home/yong/api_mining/'
command_call('rm -rf '+ home +'datasets/TCans/')
command_call('mkdir '+ home +'datasets/TCans/')
 
core_used = multiprocessing.cpu_count()-10
remove = string.punctuation.replace('_','')#keep underscore for function name
remove = remove.replace('\'','')#leave single quote in something like It's
templates = []
path = home +'datasets/ngrams/'
for (path,dirs,files) in os.walk(path):
    for file in files:
        fp = open(path+file)
        output = fp.readlines()
        fp.close()
        templates = templates + [item.split('\t')[0].replace('\n','') for item in output ]

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

posts = TextLoader(home +'raw_data/Posts.xml')
keywords = posts.read(home +'keywords.ls')
keywords = set(keywords)

sql = "SELECT `parentid`, `acceptedanswerid` FROM kws2TID"
f = fetch(sql)
threadsid = [ ids['parentid'] for ids in f]
acceptedids = [ ids['acceptedanswerid'] for ids in f if ids['acceptedanswerid'] != -1]
acceptedids = set(acceptedids)

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

c = '(?:(?!\n<p>|\n<ul>|\n<hr>|\n<pre|\n<ol>|\n<h1>|\n<h2>).)'
def f(ids):
    ids = set(ids)
    for line in posts:
        soup = BS(line, "lxml")
        pairs = [tag.attrs for tag in soup.findAll('row')][0]
        if pairs['posttypeid'] == '2' and int(pairs['parentid']) in ids:
            body = pairs['body'].lower()
            k = []
            for key in keywords:
                if filter_threads(key,body):
                    k.append(key)
            if not(len(k) > 0):
                continue
            else:
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
                    p = unit[0]
                    txt = re.sub('(<p>|</p>)','',p)
                    txt = re.sub('<[^>]*>', '', txt)#extract content of html tags
                    if len(txt.split()) < 10:
                        continue
                    # print(p)
                    # print(unit[1])
                    # print(txt)
                    # print('------------')
                    # continue
                    input_list = txt.translate(str.maketrans("","",remove)).split()
                    cleaned_str = ' '.join(input_list)
                    for idx,t in enumerate(templates):#template matching
                        t = t.replace(' ','\\s')
                        t = t.replace('*','[\w|\']+')
                        regexp = re.compile(t)
                        if regexp.search(cleaned_str):
                            answer_id = int(pairs['id'])
                            parent_id = int(pairs['parentid'])
                            tipcand = p
                            
                            code = unit[1]
                            score = -1
                            if 'score' in pairs:
                                score = int(pairs['score'])

                            OwnerUserId = -1
                            if 'owneruserid' in pairs:
                                OwnerUserId = int(pairs['owneruserid'])

                            acceptedornot = 0 # no 
                            if answer_id in acceptedids:
                                acceptedornot = 1

                            CommentCount = -1
                            if 'commentcount' in pairs:
                                CommentCount = int(pairs['commentcount'])

                            FavoriteCount = -1
                            if 'favoritecount' in pairs:
                                FavoriteCount = int(pairs['favoritecount'])

                            CreationDate = ''
                            if 'creationdate' in pairs:
                                CreationDate = pairs['creationdate']

                            LastEditorUserId = -1
                            if 'lasteditoruserid' in pairs:
                                LastEditorUserId = int(pairs['lasteditoruserid'])

                            LastEditorDisplayName = ''
                            if 'lasteditordisplayname' in pairs:
                                LastEditorDisplayName = pairs['lasteditordisplayname']

                            LastEditDate = ''
                            if 'lasteditdate' in pairs:
                                LastEditDate = pairs['lasteditdate']

                            LastActivityDate = ''
                            if 'lastactivitydate' in pairs:
                                LastActivityDate = pairs['lastactivitydate']

                            CommunityOwnedDate = ''
                            if 'communityowneddate' in pairs:
                                CommunityOwnedDate = pairs['communityowneddate']

                            ClosedDate = ''
                            if 'closeddate' in pairs:
                                ClosedDate = pairs['closeddate']

                            sql = "INSERT INTO `answers`(`answer_id`,`parent_id`,`tipcands`,`code_block`,`keywords`,`score`,`OwnerUserId`,`acceptedornot`,`CommentCount`,`FavoriteCount`,`CreationDate`,`LastEditorUserId`,`LastEditorDisplayName`,`LastEditDate`,`LastActivityDate`,`CommunityOwnedDate`,`ClosedDate`) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                            db_query(sql,
                                answer_id,
                                parent_id,
                                tipcand,
                                code,
                                ','.join(k),
                                score,
                                OwnerUserId,
                                acceptedornot,
                                CommentCount,
                                FavoriteCount,
                                CreationDate,
                                LastEditorUserId,
                                LastEditorDisplayName,
                                LastEditDate,
                                LastActivityDate,
                                CommunityOwnedDate,
                                ClosedDate)
                            pro_id = multiprocessing.current_process().name.replace('ForkPoolWorker-',"")
                            f = open(home +'datasets/TCans/TipCands%s.txt'%pro_id,'a',encoding = 'utf-8')
                            f.write(txt+'\n')
                            f.close()
                            break

if __name__ == "__main__":
    # f(threadsid)
    influence_parallel(threadsid)
    path = home +'datasets/TCans/'
    command_call('rm -rf '+ home +'datasets/w2v_text.txt')
    for (path,dirs,files) in os.walk(path):
        for file in files:
            fp = open(path+file)
            output = fp.readlines()
            fp.close()
            for line in output:
                f=open(home +'datasets/w2v_text.txt','a',encoding = 'utf-8')
                f.write(line)
                f.close()






















