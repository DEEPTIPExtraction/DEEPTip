import re
class TextLoader(object): 
    def __init__(self,file_dir):
        self.dir = file_dir

    def __iter__(self):
        for uid, line in enumerate(open(self.dir,'r',encoding='utf-8')):
            regexp = re.compile(r'<row')
            if regexp.search(line):
                yield line.replace('\n','')

    def read_iter(self):
        for uid, line in enumerate(open(self.dir,'r',encoding='utf-8')):
            yield line.replace('\n','')
            
    def read(self,path):
        lists = []
        with open(path) as obj:
            for line in obj:
                lists.append(line.replace('\n',''))
        return lists
    


