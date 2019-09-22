'''

processing the newsgroup Dataset



'''
import glob
class News:
    def __init__(self, filepath,classes):
       # self.docs = {}
        
        ng = open(filepath,"r")
        self.docID = ''
        self.subject = ''
        self.newsgroup = ''
        self.body = ''
        self.class_label=''
        bline=False
        
        n=filepath.split("/")
        
        self.newsgroup=n[len(n)-2]
        self.docID=n[len(n)-1]+"@"+n[len(n)-2]
        
        for key in classes:
            if self.newsgroup.strip() in classes[key]:
                self.class_label=key
                break
        line=ng.readline()
        lines=0
		#Reading Subject and Last xx lines in Lines:xx as body of the document,also handles if the xx in Lines:xx is not a number
        while line:
            if 'Subject' in str(line):
                j=str(line).split("Subject:")
                #print(j[1])
                self.subject=j[1]
            elif 'Lines:' in str(line):
                k=str(line).split("Lines: ")
                if k[1].strip().isdigit():
                    lines=int(k[1].strip())
                else:
                    lines=-1
                break
            line=ng.readline()
        ng.close()
        c=0
        for line1 in reversed(list(open(filepath))):
            c+=1
            if c<=lines:
                self.body=self.body+str(line1)
            else:
                break

      
class AllNews:
    def __init__(self,directoryPath,class_definition_file):
        self.news=[]
        class_dict={'comp.graphics':'1', 'comp.os.ms-windows.misc':'1', 'comp.sys.ibm.pc.hardware':'1', 'comp.sys.mac.hardware':'1',
               'comp.windows.x':'1', 'rec.autos':'2', 'rec.motorcycles':'2', 'rec.sport.baseball':'2', 'rec.sport.hockey':'2',
         'sci.crypt':'3', 'sci.electronics':'3', 'sci.med':'3', 'sci.space':'3', 'misc.forsale':'4',
         'talk.politics.misc':'5', 'talk.politics.guns':'5', 'talk.politics.mideast':'5',
         'talk.religion.misc':'6', 'alt.atheism':'6', 'soc.religion.christian':'6'}
		 #creating class_definition_file
        class_file=open(class_definition_file,"w")
        for i in class_dict.keys():
            class_file.write(class_dict[i]+" "+i+"\n")
        class_file.close()
        print("class_definition file is created\n")
		#updating the class label for each document in mini_newsgroups
        fil=open(class_definition_file,"r")
        classes={}
        r=fil.readline()
        while r:
            p=str(r.strip()).split(" ")
            if p[0] in classes:
                classes[p[0]].append(p[1])
            else:
                classes[p[0]]=[p[1]]
            r=fil.readline()
        fil.close()
       #create a news  object that contains the docID,class_label,subject,body of all the documents/files in mini_newsgroups directory
        path = directoryPath+"//"
      
        folders = [f for f in glob.glob(path + "**/*")]
       
        for f in folders:
         
            k=f.split("/")
          
            file_ID=k[2]
            news_group=k[1]
            n=News(f,classes)
            self.news.append(n)

    # def test():
        

if __name__ == '__main__':
    ''' testing '''
    ng = AllNews ('mini_newsgroups','class_definition_file')
    for doc in ng.news:
        print(doc.subject + " " + doc.docID + " " + doc.class_label + '\n')
         #print(doc.subject+" "+doc.docID+" "+doc.body+" "+doc.class_label+'\n')