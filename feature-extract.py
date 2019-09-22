import news
import util
import math
import operator
import sys


class Posting:
    def __init__(self, docID):
        self.docID = docID
        self.positions = []

    def append(self, pos):
        self.positions.append(pos)

    def sort(self):
        ''' sort positions'''
        self.positions.sort()

    def merge(self, positions):
        self.positions.extend(positions)

    def term_freq(self):
        ''' return the term frequency in the document'''
        return len(self.positions)


class IndexItem:
    def __init__(self, term):
        self.term = term
        self.posting = {}  # postings are stored in a python dict for easier index building
        self.sorted_postings = []  # may sort them by docID for easier query processing

    def add(self, docid, pos):
        ''' add a posting'''
        if docid not in self.posting:
            self.posting[docid] = Posting(docid)
        self.posting[docid].append(pos)

    def sort(self):
        ''' sort by document ID for more efficient merging. For each document also sort the positions'''
        for key in self.posting:  # sorting documentIDs and positions
            self.posting[key].sort()
        self.sorted_postings = sorted(self.posting.items(), key=operator.itemgetter(0))

    def __str__(self):
        return "{} {}".format(self.term, self.posting)


class InvertedIndex:

    def __init__(self):
        self.items = {}  # list of IndexItems
        self.nDocs = 0  # the number of indexed documents

    def indexDoc(self, doc):  # indexing a Document object
        ''' indexing a docuemnt, using the simple SPIMI algorithm, but no need to store blocks due to the small collection we are handling. Using save/load the whole index instead'''
        self.nDocs += 1
        # print(doc.subject+'\n'+doc.body)
        tokens = util.tokenize(doc.subject + '\n' + doc.body)
        # print(tokens)
        for i, token in enumerate(tokens):
            if token not in self.items:
                self.items[token] = IndexItem(token)
            self.items[token].add(doc.docID, i)
        # print(self.items)

    # return self.items

    # ToDo: indexing only title and body; use some functions defined in util.py
    # (1) convert to lower cases,
    # (2) remove stopwords,
    # (3) stemming

    def sort(self):
        ''' sort all posting lists by docID'''
        for key in self.items:
            self.items[key].sort()

    def find(self, term):
        return self.items[term]

    def save(self, filename, type_of_feature):
        ''' save to disk'''
        # ToDo: using your preferred method to serialize/deserialize the index

    def load(self, filename):

        ''' load from disk'''
        # ToDo

    def idf(self, term):
        """ compute the inverted document frequency for a given term"""
        # ToDo: return the IDF of the term
        if term in self.items.keys():
            return 1.0 + math.log(float(self.nDocs) / len(self.items[term].posting.keys()))
        else:
            return 0.0

    # more methods if needed
    def tf(self, term, docId):
        if docId not in self.items[term].posting:
            return 0
        else:
            return self.items[term].posting[docId].term_freq()

    def tfidf(self, term, docId):
        if self.tf(term, docId) == 0:
            return 0.0
        return self.tf(term, docId) * self.idf(term)

    def __str__(self):
        return "{}".format(self.items)


def indexingNewsGroup(directoryngdata, feature_definition_file, class_definition_file, training_data_file):
    # ToDo: indexing the Cranfield dataset and save the index to a file
    # command line usage: "python index.py cran.all index_file"

    # creating inverted index of all the files in mini newsgroup directory
    ng = news.AllNews(directoryngdata, class_definition_file)

    indexobj = InvertedIndex()
    # creating feature definition file for all the documents
    for doc in ng.news:
        indexobj.indexDoc(doc)
    # print(indexobj.nDocs)
    # print(indexobj.items.keys())
    f = open(feature_definition_file, 'w')
    for feature_id, term in enumerate(indexobj.items.keys()):
        f.write(str(feature_id + 1) + ' ' + term + '\n')
    f.close()
    print("feature_definition_file is created\n")
    features = open(feature_definition_file, "r")
    h = features.readline()
    features_dict = {}
    while h:
        # print(h)
        k = h.split(" ")
        features_dict[k[1].strip()] = k[0]
        h = features.readline()
    features.close()
    # creating training data file in libsvm format
    print("Creating training_data_files\n")
    training_file = open(training_data_file, "w")#training data file containining TFIDF as feature value
    training_file_TF = open('training_data_file.TF', "w")#training data file containining TF as feature value
    training_file_IDF = open('training_data_file.IDF', "w")#training data file containining IDF as feature value

    for doc in ng.news:
        tokens1 = set(util.tokenize(doc.subject + '\n' + doc.body))
        tokens = list(tokens1)

        write_dict = {}
        write_dict_TF={}
        write_dict_IDF={}

        training_file.write(doc.class_label + " ")
        training_file_TF.write(doc.class_label + " ")
        training_file_IDF.write(doc.class_label + " ")
        for token in tokens:
            write_dict[int(features_dict[token])] = ":" + str(indexobj.tfidf(token, doc.docID)) + " "
            write_dict_TF[int(features_dict[token])] = ":" + str(indexobj.tf(token, doc.docID)) + " "
            write_dict_IDF[int(features_dict[token])] = ":" + str(indexobj.idf(token)) + " "
        # training_file.write("\n")
        id_list = list(write_dict.keys())
        id_list_TF = list(write_dict_TF.keys())
        id_list_IDF = list(write_dict_IDF.keys())
        id_list.sort()
        id_list_TF.sort()
        id_list_IDF.sort()
        for id in id_list:
            training_file.write(str(id) + write_dict[id])
        training_file.write("\n")
        for id in id_list_TF:
            training_file_TF.write(str(id) + write_dict_TF[id])
        training_file_TF.write("\n")
        for id in id_list_IDF:
            training_file_IDF.write(str(id) + write_dict_IDF[id])
        training_file_IDF.write("\n")
    training_file.close()
    training_file_TF.close()
    training_file_IDF.close()
    print("training_data_file is created\n")
    print("training_data_file.TF is created\n")
    print("training_data_file.IDF is created\n")
    print("Done")


def test():
    print("test1:displays number of features generated from all the documents\n")
    f = open('feature_definition_file', 'r')
    count = 0
    k = f.readline()
    while k:
        count += 1
        k = f.readline()
    print("number of features generated in feature_defintion_file : " + str(count))
    f.close()
    print("test2:verified that all the documents are read and parsed from the mininewsgroup directory\n")
    f = open('training_data_file', 'r')
    count = 0
    k = f.readline()
    while k:
        count += 1
        k = f.readline()
    print("number of documents parsed from mininewsgroup : " + str(count))
    print("test3 : Given a filename and filepath parse the document\n")
    fil = open('class_definition_file', "r")
    classes = {}
    r = fil.readline()
    while r:
        p = str(r.strip()).split(" ")
        if p[0] in classes:
            classes[p[0]].append(p[1])
        else:
            classes[p[0]] = [p[1]]
        r = fil.readline()
    fil.close()
    directorypath = input("Enter the filepath (eg:localpath/mini_newsgroups/alt.atheism/51121):\n")
    ngobj = news.News(directorypath, classes)
    print("DOCID : " + ngobj.docID)
    print("Newsgroup : " + ngobj.newsgroup)
    print("Class : " + ngobj.class_label)
    print("Subject : " + ngobj.subject)
    print("Body : " + ngobj.body)
    print("test4\n")
    print("Tokenizing the subject and body of the above given file,removing stop words and stemming: \n")
    print(util.tokenize(ngobj.subject + " " + ngobj.body))
    print("test5 : printing inverted index of the given file\n")
    indexobjtest = InvertedIndex()
    indexobjtest.indexDoc(ngobj)
    for key in indexobjtest.items:
        print(key + " " + str(ngobj.docID) + " " + str(indexobjtest.items[key].posting[ngobj.docID].positions))


if __name__ == '__main__':
    # test()
    mini_newsgroups = str(sys.argv[1])
    feature_definition_file = str(sys.argv[2])
    class_definition_file = str(sys.argv[3])
    training_data_file = str(sys.argv[4])
    indexingNewsGroup(mini_newsgroups, feature_definition_file, class_definition_file, training_data_file)
    #indexingNewsGroup('mini_newsgroups', 'feature_definition_file', 'class_definition_file', 'training_data_file')
