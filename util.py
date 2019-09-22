"""
   utility functions for processing terms

    shared by both indexing and query processing
"""

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re

def tokenize(text):
    text=re.sub("[^a-zA-Z0-9]+"," ",text)
    tokens = word_tokenize(text)
    # alpha_tokens = [token for token in tokens if token]
    # print(alpha_tokens)
    remaining_tokens = [
        # (3) stemming
        stemming(token.lower())
        # (1) convert to lower cases
        for token in tokens if
        (not isStopWord(token) and token.isalpha())]

    for token in tokens:
        if not token.isalpha() and token.isalnum():
            remaining_tokens.append(token)
    #print(remaining_tokens)
    return remaining_tokens


def isStopWord(word):
    stop_words = [newline.strip() for newline in open("stopwords", "r")]
    #print(word)
    if word.lower() in stop_words:
        #print(True)
        return True
    else:
        #print(False)
        return False

""" using the NLTK functions, return true/false"""


def stemming(word):
    stemmer_word = PorterStemmer().stem(word)
    return stemmer_word


""" return the stem, using a NLTK stemmer. check the project description for installing and using it"""

if __name__ == '__main__':
    ''' testing '''
    print(tokenize('This is in a project i'))
    print(isStopWord('i'))
