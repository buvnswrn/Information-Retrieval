from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from math import log
materials = "materials"
patterns = materials+"/patterns.txt"
questions = materials+"/test_questions.txt"
corpus_xml = materials+"/trec_documents.xml"
corpus = dict()
idf = dict()
tf = dict()
corpus_string = '' 
tokenizer = RegexpTokenizer(r'\w+')
with open(corpus_xml) as infile:
    soup = BeautifulSoup(infile,"lxml")
    
all_doc = soup.find_all('doc')
for docs in all_doc:
    doc_no=docs.find("docno").contents[0]
    text = docs.find("text")
    p = text.find_all("p")
    content = []
    content_string = ''
    if not p:
        content_string = text.contents[0]
    else:
        for text in p:
            content_string+=text.contents[0]
    corpus[doc_no.strip()] = tokenizer.tokenize(content_string)
    corpus_string += content_string

# print(corpus['LA010189-0120'])
vocab = set(tokenizer.tokenize(corpus_string))
doc_freq = dict()
for term in vocab:
    doc_freq[term] = 0

N = len(corpus.keys())
term_freq = dict()
for doc in corpus.keys():
    content = corpus[doc]
    content_counter = Counter(content)
    most_common = content_counter.most_common()
    term_freq[doc] = dict(most_common)
    term_set = set(term_freq[doc].keys())
    max_freq = most_common[0][1]
    for term in term_set:
        doc_freq[term] +=1
        tf[(term,doc)] = term_freq[doc][term]/max_freq

for term in vocab:
    idf[term] = log(N/doc_freq[term])



# print(idf)
print(tf)