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

N = len(corpus.keys())
term_freq = dict()
for doc in corpus.keys():
    content = corpus[doc]
    term_freq[doc] = dict(Counter(content).most_common())

vocab = set(tokenizer.tokenize(corpus_string))
print(vocab)

for term in vocab:
    term_count = 0
    for doc_id in corpus.keys():
        if term in corpus[doc_id]:
            term_count+=1
    idf[term] = log(N/term_count)


print(idf)