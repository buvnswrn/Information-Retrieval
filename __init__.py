#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import xml.etree.ElementTree as ElementTree
from nltk.tokenize import RegexpTokenizer
import string as STR
import sys
from collections import Counter
import math
from tqdm import tqdm
from lxml import etree


# In[23]:


def retrieval():
    score = {}
    magnitude_doc = {}
    for query in tqdm(queries.keys()):
        term_freq = queries[query]
        tfidf_query = {}
        for term in term_freq.keys():
            tfidf_query[term] = term_freq[term] * idf.get(term, 0)
        temp = {}
        magnitude_query = math.sqrt(sum([math.pow(tfidf_query[term], 2) for term in tfidf_query.keys()]))
        for doc in corpus.keys():
            similarity = 0
            for term in tfidf_query.keys():
                tfidf_term_doc = term_frequency.get(term, {}).get(doc, 0) * idf.get(term, 0)
                similarity += tfidf_term_doc * tfidf_query[term]
            if doc not in magnitude_doc.keys():
                magnitude = 0
                for term in corpus[doc].keys():
                    magnitude += math.pow(term_frequency[term][doc] * idf[term], 2)
                magnitude_doc[doc] = math.sqrt(magnitude)

            temp[doc] = similarity / (magnitude_doc[doc] * magnitude_query)
        score[query] = temp
    return score


# In[3]:


with open('trec_documents.xml', 'r') as f:  # Reading file
    xml = f.read()
xml = '<ROOT>' + xml + '</ROOT>'
root = ElementTree.fromstring(xml)
corpus = {}

for doc in root:
    if not doc.find('DOCNO').text.strip().startswith("LA"):
        corpus[doc.find('DOCNO').text.strip()] = doc.find('TEXT').text.strip()
    else:
        text = ""
        for child in doc.find('TEXT'):
            if child.tag == 'P':
                string = child.text.strip()
                text = ' '.join([text, string])

        corpus[doc.find('DOCNO').text.strip()] = text.strip()

# In[4]:


tokenizer = RegexpTokenizer(r'\w+')
table = str.maketrans('', '', '!\"#$%&\'()*+,./:;<=>?@[\]^_`{|}~')
corpus_combined = []
for doc in tqdm(corpus.keys()):
    corpus[doc] = corpus[doc].lower().translate(table)
    corpus[doc] = tokenizer.tokenize(corpus[doc])
    corpus_combined += corpus[doc]
    temp = Counter(corpus[doc])
    corpus[doc] = {key: temp[key] / temp.most_common(1)[0][1] for key in temp}

# In[5]:


vocab = set(corpus_combined)
Total_docs = len(corpus.keys())
idf = {}
term_frequency = {}
tfidf_doc = {}
for term in tqdm(vocab):
    count_term = 0
    temp = {}
    for doc in corpus.keys():
        if term in corpus[doc]:
            count_term += 1
        temp[doc] = corpus[doc].get(term, 0)
    idf[term] = math.log(Total_docs / count_term)
    term_frequency[term] = temp

# In[6]:


queries = {}
with open('test_questions.txt', 'r') as f:
    file = f.read()

parser = etree.HTMLParser()
root = etree.fromstring(file, parser)

num = 1
for num, query in zip(list(range(1, 101)), root.find('body')):
    text = ' '.join(query.find('num').find('desc').text.strip().split()[1:])
    text = text.lower().translate(table)
    text = tokenizer.tokenize(text)
    temp = Counter(text)
    queries[num] = {key: temp[key] / temp.most_common(1)[0][1] for key in temp}

# In[26]:


score = retrieval()

# In[28]:


print(score[100])

# In[ ]:




