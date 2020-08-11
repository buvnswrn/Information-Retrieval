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
from bs4 import BeautifulSoup
import re


# In[2]:


def retrieval():
    score = {}
    magnitude_doc = {}
    for query in tqdm(queries.keys()):
        term_freq_query = queries[query]
        tfidf_query = {}
        for term in term_freq_query.keys():
            tfidf_query[term] = term_freq_query[term]*idf.get(term,0)
        temp = {}
        magnitude_query = math.sqrt(sum([math.pow(tfidf_query[term],2) for term in tfidf_query.keys() ])) 
        for doc in corpus.keys():
            similarity = 0
            for term in tfidf_query.keys():
                tfidf_term_doc = term_frequency.get(doc,{}).get(term,0)*idf.get(term,0)
                similarity += tfidf_term_doc * tfidf_query[term]
            if doc not in magnitude_doc.keys():
                magnitude = 0
                for term in term_frequency[doc].keys():
                    magnitude += math.pow(term_frequency[doc][term]*idf[term],2)
                magnitude_doc[doc] = math.sqrt(magnitude)
           
            temp[doc] = similarity/(magnitude_doc[doc]*magnitude_query)
        score[query] = temp
    return score

def precision():
    precision_50 = {}
    for query in tqdm(score_50.keys()):
        num_relevant = 0
        for doc,_ in score_50[query]:
            if any (re.search(regex,corpus_raw[doc],re.IGNORECASE) for regex in query_answer[query]):
                num_relevant+=1
        #print('number of relevant docs for query {} are {}'.format(query,num_relevant))
        precision_50[query] = num_relevant/50
    return precision_50
            
            
            
            
            
                
                
                
            
            


# In[3]:


with open('trec_documents.xml', 'r') as f:  # Reading file
    xml = f.read()
xml = '<ROOT>' + xml + '</ROOT>'
root = BeautifulSoup(xml, 'lxml-xml')
corpus = {}

for doc in root.find_all('DOC'):
    if not doc.find('DOCNO').text.strip().startswith("LA"):
        corpus[doc.find('DOCNO').text.strip()] = doc.find('TEXT').text.strip()
    else:
        text = ""
        for child in  doc.find('TEXT').findChildren("P" , recursive=False):
          
            string = child.text.strip()
            text = ' '.join([text, string])

        corpus[doc.find('DOCNO').text.strip()] = text.strip()

corpus_raw = corpus.copy()


# In[4]:


tokenizer = RegexpTokenizer(r'\w+')
table = str.maketrans('','','!\"#$%&\'()*+,./:;<=>?@[\]^_`{|}~')
corpus_combined = []
term_frequency = {}
for doc in tqdm(corpus.keys()):
    corpus[doc] = corpus[doc].lower().translate(table)
    corpus[doc] = tokenizer.tokenize(corpus[doc])
    corpus_combined+= corpus[doc]
    #corpus_raw[doc] = ' '.join(corpus[doc])
    temp = Counter(corpus[doc])
    term_frequency[doc] = {key:temp[key]/temp.most_common(1)[0][1] for key in temp} 
    



# In[5]:


vocab = set(corpus_combined)
Total_docs = len(corpus.keys())
idf = {}
#term_frequency = {}
for term in tqdm(vocab):
    count_term = 0
    
    for doc in corpus.keys():
        if term in term_frequency[doc]:
            count_term+=1
        #temp[doc] = corpus[doc].get(term,0)
    idf[term] = math.log(Total_docs/count_term)
    #term_frequency[term] = temp


# In[6]:


queries = {}
with open('test_questions.txt', 'r') as f:  
    file = f.read()


root = BeautifulSoup(file, 'lxml')
for num, query in zip(list(range(1,101)),root.body.find_all('top')):
    text =  ' '.join(query.num.desc.text.strip().split()[1:])
    text = text.lower().translate(table)
    text = tokenizer.tokenize(text)
    temp = Counter(text)
    queries[num] = {key:temp[key]/temp.most_common(1)[0][1] for key in temp} 
   


   
   
    
        
  


# In[7]:


score = retrieval()
score_50 = {}
score_1000 = {}
for key in score.keys():
    score[key]  =sorted(score[key].items(), key=lambda x: x[1], reverse=True)
    score_50[key] = score[key][:50]
    score_1000[key] = score[key][:1000]


# In[8]:


with open('patterns.txt', 'r') as f:  
        file = f.read()
#file = file.lower()
file = file.split('\n')

query_answer = {}

for line in file:
    if query_answer.get(int(line.split()[0])) == None:
        query_answer[int(line.split()[0])] = [' '.join(line.split()[1:])]
    else:
         query_answer[int(line.split()[0])].append(' '.join(line.split()[1:]))

    

    


# In[9]:


precision_50 = precision()
print(precision_50)
print('Mean Avg Precision is {}'.format(sum(list(precision_50.values()))/len(list(precision_50.values()))))


# In[10]:


print(len([value for value in list(precision_50.values()) if value !=0]))


# In[ ]:




