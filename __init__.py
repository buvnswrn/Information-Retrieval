#In[1]:
from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
from math import log
from numba import jit
import re
materials = "materials"
patterns_file = materials+"/patterns.txt"
questions = materials+"/test_questions.txt"
corpus_xml = materials+"/trec_documents.xml"
corpus = dict()
doc_ids = list()
idf = dict()
tf = dict()
q_tf = dict()
tf_idf = dict()
q_tf_idf = dict()
cosine = dict()
queries = dict()
max_freq = dict()
retrieved_docs = dict()
unprocessed_corpus = dict()
relavant_docs = dict()
corpus_string = '' 
tokenizer = RegexpTokenizer(r'\w+')
stemmer = PorterStemmer()
stop_words = stopwords.words("english")
# Reading file and converting to soup
with open(corpus_xml) as infile:
    soup = BeautifulSoup(infile,"lxml")
with open(questions) as qfile:
    q_soup = BeautifulSoup(qfile,"lxml")
all_doc = soup.find_all('doc')
all_query = q_soup.find_all('top')

# Extracting query and tokenizing
for query in all_query:
    desc = query.find("desc").contents[0].split('\n')[1]
    num = query.find("num").contents[0].split()[1]
    queries[int(num)] = [stemmer.stem(word) for word in tokenizer.tokenize(desc) if word not in stop_words]
# print(queries)

#In[2]:
# Extracting document text, preprocessing
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
    unprocessed_corpus[doc_no.strip()] = content_string
    corpus[doc_no.strip()] = [stemmer.stem(word) for word in tokenizer.tokenize(content_string) if word not in stop_words]
    doc_ids.append(doc_no.strip())
    corpus_string += content_string

# Setting up vocabulary
# print(corpus['LA010189-0120'])
vocab = set([stemmer.stem(word) for word in tokenizer.tokenize(corpus_string) if word not in stop_words])
doc_freq = dict()
for term in vocab:
    doc_freq[term] = 0

# Computing term frequency (tf) and document frequency
N = len(corpus.keys())
term_freq = dict()
for doc in corpus.keys():
    content = corpus[doc]
    content_counter = Counter(content)
    most_common = content_counter.most_common()
    term_freq[doc] = dict(most_common)
    term_set = set(term_freq[doc].keys())
    max_freq[doc] = most_common[0][1]
    for term in term_set:
    # for term in content:
        doc_freq[term] +=1
        tf[(term,doc)] = term_freq[doc][term]/max_freq[doc]
    # for term in queries.keys():
    #     pass
for q_no in queries.keys():
    q_content = queries[q_no]
    q_content_counter = Counter(q_content)
    q_most_common = q_content_counter.most_common()
    q_term_freq = dict(q_most_common)
    q_term_set = set(q_term_freq.keys())
    q_max_freq = q_most_common[0][1]
    for term in q_term_set:
        if(term in vocab):
            q_tf[(term,q_no)] = 0.5+((0.5*q_term_freq[term])/q_max_freq)
    
# print(len(doc_ids))
# print(tf)
# Computing Inverse document frequency (idf)
for term in vocab:
    idf[term] = log(N/doc_freq[term])

# Computing tf-idf for document terms
for term_doc in tf.keys():
    tf_idf[term_doc] = tf[term_doc]*idf[term_doc[0]]

# Computing tf-idf for query terms
for term_doc in q_tf.keys():
    q_tf_idf[term_doc] = q_tf[term_doc]*idf[term_doc[0]]

print(q_tf_idf)
# for q_no in queries.keys():
#     set_a = set(queries[q_no])
#     l1= list()
#     for d_id in doc_ids:
#         set_b = set(term_freq[d_id].keys())
#         l2 = list()
#         r_vector = set_a.union(set_b) 
#         for word in r_vector:
#             if word in set_a: l1.append(1)
#             else: l1.append(0)
#             if word in set_b: l2.append(1)
#             else: l2.append(0)
#         c = 0
#         for i in range(len(r_vector)):
#             c+=l1[i]*l2[i]
#         cosine[(q_no,d_id)] = c / float((sum(l1)*sum(l2))**0.5)

# print(cosine)



# print(idf)
# print(tf_idf)

# In[3]:
# print(tf_idf)
# print(q_tf_idf)
# In[4]:
# Calculating cosine values
def cosine_similarity(queries,doc_ids,vocab,term_freq,q_tf_idf,tf_idf):
    cosine = dict()
    for q_no in queries.keys():
        terms = set(queries[q_no])
        for doc_id in doc_ids:
            doc_terms = set(term_freq[doc_id].keys())
            doc_query_vector = terms.union(doc_terms)
            doc_vector = list()
            query_vector = list()
            for word in doc_query_vector:
                if (word in terms) and (word in vocab):
                    query_vector.append(q_tf_idf[(word,q_no)])
                    # query_vector.append(1)
                else:
                    query_vector.append(0)
                if word in doc_terms:
                    doc_vector.append(tf_idf[(word,doc_id)])
                    # doc_vector.append(1)
                else:
                    doc_vector.append(0)
            c = 0
            for i in range(len(doc_query_vector)):
                c+=doc_vector[i]*query_vector[i]
            cosine[(q_no,doc_id)] = c/float((sum(doc_vector)*sum(query_vector))**0.5)
        return cosine
cosine = cosine_similarity(queries,doc_ids,vocab,term_freq,q_tf_idf,tf_idf)

# In[5]:
# sorting similarity scores and fetching top 50
def sort_scores(queries,doc_ids,cosine):
    retrieved_docs = dict()
    for q_no in queries.keys():
        retrieved_docs[q_no]=(sorted([(doc_id,cosine[(q_no,doc_id)]) 
                            for doc_id in doc_ids],key=lambda x:x[1])[::-1])[:50]
    return retrieved_docs

retrieved_docs = sort_scores(queries,doc_ids,cosine)

# In[6]:
def get_docs_from_pattern(unprocessed_corpus):
    relevant_docs = dict()
    patterns = dict()
    with open(materials+'/patterns.txt','r') as patfile:
        for cnt,line in enumerate(patfile):
            line = line.split()
        # while line:
            pattern = ''
            q_no = line[0].strip()
            if len(line)>2:
                pattern = ' '.join(line[1:])
            else:
                pattern = line[1]
            if(q_no in patterns):
                patterns[q_no] = patterns[q_no]+"|"+pattern
            else:
                patterns[q_no] = pattern
            line = patfile.readline
    for q_no in patterns.keys():
        regex = re.compile("("+patterns[q_no]+")")
        for doc_id in unprocessed_corpus.keys():
            string = unprocessed_corpus[doc_id]
            if(regex.search(string)!=None):
                if q_no in relevant_docs:
                    relevant_docs[q_no].append(doc_id)
                else:
                    relevant_docs[q_no] = [doc_ids]
    return relevant_docs

relavant_docs = get_docs_from_pattern(unprocessed_corpus)

    

# %%
# print(relavant_docs)
# print(["{0}:{1}".format(q_no,len(relavant_docs[q_no])) for q_no in relavant_docs.keys()])


# %%
# print(len(relavant_docs))

# In[7]:

# %%
