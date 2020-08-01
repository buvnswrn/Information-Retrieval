from bs4 import BeautifulSoup
from nltk.tokenize import RegexpTokenizer
materials = "materials"
patterns = materials+"/patterns.txt"
questions = materials+"/test_questions.txt"
corpus_xml = materials+"/trec_documents.xml"
corpus = dict()
tokenizer = RegexpTokenizer(r'\w+')
with open(corpus_xml) as infile:
    soup = BeautifulSoup(infile,"lxml")
    
all_doc = soup.find_all('doc')
for docs in all_doc:
    doc_no=docs.find("docno").contents[0]
    text = docs.find("text")
    p = text.find_all("p")
    content = []
    if not p:
        content = tokenizer.tokenize(text.contents[0])
    else:
        content_string = ''
        for text in p:
            content_string+=text.contents[0]
        content = tokenizer.tokenize(content_string)
    corpus[doc_no.strip()] = content

print(corpus['LA010189-0120'])
