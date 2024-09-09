from nltk import ne_chunk

import nltk

from nltk.tokenize import word_tokenize



from nltk.corpus import wordnet 

nltk.download('wordnet')

import pandas as pd

#list of positive words
positive_words=['amazing','beautiful','happy','gorgeous','pleasent','good','excellent','nice']

#list of negative words
negative_words=['bad','annoying','sad','sorrow','not','harmful','fail','evil']

synonyms=[]
for syn in wordnet.synsets("sad"):
    for i in syn.lemmas():
        synonyms.append(i.name)
print(set(synonyms))

antonyms=[]
for syn in wordnet.synsets("love"):
    for i in syn.lemmas():
        if i.antonyms():
            antonyms.append(i.antonyms()[0].name())
print(set(antonyms))

#SAMPLE TEXT FOR ANALYSIS
text="The movie was good and made me happy,but the ending was sad."

#Tokenize the text
nltk.download('punkt_tab')
tokens=word_tokenize(text)

#initialize positive and negative word counts 
positive_count=0
negative_count=0

#count positive and negative words
for word in tokens:
    if word in positive_words:
        positive_count+=1
    elif word in negative_words:
        negative_count+=1

#create a dictionary for bag of words
bag_of_words={'positive':positive_count,'Negative':negative_count}

#convert dictionary to df for visualization 
df=pd.DataFrame(list(bag_of_words.items()),columns=['sentiment','count'])

#display the bag of words
print(df)

import spacy
nlp=spacy.load('en_core_web_sm')
