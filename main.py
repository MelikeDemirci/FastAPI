# -*- coding: utf-8 -*-
"""
Created on 5/8/2022

@author: Melike Demirci
"""

# 1. Library imports
from typing import Optional
from typing import List
import uvicorn
from fastapi import FastAPI
from form import Form
import pickle
import re
import string
import json
import requests
import pandas as pd 

from pydantic import BaseModel

# ML
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

nltk.download('stopwords')
nltk.download('wordnet')

# 2. Create the app object
app = FastAPI()
svm_model = pickle.load(open("model.sav", 'rb'))
tfidf = pickle.load(open("tfidf.sav", 'rb'))

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

def getFormText(IDs):
	formText = []
	for formID in IDs:
		apikey = '7f8ce90b8d898a20bebcb12c3d8b52be'
		# Get form questions
		response = requests.get(f"https://api.jotform.com/form/{formID}/questions?apiKey={apikey}")
		responseObject = json.loads(response.content.decode('utf-8'))
		formQuestions = responseObject['content']
		# Get form properties for product text
		response = requests.get(f"https://api.jotform.com/form/{formID}/properties?apiKey={apikey}")    
		responseObject = json.loads(response.content.decode('utf-8'))
		formProperties = responseObject['content']

		text = ""
		for q in formQuestions:
			if "text" in formQuestions[q].keys():
				text += formQuestions[q]["text"].strip() + " "
			if "subHeader" in formQuestions[q].keys():
				text += formQuestions[q]["subHeader"].strip() + " "
			if "subLabel" in formQuestions[q].keys():
				text += formQuestions[q]["subLabel"].strip() + " "
			if "options" in formQuestions[q].keys():
				text += formQuestions[q]["options"].strip() + " "
		
		if "products" in formProperties.keys():
			typeVar = type(formProperties["products"])
			
			if typeVar == dict:
				for p in formProperties["products"]:
					if "name" in formProperties["products"][p].keys():
						text += formProperties["products"][p]["name"].strip() + " "
					if "description" in formProperties["products"][p].keys():
						text += formProperties["products"][p]["description"].strip() + " "
			elif typeVar == list:
				for p in formProperties["products"]:
					if "name" in p.keys():
						text += p["name"].strip() + " "
					if "description" in p.keys():
						text += p["description"].strip() + " "

		formText.append(text)
	return formText
#---------------------------------------------------------------------------------------------------------------
# Removing the html tags
def cleanhtml(text):
	CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
	cleantext = re.sub(CLEANR, '', text)
	return cleantext

# Removing the urls
def remove_urls(text):
    cleantext = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text)
    return(cleantext)

#Removing the punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).replace(' '*4, ' ').replace(' '*3, ' ').replace(' '*2, ' ').strip()

# Removing the stop words
def remove_stopwords(text):
	STOPWORDS = set(stopwords.words('english'))
	return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def remove_freqwords(text):
	FREQWORDS = ['name', 'please', 'e', 'mail', 'email','address','number', 'payment','submit', 'phone','date','form','may','us', 'card','example', 'com','yes' , 'no','one','full','like','page','would', 'per','must']
	return " ".join([word for word in str(text).split() if word not in FREQWORDS])

# Remove wors which include digits
def remove_wordswdigit(text):
    cleantext = re.sub("\S*\d\S*", "", text)
    return(cleantext)

# Lemmatize words
def lemmatize_words(text):
	lemmatizer = WordNetLemmatizer()
	return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

# Check the spelling mistakes
def correct_spellings(text):
	spell = SpellChecker()
	corrected_text = []
	misspelled_words = spell.unknown(text.split())
	for word in text.split():
		if word in misspelled_words:
			corrected_text.append(spell.correction(word))
		else:
			corrected_text.append(word)
	
	return " ".join(corrected_text)
#----------------------------------------------------------------------------------------------------------------------------

def preprocess(formText):
    proc_texts = []
    for text in formText:
        processed = text.lower()
        processed = cleanhtml(processed)
        processed = remove_urls(processed)
        processed = remove_punctuation(processed)
        processed = remove_stopwords(processed)
        processed = remove_freqwords(processed)
        processed = remove_wordswdigit(processed)
        processed = lemmatize_words(processed)
        processed = correct_spellings(processed)
        proc_texts.append(processed)
    return proc_texts

def predict(processedText):

	id_to_cluster = {0: 'Application Fee',
						1: 'Course Fee',
						2: 'Deposit Fee',
						3: 'Donation',
						4: 'Membership',
						5: 'Product Fee',
						6: 'Registration Fee',
						7: 'Service Fee',
						8: 'Subscription'}

	if len(processedText) == 1:
		data = {'text': processedText}
		df = pd.DataFrame(data)
		X = tfidf.transform(df["text"]).toarray()
		prediction = svm_model.predict(X)

		probs = svm_model.predict_proba(X)[0]
		return id_to_cluster[prediction[0]], probs.tolist()
	else:
		data = {'text': processedText}
		df = pd.DataFrame(data)
		X = tfidf.transform(df["text"]).toarray()
		predictions = svm_model.predict(X)

		result = []
		for p in predictions:
			result.append(id_to_cluster[p])	
		return result


@app.post('/predict')
def predict_cluster(data:Form):
	data = data.dict()
	formID=data['formID']
	formText = getFormText(formID)
	processedText = preprocess(formText)

	if len(processedText) == 1:
		prediction, probs = predict(processedText)
		print(prediction)
		print(probs)
		return {
            "formText": formText,
            "processedText": processedText,
            "prediction": prediction,
            "probabilities" : probs
        }
	else:
		prediction = predict(processedText)
		return {
            "formText": formText,
            "processedText": processedText,
            "prediction": prediction
        }


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
	uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload