import numpy as np
import pandas as pd
import simplejson as json
from collections import OrderedDict
from joblib import dump as jbdump
from joblib import load as jbload

import nltk
#nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
#import nltk

from nltk.corpus import stopwords

from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask


app = Flask(__name__)

results_path = "./"
    
# remove code parts of the question, remove other tags but keeps their contents
def clean_text_without_code(html) :
    soup = BeautifulSoup(html, 'lxml')
    [s.extract() for s in soup('code')]

    for match in soup.findAll(True):
        match.replaceWithChildren()
    return str(soup)

# extract code parts of the question, between <code> and </code> and keep them, without the tags <code> and </code>
def clean_code(html) :
    code = ''
    soup = BeautifulSoup(html,"lxml")
    for node in soup.findAll('code'):
        code = code.join(node.findAll(text=True))
   
    return code

def display_question(html) :
    soup = BeautifulSoup(html, 'lxml').get_text()

    return soup
    
nltk_stopwords = set(stopwords.words('english'))

def tokenize(text):
    # tokenize into words
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    # remove stopwords
    stop = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop]

    # remove words less than three letters
    tokens = [word for word in tokens if len(word) >= 3]

    # lower capitalization
    tokens = [word.lower() for word in tokens]

    # lemmatize
    lmtzr = WordNetLemmatizer()
    tokens = [lmtzr.lemmatize(word) for word in tokens]
    tokenized_text= ','.join(tokens)

    return tokenized_text 
	
class TextExtractor(BaseEstimator, TransformerMixin):
    """Concat the 'title', 'body' and 'code' from the results of 
    Stackoverflow query
    Keys are 'title', 'body' and 'code'.


    """

    def __init__(self, weight = {'title' : 1, 'body': 1, 'code' : 1}):

        self.weight = weight

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        
        x['subject_bow'] = x['Title'].apply(clean_text_without_code).apply(tokenize)
        x['body_bow'] = x['Body'].apply(clean_text_without_code).apply(tokenize)
        x['code_bow'] = x.Body.apply(clean_code).apply(tokenize)

        x['text'] = self.weight['title']*x['subject_bow'] + self.weight['body']*x['body_bow'] + self.weight['code']*x['code_bow']
        
        # convert to string, required by count vectorizer
        #replace nan values by "" because counvectorizer cannot handle nan values
        x['text'] = x['text'].apply(lambda x: np.str_(x) if pd.notnull(x) else '')
        
        return x['text']
		

		
@app.route('/')		
def hello_world() :
	return 'to get tags recommendation for the question with id "question_id", go to url "api_url/question_id/" . question_id should be an integer between 0 and 9999. '


@app.route('/<int:question_index>/')
def return_json_recommended_tags(question_index) :


    # load questions
    df = pd.read_csv('df_questions.csv', index_col='id')[0:9999].reset_index()  
    
    
    if (question_index in df.index.values) :
        
        question = df.loc[question_index]

        # load tags_dictionnary
        frequent_tags = np.load(results_path + 'frequent_500_tags.npy')


        #load classifier
        classifier = jbload(results_path + 'classifier_pipeline')

        #predict tags
        pred = classifier.predict_proba(pd.DataFrame([question[['Title', 'Body']]]))[0]
        recommended_tags = np.array(frequent_tags)[pred.argsort()[::-1][0:3]]

        question_body = display_question(question['Body'])


        results = []
        for tag in recommended_tags :
            dict_tag = {'tag' : tag }
            results.append(dict_tag)
        dict_results = {'recommended_tags': results }

    else :
        dict_results = {'error': str(question_index) + " is not a valid question id. Please retry with a question_id between 0 and 9999" }

    return json.dumps(OrderedDict(dict_results), sort_keys=True)
	
	
#if __name__ == '__main__':
#	app.run()	

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

	
