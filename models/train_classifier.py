import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, make_scorer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
import time
import pickle

#nltk.download()
nltk.download('wordnet')
nltk.download('stopwords')


#    a text parse reference from this site:
#    https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
class BasicTextAnalytics(BaseEstimator, TransformerMixin):
    '''
    Class for returning some basic numerical data for text analysis to include in 
    modelling. Such as: 
    - Number of sentences
    - Number of words
    - Number of nouns
    - Number of verbs
    - Number of adjectives
    '''
    pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
    }

    # function to check and get the part of speech tag count of a words in a given sentence
    def check_pos_tag(self, text, flag):
        '''
        Returns the count of a given NL pos_tag, based on user selection. E.g. number of nouns.
        INPUTS
        text - the given text to analyse
        flag - pos family to analyse, one of 'noun', 'pron' , 'verb', 'adj' or 'adv'
        '''
        count = 0
        try:
            wiki = textblob.TextBlob(text)
            for tup in wiki.tags:
                ppo = list(tup)[1]
                if ppo in pos_family[flag]:
                    count += 1
        except:
            pass
        return count
    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        trainDF = pd.DataFrame()
        trainDF['text'] = X
        trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))
        trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
        trainDF['noun_count'] = trainDF['text'].apply(lambda x: self.check_pos_tag(x, 'noun'))
        trainDF['verb_count'] = trainDF['text'].apply(lambda x: self.check_pos_tag(x, 'verb'))
        trainDF['adj_count'] = trainDF['text'].apply(lambda x: self.check_pos_tag(x, 'adj'))
        trainDF['adv_count'] = trainDF['text'].apply(lambda x: self.check_pos_tag(x, 'adv'))
        trainDF['pron_count'] = trainDF['text'].apply(lambda x: self.check_pos_tag(x, 'pron'))
        
        return trainDF.drop('text',axis=1)



# 'sqlite:///cleanTable.db'
def load_data(database_filepath):
    '''
    Imports the "cleanTable" table from a specified database file
    Returns X and Y datasets as pandas DataFrames as well as a list of column names
    '''
    # load data from database
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table("cleanTable", con=engine)
    X = df[["message", "genre"]]
    Y = df.drop(axis=1, labels=["id", "message", "original", "genre"])
    return X, Y, df

def tokenize(text):
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for token in tokens:
        clean_tok = lemmatizer.lemmatize(token).lower().strip()
        
        # remove stop words 
        if clean_tok not in set(stopwords.words("english")):
            clean_tokens.append(clean_tok) 
    return clean_tokens


def build_model():
    ''' 
    Build preprocessing pipelines for both numeric and text data, then completes a quick
    Add grid search over RandomForestClassifier key parameters. 
    '''
    pipeline_model = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize, 
                                        ngram_range=(1, 2),
                                        max_features=5000,
                                        max_df=0.5)),
                ('tfidf', TfidfTransformer())
            ])),

            ('numerical_pipeline', Pipeline([
                ('analytics', BasicTextAnalytics()),
                ('norm', StandardScaler())
                ]))
        ])),

        ('clf', MultiOutputClassifier(LogisticRegression()))
    ])
    # specify parameters for grid search
    parameters = {
        "features__text_pipeline__vect__max_features" : [None],
        "clf__estimator__C" : [10]
    }

    # create grid search object
    cv = GridSearchCV(pipeline_model, parameters, n_jobs=-1, cv=2)

    return cv


def evaluate_model(model, X_test, Y_test):
    '''
    Evalutes the model using sklearns 'classification report' function to return precision and recall for each class
    Inputs: model - model should be fitted by data, and can be used to predict
            X_test - test set data [pandas dataframe/series]
            Y_test - correct categories for the test data [pandas dataframe/series]
    '''
    Y_test_pred = model.predict(X_test)
    
    for index, feature in enumerate(Y_test.columns):
        print(feature)
        print(classification_report(Y_test[feature], Y_test_pred[:, index]))
        print('f1 score: {}'.format(f1_score(Y_test[feature], Y_test_pred[:, index], average='weighted')))
    return


# classifier.pkl
def save_model(model, model_filepath):
    '''
    Saves the model 
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X['message'], Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()