import pandas as pd
import re
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import *
from sklearn.metrics import f1_score

input_set = pd.read_csv('input/converted.csv', sep='\t')
input_set.rename(columns={'Oceniona przez nas' : 'label'},inplace=True)
training_set = input_set
print(training_set.columns)
#print(training_set['Text'])
print('Training set:')
print(training_set)

def clean_set(df, field):
    df[field] = df[field].str.lower()
    #removal of @anonimized_account
    df[field] = df[field].apply(lambda elem: re.sub(r'@anonymized_account', '', elem))
    #removal of url links
    df[field] = df[field].apply(lambda elem: re.sub('https?://[A-Za-z0-9./]+','',elem))
    #removal of excessive whitespace,tabs etc.
    df[field] = df[field].apply(lambda elem: re.sub('\s+', ' ', elem))

clean_set(training_set, 'Text')
print("Cleaned set:")
print(training_set)

test_set = input_set[:100].copy()
input_set = input_set[100:].copy()

train_majority = input_set[input_set.label == 0]
train_minority = input_set[input_set.label == 1]

#train_downsampled = input_set
train_majority_downsampled = resample(train_majority, replace=True, n_samples = len(train_minority), random_state = 455)
train_downsampled = pd.concat([train_majority_downsampled, train_minority])

pipeline_sgd = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('nb', SGDClassifier()),
])

X_train, X_test, y_train, y_test = train_test_split(train_downsampled['Text'], train_downsampled['label'], random_state=0)

model = pipeline_sgd.fit(X_train, y_train)
y_predict = model.predict(X_test)

print(f1_score(y_test, y_predict))
