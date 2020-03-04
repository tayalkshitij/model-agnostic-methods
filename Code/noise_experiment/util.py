from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from numpy.random import permutation

def encodeLabel(Y):

	#Encode labels
	encoder = LabelEncoder()
	encoder.fit(Y)
	encodedLabels = encoder.transform(Y)
	encodedLabels = np_utils.to_categorical(encodedLabels)
	return encodedLabels,encoder

def encodeTraining(vectorcoloumn):

	temp = []
	for i in vectorcoloumn:
	    temp.append(i)
	X = np.array(temp)
	X = X.reshape((X.shape[0],X.shape[1]))
	return X

# def encode_onehot_LR(labels):
#     classes = set(labels)
#     classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
#     labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
#     return labels_onehot

def returnFeatureMatrix_LR(documents):
    vect = TfidfVectorizer(min_df=30,stop_words = 'english',max_features = 5000)
    tfidf_matrix = vect.fit_transform(documents)
    print (tfidf_matrix.shape)
    features = tfidf_matrix.toarray()
    return features

#     return tfidf_matrix 
    
    
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('x', '')
#    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text


def label_smoothing(y_train_cat, eps_LSR_noisy=0):
    """
    Label smoothing
    y_train_cat: Categorical labels in the training data
    num_classes: number of classes in the training set
    delta_eps_LSR: delta epsilon to add to / substract from epsilon, based on a prior. This defines the final epsilon
    to be applied to the active class label
    """
    num_classes = y_train_cat.shape[1]
    # standard LSR: all classes have the same epsilon for the ACTIVE class label
    
    y_train_cat = y_train_cat*(1-eps_LSR_noisy) + eps_LSR_noisy/num_classes

    return y_train_cat