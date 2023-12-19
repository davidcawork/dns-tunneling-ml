import time
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.naive_bayes import GaussianNB # GNB
from sklearn import model_selection # for model selection during training
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')

# Globas vars
input_dir='datasets/'

# Train data
training_data = pd.read_csv(input_dir+"training.csv")
print('\nHead of training data:\n')
print(training_data.head())

# Test data
test_data = pd.read_csv(input_dir+"validating.csv")
print('\nHead of test data:\n')
print(test_data.head())

# Preliminary analysis of the database to be used 
counts = training_data['Label'].value_counts()
print(counts)
counts.plot(kind = 'bar',color = ["royalblue","lightcoral"])
plt.title('Bar Plot on Training Data')
plt.show()

counts = test_data['Label'].value_counts()
print(counts)
counts.plot(kind = 'bar',color = ["royalblue","lightcoral"])
plt.title('Bar Plot on Test Data')
plt.show()

# We are going to use entropy to define each domain
#
# So.. we are going to use this function :)
def calculate_entropy(text):
    if not text: 
        return 0 
    
    entropy = 0

    # Each char has 256 possible values
    for x in range(256): 
        # Calc prob of that symbol
        p_x = float(text.count(chr(x)))/len(text) 
        if p_x > 0: 
            # shannon formula
            entropy += - p_x*math.log(p_x, 2) 
    return entropy

# Let's run entropy function on the train dataset for each query (domain)
entropy_train_vals = []

for query in training_data['Query']:
    entropy = calculate_entropy(query)
    entropy_train_vals.append(entropy)
    
training_data['Entropy'] = entropy_train_vals

# Let's do the same for test dataset
entropy_test_vals = []

for query in test_data['Query']:
    entropy = calculate_entropy(query)
    entropy_test_vals.append(entropy)

test_data['Entropy'] = entropy_test_vals


# We need to evaluate the models
def evaluate(predictions, targets):
    targets = targets.to_numpy()
    
    tp = 0; tn = 0; fp = 0; fn = 0;
    
    for t in range(targets.shape[0]):
        if targets[t] == -1:
            if predictions[t] == 1:
                fp += 1
            else:
                tn += 1
        else:
            if predictions[t] == 1:
                tp += 1
            else:
                fn += 1

    print("True Positives :", tp)
    print("True Negatives :", tn)
    print("False Positives :", fp)
    print("False Negatives :", fn)


# Pre-process input data
X_train = training_data['Entropy'] # the training input entropy
Y_train = training_data['Label']   # the corresponding classifying label for training
X_train, Y_train = shuffle(X_train, Y_train) # to reduce overfitting during training
X_train.ravel()
# We have to reshape the training features into unknown rows but 1 column
X_train = X_train.values.reshape(-1, 1)
Y_train = Y_train.values.reshape(-1, 1)

# We have to do the same with the test values 
X_test = test_data['Entropy']      # the test entropy for testing
Y_test = test_data['Label']       # the expected corresponding Label after training
X_test, Y_test = shuffle(X_test, Y_test)
X_test  = X_test.values.reshape(-1, 1)

#X_train, Y_train = shuffle(X_train, Y_train) # to reduce overfitting during training

