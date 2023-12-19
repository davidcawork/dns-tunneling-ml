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
    for x in range(256): 
        p_x = float(text.count(chr(x)))/len(text) 
        if p_x > 0: 
            entropy += - p_x*math.log(p_x, 2) 
    return entropy


