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

# Lets play jeje
seed = 7
models = []
models.append(('DTC', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('KNN', KNeighborsClassifier(n_neighbors = 2)))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
dfs = []
target_names = ['fail', 'passed']

for name, model in models:
    kfold = model_selection.KFold(n_splits=200, shuffle=True, random_state=seed)
    cv_results = model_selection.cross_validate(model, X_train, Y_train, cv=kfold, scoring=scoring)
    clf = model.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    
    print(name)
    print(classification_report(Y_test, y_pred, target_names=target_names))
    
    results.append(cv_results)
    names.append(name)

    this_df = pd.DataFrame(cv_results)
    this_df['model'] = name
    dfs.append(this_df)
    final = pd.concat(dfs, ignore_index=True)

model_list = list(set(final.model.values))
model_list.sort()
bootstraps = []
for model in model_list:
    model_df = final.loc[final.model == model]
    bootstrap = model_df.sample(n=30, replace=True)
    bootstraps.append(bootstrap)
        
bootstrap_df = pd.concat(bootstraps, ignore_index=True)
results_long = pd.melt(bootstrap_df,id_vars=['model'],var_name='metrics', value_name='accuracy')
time_metrics = ['fit_time','score_time'] # fit time metrics
results_long_nofit = results_long.loc[~results_long['metrics'].isin(time_metrics)] # get df without fit data
results_long_nofit = results_long_nofit.sort_values(by='accuracy')
results_long_fit = results_long.loc[results_long['metrics'].isin(time_metrics)] # df with fit data
results_long_fit = results_long_fit.sort_values(by='accuracy')
plt.figure(figsize=(30, 20))
sns.set(font_scale=1.5)
g = sns.boxplot(x="model", y="accuracy", data=results_long_nofit)
plt.title('Comparison of Model by Classification Metric')
plt.savefig('./benchmark_models_performance.png',dpi=300)
plt.show()


# Lets go one by one :)

# DTC
X_train, Y_train = shuffle(X_train, Y_train) # to reduce overfitting during training
start = time.time()
model = DecisionTreeClassifier(max_depth=4)   
model.fit(X_train, Y_train.ravel())
end = time.time()
dtc_time = (end-start)*1000
print("TIME consu: ", dtc_time,"millisec")