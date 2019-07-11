
'''
Irony detection in English Tweets
'''
import itertools
import csv
import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import feature_extraction



#to print out the entire matrix after vectorising it
#np.set_printoptions(threshold=np.nan)


#reading the training data
df = pd.read_csv("classificationOne/preprocessed_trainingdata.csv", encoding='utf-8')

#reading the test data
df1 = pd.read_csv("classificationOne/preprocessed_testdata.csv", encoding='utf-8')



#separating the target class from the data (from csv file)
df_x = df["Tweet Text"]
df_y = df["index Label"]

df_x1 = df1["Tweet Text"]
df_y1 = df1["index Label"]



#VECTORISER USED
cv = CountVectorizer()
#cv = TfidfVectorizer(min_df=0, stop_words='english', ngram_range=(1, 2))    #remove the stop words from the english language

#to split the data into test and training
#x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)


x_traincv = cv.fit_transform(df_x.values.astype(str))
#x_traincv = cv.fit_transform(x_train.values.astype(str))
#array = x_traincv.toarray()
print(x_traincv)


#mnb = MultinomialNB()
#model = LinearSVC()
model = tree.DecisionTreeClassifier()
#model = LogisticRegression()
#y_train = y_train.astype('int')    #converting type to int of training data of target class
#print(df_y)
#creating the classifier
print(model.fit(x_traincv, df_y))



#extract tfid features to test data too

x_testcv = cv.transform(df_x1.values.astype(str))
#for checking predictions
predictions = model.predict(x_testcv)

#single ouput
basic_test=["This is just a long sentence, to make sure that it's not how long the sentence is that matters the most",\
            'I just love when you make me feel like shit','Life is odd','Just got back to the US !', \
            "Isn'it great when your girlfriend dumps you ?", "I love my job !", 'I love my son !']
feature_basictest=[]
for tweet in basic_test:
    feature_basictest.append(feature_extraction.getallfeatureset(tweet))
feature_basictest=np.array(feature_basictest)
feature_basictestvec = vector.transform(feature_basictest)

print(basic_test)
print(classifier.predict(feature_basictestvec))




#to look at how many predictions were right
actual_results= np.array(df_y1)
print(actual_results)


count = 0
for i in range(len(predictions)):
    if predictions[i] == actual_results[i]:
        count = count +1

#count is the number of correct predictions
a = count
print(count)

#predictions is total number of predictions
b = len(predictions)
print(len(predictions))

accuracy = a/b
#we can also use the built in library method to print accuracy
print("the accuracy is:")
print(accuracy_score(df_y1, predictions))

#print("The calculated accuracy is :",accuracy)

#calcualting the F1 score - check out the various options with many other average paramters
#score = metrics.f1_score(df_y1, predictions, average="macro")
#print("the f1 score is: ", score)
target_names = ['Ironic', 'Non ironic']
print(classification_report(df_y1, predictions, target_names=target_names))

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
#
# #compute confusion matrix
# cnf_matrix=confusion_matrix(df_y1, predictions)
# np.set_printoptions(precision=2)
#
# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=target_names,
#                       title='Confusion matrix, without normalization')
#
# # Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
#                       title='Normalized confusion matrix')
#
# plt.show()







