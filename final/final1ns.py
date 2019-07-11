import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import feature_extraction
import pickle
import scipy as sp
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction import DictVectorizer
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_extraction import DictVectorizer
import pickle
import feature_extraction
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import pandas as pd

print('Pickling out')
# ironicData = np.load('ironic_semeval.npy')
# nonIronicData = np.load('nonironic_semeval.npy')
ironicData=[]
nonIronicData=[]
df=pd.read_json('csvjson.json')
#print(df)
for a,b in df.iterrows():
    if(b["label"]==1):
       ironicData.append(b['text'])
    elif(b["label"]==-1):
        nonIronicData.append(b['text'])
print('Number of ironic tweets :', len(ironicData))
print('Number of non-ironic tweets :', len(nonIronicData))

print('Feature engineering')
classificationSet = ['Ironic', 'Non-Ironic'] #label set
featureSets = []

index=0
for tweet in ironicData:
    if (np.mod(index, 10000) == 0):
        print("Processed Ironic Tweets: ", index)
    featureSets.append((feature_extraction.getallfeatureset(tweet), classificationSet[0]))
    index+=1

index = 0
for tweet in nonIronicData:
    if (np.mod(index, 10000) == 0):
        print("Processed Non-Ironic Tweets: ", index)
    featureSets.append((feature_extraction.getallfeatureset(tweet), classificationSet[1]))
    index+=1
featureSets=np.array(featureSets)
targets=(featureSets[0::,1]=='Ironic').astype(int)

#Transforms lists of feature-value mappings to vectors
vector = DictVectorizer()
featureVector = vector.fit_transform(featureSets[0::,0])
a=['svm','DT']
for x in a:
    if(x=='svm'):


        #Saving the dictionary vectorizer
        fName = "dictionaryFileSVM.p"
        fObject = open(fName, 'wb')
        pickle.dump(vector, fObject)
        fObject.close()

        #Feature splitting
        print('Feaature Splitting')
        order = shuffle(list(range(len(featureSets))))
        targets=targets[order]
        featureVector=featureVector[order, 0::]

        #Splitting data set in training and test set
        size = int(len(featureSets) * .3)

        trainVector = featureVector[size:,0::]
        trainTargets = targets[size:]
        testVector = featureVector[:size, 0::]
        testTargets = targets[:size]

        print('Training')

        #Artificial weights
        ironicP=(trainTargets==1)
        nonIronicP = (trainTargets==0)
        ratio = np.sum(nonIronicP.astype(float))/np.sum(ironicP.astype(float))

        newTrainVector = trainVector
        newTrainTargets=trainTargets

        for j in range(int(ratio-1.0)):
            newTrainVector=sp.sparse.vstack([newTrainVector,trainVector[ironicP,0::]]) #Stack sparse matrices vertically
            newTrainTargets=np.concatenate((newTrainTargets,trainTargets[ironicP]))


        classifier = SVC(C=0.1, kernel='linear')
        #estimate the accuracy of a linear kernel support vector machine on the iris dataset
        # by splitting the data, fitting a model and computing the score 5 consecutive times (with different splits each time)
        scores = cross_val_score(classifier, newTrainVector, newTrainTargets, cv=5)
        #	Fit the SVM model according to the given training data
        classifier.fit(newTrainVector, newTrainTargets)


        #Saving the classifier
        fName = "classifierSVM.p"
        fObject=open(fName,'wb')
        pickle.dump(classifier,fObject)
        fObject.close()


        #Validation
        print('Validating')
        class_names = ['ironic', 'regular']
        #Perform classification on samples in X.
        output = classifier.predict(testVector)
        classificationReport = classification_report(testTargets,output,target_names=classificationSet)
        print(classificationReport)
        print(accuracy_score(testTargets, output)*100)
        svm=accuracy_score(testTargets, output)*100
        def plot_confusion_matrix(cm, classes,
                                normalize=False,
                                title='Confusion matrix',
                                cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                #print("Normalized confusion matrix")
            else:
                #print('Confusion matrix, without normalization')
                print()

            #print(cm)

            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')


        #compute confusion matrix
        cnf_matrix=confusion_matrix(testTargets, output)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                            title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                            title='Normalized confusion matrix')

        plt.show()

        basic_test=["I have been on rather too many planes and trains recently."]
        feature_basictest=[]
        for tweet in basic_test:
            feature_basictest.append(feature_extraction.getallfeatureset(tweet))
        feature_basictest=np.array(feature_basictest)
        feature_basictestvec = vector.transform(feature_basictest)

        print(basic_test)
        print(classifier.predict(feature_basictestvec))

    elif(x=='DT'):
        # featureSets=np.array(featureSets)
        # targets=(featureSets[0::,1]=='Ironic').astype(int)

        # #Transforms lists of feature-value mappings to vectors
        # vector = DictVectorizer()
        # featureVector = vector.fit_transform(featureSets[0::,0])

        #Saving the dictionary vectorizer
        fName = "dictionaryFile.p"
        #write binary mode
        fObject = open(fName, 'wb')
        pickle.dump(vector, fObject)
        fObject.close()

        #Feature splitting
        print('Feaature Splitting')
        order = shuffle(list(range(len(featureSets))))
        #shuffling in the same order
        targets=targets[order]
        featureVector=featureVector[order, 0::]

        #Splitting data set in training and test set
        size = int(len(featureSets) * .3)

        trainVector = featureVector[size:,0::]
        trainTargets = targets[size:]
        testVector = featureVector[:size, 0::]
        testTargets = targets[:size]

        print('Training')

        #Artificial weights
        ironicP=(trainTargets==1)
        nonIronicP = (trainTargets==0)
        ratio = np.sum(nonIronicP.astype(float))/np.sum(ironicP.astype(float))

        newTrainVector = trainVector
        newTrainTargets=trainTargets
        #CORE PART OF WHOLE PROJECT
        for j in range(int(ratio-1.0)):
            newTrainVector=sp.sparse.vstack([newTrainVector,trainVector[ironicP,0::]]) #Stack sparse matrices vertically
            newTrainTargets=np.concatenate((newTrainTargets,trainTargets[ironicP]))

        classifier = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                                    max_depth=3, min_samples_leaf=2)
        classifier.fit(newTrainVector, newTrainTargets)

        #Saving the classifier
        fName = "classifier.p"
        fObject=open(fName,'wb')
        pickle.dump(classifier,fObject)
        fObject.close()

        #Validation
        print('Validating')
        class_names = ['ironic', 'regular']
        output = classifier.predict(testVector)
        classificationReport = classification_report(testTargets,output,target_names=classificationSet)
        print(classificationReport)
        print(accuracy_score(testTargets, output)*100)
        id3=accuracy_score(testTargets, output)*100

        def plot_confusion_matrix(cm, classes,
                                normalize=False,
                                title='Confusion matrix',
                                cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                #print("Normalized confusion matrix")
            else:
                #print('Confusion matrix, without normalization')
                print()

            #print(cm)

            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')


        #compute confusion matrix
        cnf_matrix=confusion_matrix(testTargets, output)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                            title='Confusion matrix, without normalization')

        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                            title='Normalized confusion matrix')

        plt.show()


        #BASIC TEST
        basic_test=["I have been on rather too many planes and trains recently."]
        feature_basictest=[]
        for tweet in basic_test:
            feature_basictest.append(feature_extraction.getallfeatureset(tweet))
        feature_basictest=np.array(feature_basictest)
        feature_basictestvec = vector.transform(feature_basictest)

        print(basic_test)
        print(classifier.predict(feature_basictestvec))


labels = ['SVM','ID3']
#sizes = [5, neg_per, neu_per]
sizes = [svm,id3]
index = np.arange(len(labels))
plt.bar(index, sizes)
plt.xlabel('Algorithm', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.xticks(index, labels, fontsize=10, rotation=0)
plt.title('comparative study')
plt.show()
