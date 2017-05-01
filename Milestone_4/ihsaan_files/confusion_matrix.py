import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import sys
import os



def accuracy_score(y_test, y_pred):
    if y_test == y_pred:
        return 1
    else:
        return 0

def convert_to_class(predictions, y_test):
    '''
    inputs: array of predictions and array of y_test
    outputs: two lists with the class number for the prediction and the class number for the list
    '''
    prediction_class = []
    for i in predictions:
        prediction_class.append(i.argmax())
    y_test_class = []
    for i in y_test:
        y_test_class.append(i.argmax())
    return prediction_class, y_test_class

def score(prediction_class, y_test_class):
    '''
    inputs: 2 lists of predicted classes and actual classes
    outptus: accuracy score of lists
    '''
    accuracy = []
    for i in range(0,3495):
        y_test = y_test_class[i]
        y_pred = prediction_class[i]
        accuracy.append(accuracy_score(y_test, y_pred))
    print ('accuracy score:', float(sum(accuracy))/len(prediction_class))
    return float(sum(accuracy))/len(prediction_class)
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

####################
y_test_filepath = sys.argv[1]
pred_filepath = sys.argv[2]

y_test= np.load(y_test_filepath)

pred = np.load(pred_filepath)#import in your numpy predictions

pred_class, y_test_class = convert_to_class(pred, y_test) #change np arrays into lists
        
score(pred_class, y_test_class) #get accuracy score
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_class, pred_class)
class_names = ['1','2','3','4','5','6','7','8','9','10'] #fix this when we know the class names
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, of genre classifications')
plt.show(block = True)


