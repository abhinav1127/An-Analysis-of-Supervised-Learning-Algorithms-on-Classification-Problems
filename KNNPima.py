import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score,learning_curve

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

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    # plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# dataset = pd.read_csv("/Users/abhinavtirath/GT/Fall18/CS4641/HW1/creditcard.csv")
# dataset.shape
# dataset.head()
# X = dataset.drop(['Class'], axis=1)
# y = dataset['Class']
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
train = pd.read_csv("pimaDiabetes_train.csv")
test = pd.read_csv("pimaDiabetes_test.csv")
X_train = train.drop(['Class'], axis=1)
y_train = train['Class']
X_test = test.drop(['Class'], axis=1)
y_test = test['Class']


from sklearn.metrics import roc_curve, auc, recall_score
from sklearn.neighbors import KNeighborsClassifier

neighbors = np.linspace(1, 20, 20, endpoint=True)
train_results = []
CV_results = []
recall_train_results = []
recall_CV_results = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=int(k))
    scores = cross_val_score(knn, X_train, y_train, cv=3, scoring='recall')
    recall_CV_results.append(sum(scores) / float(len(scores)))
    scores = cross_val_score(knn, X_train, y_train, cv=3, scoring= 'roc_auc')
    CV_results.append(sum(scores) / float(len(scores)))
    knn.fit(X_train, y_train)
    train_pred = knn.predict(X_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    recall_train_results.append(recall_score(y_train, train_pred))

from matplotlib.legend_handler import HandlerLine2D
plt.figure()
line1, = plt.plot(neighbors, train_results, 'b', label='Train AUC')
line2, = plt.plot(neighbors, CV_results, 'r', label='CV AUC')
line3, = plt.plot(neighbors, recall_train_results, 'b', label='Train recall', color='green')
line4, = plt.plot(neighbors, recall_CV_results, 'r', label='CV recall', color='yellow')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('recall/ AUC')
plt.xlabel('# of Neighbors')
plt.title("Neighbors Evaluation")
plt.draw()
plt.savefig("NeighborsPD.png")

idealNeighbors = CV_results.index(max(CV_results))
idealNeighbors += 1


train_results = []
CV_results = []
recall_train_results = []
recall_CV_results = []

possibleWeights = ['uniform', 'distance']

for weights in possibleWeights:
    # we create an instance of Neighbours Classifier and fit the data.
    knn = KNeighborsClassifier(n_neighbors=idealNeighbors, weights=weights)
    scores = cross_val_score(knn, X_train, y_train, cv=3, scoring='recall')
    recall_CV_results.append(sum(scores) / float(len(scores)))
    scores = cross_val_score(knn, X_train, y_train, cv=3, scoring='roc_auc')
    CV_results.append(sum(scores) / float(len(scores)))
    knn.fit(X_train, y_train)
    train_pred = knn.predict(X_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    recall_train_results.append(recall_score(y_train, train_pred))
plt.figure()
line1, = plt.plot(possibleWeights, train_results, 'b', label='Train AUC')
line2, = plt.plot(possibleWeights, CV_results, 'r', label='CV AUC')
line3, = plt.plot(possibleWeights, recall_train_results, 'b', label='Train recall', color='green')
line4, = plt.plot(possibleWeights, recall_CV_results, 'r', label='CV recall', color='yellow')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('recall/ AUC')
plt.xlabel('Type of Weight')
plt.title("Weight Selection")
plt.draw()
plt.savefig("WeightPD.png")

idealWeightIndex = recall_CV_results.index(max(recall_CV_results))
idealWeight = possibleWeights[idealWeightIndex]

knn = KNeighborsClassifier(n_neighbors=idealNeighbors, weights=idealWeight)
knn.fit(X_train,y_train)

predictions = knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

plt.figure()
cnf_matrix = confusion_matrix(y_test, predictions)
plot_confusion_matrix(cnf_matrix, classes=[0, 1],
                      title='Confusion matrix')
plt.savefig("CFMatrix_KNNPD.png")

plt.figure()
plot_learning_curve(knn, "Learning Curve", X_train, y_train, cv=3)
plt.savefig("LearningCurveKNNPD.png")