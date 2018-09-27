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

# dataset = pd.read_csv("/Users/abhinavtirath/GT/Fall18/CS4641/HW1/pimaDiabetes.csv")
# dataset.shape
# dataset.head()
# X = dataset.drop(['Class'], axis=1)
# y = dataset['Class']
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
train = pd.read_csv("creditcard_train.csv")
test = pd.read_csv("creditcard_test.csv")
X_train = train.drop(['Class'], axis=1)
y_train = train['Class']
X_test = test.drop(['Class'], axis=1)
y_test = test['Class']


from sklearn.metrics import roc_curve, auc, recall_score
from sklearn import tree

max_depths = np.linspace(1, 25, 25, endpoint=True)
train_results = []
CV_results = []
recall_train_results = []
recall_CV_results = []
for max_depth in max_depths:
   clf = tree.DecisionTreeClassifier(max_depth=max_depth)
   scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='recall')
   recall_CV_results.append(sum(scores) / float(len(scores)))
   scores = cross_val_score(clf, X_train, y_train, cv=3, scoring= 'roc_auc')
   CV_results.append(sum(scores) / float(len(scores)))
   clf.fit(X_train, y_train)
   train_pred = clf.predict(X_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   recall_train_results.append(recall_score(y_train, train_pred))

from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_depths, CV_results, 'r', label='CV AUC')
line3, = plt.plot(max_depths, recall_train_results, 'b', label='Train Recall', color='green')
line4, = plt.plot(max_depths, recall_CV_results, 'r', label='CV Recall', color='yellow')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Recall/ AUC')
plt.xlabel('Tree depth')
plt.title("Pre-pruning Metrics")
plt.draw()
plt.savefig("Pre-pruningMetricsCC.png")

# line3, = plt.plot(max_depths, recall_train_results, 'b', label='recall_train_results')
# line4, = plt.plot(max_depths, recall_test_results, 'r', label='recall_test_results')
# plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
# plt.ylabel('recall score')
# plt.xlabel('Tree depth')


depth = recall_CV_results.index(max(recall_CV_results))
depth += 1
print("max_depth", depth)

classifier = tree.DecisionTreeClassifier(criterion= 'gini', max_depth = depth)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(classifier.get_params())

plt.figure()
cnf_matrix = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cnf_matrix, classes=[0, 1],
                      title='Confusion matrix, pre-pruning')
plt.title('Confusion matrix, prepruning')
plt.savefig("CFMatrix_prepruningCC.png")

plt.figure()
plot_learning_curve(classifier, "Learning Curve", X_train, y_train, cv=3)
plt.savefig("LearningCurveDTCC.png")



from sklearn.tree._tree import TREE_LEAF

def prune_index(inner_tree, index, threshold, count):
    if inner_tree.value[index].min() < threshold:
        count += 1
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
    # if there are children, visit them as well
    if inner_tree.children_left[index] != TREE_LEAF:
        count = prune_index(inner_tree, inner_tree.children_left[index], threshold, count)
        count = prune_index(inner_tree, inner_tree.children_right[index], threshold, count)
    return count

import copy

postPrunedClassifier = copy.deepcopy(classifier)

thresholds = np.linspace(0, 15, 15, endpoint=True)
recall_values = []
AUC_values = []

for threshold in thresholds:
    # start pruning from the root
    numOfNodesRemoved = prune_index(classifier.tree_, 0, threshold, 0)
    y_pred = classifier.predict(X_test)
    recall_values.append(recall_score(y_test, y_pred))

    false_positive_rate, true_positive_rate, thresholds1 = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    AUC_values.append(roc_auc)


plt.figure()
line1, = plt.plot(thresholds, AUC_values, 'b', label='AUC')
line2, = plt.plot(thresholds, recall_values, 'r', label='recall')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Recall/ AUC')
plt.xlabel('Threshold to Remove Nodes')
plt.title("Post-Pruning")
plt.draw()
plt.savefig("Post-PruningCC.png")

threshold = recall_values.index(max(recall_values))
# threshold += 1

numOfNodesRemoved = prune_index(postPrunedClassifier.tree_, 0, threshold, 0)
y_pred = postPrunedClassifier.predict(X_test)

print("numOfNodesRemoved", numOfNodesRemoved)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

plt.figure()
cnf_matrix = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cnf_matrix, classes=[0, 1],
                      title='Confusion matrix, post-pruning')
plt.savefig("CFMatrix_postpruningCC.png")

for name, importance in zip(X_train.columns, classifier.feature_importances_):
    print(name, importance)