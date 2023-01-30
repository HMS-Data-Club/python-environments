import pandas as pd
from sklearn import model_selection, ensemble, metrics
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme(style='ticks')

# Load data
data = pd.read_csv("../data/telomere_ALT/telomere.csv", sep='\t')
data.head(10)    # Show first ten samples
num_row = data.shape[0]     # number of samples in the dataset
num_col = data.shape[1]     # number of features in the dataset (plus the label column)

X = data.iloc[:, 0: num_col-1]    # feature columns
y = data['TMM']                   # label column
X_train, X_test, y_train, y_test = \
    model_selection.train_test_split(X, y, 
                                     test_size=0.2,      # reserve 20 percent data for testing
                                     stratify=y,         # stratified sampling
                                     random_state=0)
rf = ensemble.RandomForestClassifier(
    n_estimators = 10,           # 10 random trees in the forest
    criterion = 'entropy',       # use entropy as the measure of uncertainty
    max_depth = 3,               # maximum depth of each tree is 3
    min_samples_split = 5,       # generate a split only when there are at least 5 samples at current node
    class_weight = 'balanced',   # class weight is inversely proportional to class frequencies
    random_state = 0)
k = 3   # number of folds
# split data into k folds
kfold = model_selection.StratifiedKFold(
    n_splits = k,
    shuffle = True,
    random_state = 0)
cv = list(kfold.split(X_train, y_train))    # indices of samples in each fold

for j, (train_idx, val_idx) in enumerate(cv):
    print('validation fold %d:' % (j+1))
    print(val_idx)
accuracy = []

# Compute classifier's accuracy on each fold
for j, (train_idx, val_idx) in enumerate(cv):
    rf.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
    accuracy_per_fold = rf.score(X_train.iloc[val_idx], y_train.iloc[val_idx])
    accuracy.append(accuracy_per_fold)
    print('Fold %d accuracy: %.3f' % (j+1, accuracy_per_fold))
print('Average accuracy: %.3f' % np.mean(accuracy))

# Refit the classifier using the full training set
rf = rf.fit(X_train, y_train)

# Compute classifier's accuracy on the test set
test_accuracy = rf.score(X_test, y_test)
print('Test accuracy: %.3f' % test_accuracy)

# Predict labels for the test set
y = rf.predict(X_test)             # Solid prediction
y_prob = rf.predict_proba(X_test)  # Soft prediction

# Compute the ROC and PR curves for the test set
fpr, tpr, _ = metrics.roc_curve(y_test, y_prob[:, 0], pos_label='+')
precision_plus, recall_plus, _ = metrics.precision_recall_curve(y_test, y_prob[:, 0], pos_label='+')
precision_minus, recall_minus, _ = metrics.precision_recall_curve(y_test, y_prob[:, 1], pos_label='-')

# Compute the AUROC and AUPRCs for the test set
auroc = metrics.auc(fpr, tpr)
auprc_plus = metrics.auc(recall_plus, precision_plus)
auprc_minus = metrics.auc(recall_minus, precision_minus)

# Compute the confusion matrix for the test set
cm = metrics.confusion_matrix(y_test, y)

# Plot the ROC curve for the test set
plt.plot(fpr, tpr, label='test (area = %.3f)' % auroc)
plt.plot([0,1], [0,1], linestyle='--', color=(0.6, 0.6, 0.6))
plt.plot([0,0,1], [0,1,1], linestyle='--', color=(0.6, 0.6, 0.6))

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc='lower right')
plt.title('ROC curve')
plt.show()

# Plot the PR curve for the test set
plt.plot(recall_plus, precision_plus, label='positive class (area = %.3f)' % auprc_plus)
plt.plot(recall_minus, precision_minus, label='negative class (area = %.3f)' % auprc_minus)
plt.plot([0,1,1], [1,1,0], linestyle='--', color=(0.6, 0.6, 0.6))
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend(loc='lower left')
plt.title('Precision-recall curve')
plt.show()

# Plot the confusion matrix for the test set
plt.matshow(cm, interpolation='nearest', cmap=plt.cm.Blues, alpha=0.5)
plt.xticks(np.arange(2), ('+', '-'))
plt.yticks(np.arange(2), ('+', '-'))

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()