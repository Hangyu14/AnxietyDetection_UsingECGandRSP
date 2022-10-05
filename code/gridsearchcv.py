from sklearn import svm,neighbors,naive_bayes,discriminant_analysis,neural_network,metrics,tree
from sklearn.cluster import k_means
from sklearn.model_selection import ShuffleSplit,KFold,cross_val_score,train_test_split,GridSearchCV,validation_curve,learning_curve
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import sem
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SequentialFeatureSelector,RFE,RFECV,SelectKBest, f_classif
from function import normalize,train_classifier
import matplotlib.pyplot as plt
import numpy as np
alldata =  pd.read_csv('HRV_1min/classification/twoleveldata_hrvrrv.csv')
alldata = alldata.iloc[:, 1:44]
hrv_feature = list(alldata.iloc[:, 0:24].columns)
rrv_feature = list(alldata.iloc[:, 24:42].columns)



hrv_data = alldata[['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_SDSD', 'HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_HTI', 'HRV_TINN', 'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_LFHF', 'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2','label']]
rrv_data = alldata [['RRV_RMSSD', 'RRV_MeanBB', 'RRV_SDBB', 'RRV_SDSD', 'RRV_CVBB', 'RRV_CVSD', 'RRV_MedianBB', 'RRV_MadBB', 'RRV_MCVBB', 'RRV_VLF', 'RRV_LF', 'RRV_HF', 'RRV_LFHF', 'RRV_LFn', 'RRV_HFn', 'RRV_SD1', 'RRV_SD2', 'RRV_SD2SD1','label']]
hrv_data_norm = normalize(hrv_data)
rrv_data_norm = normalize(rrv_data)
hrvrrv_data_norm = normalize(alldata)
X = hrv_data_norm.iloc[:,:-1]
y = hrv_data_norm['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

estimator =neighbors.KNeighborsClassifier(p=2,n_neighbors=35)
train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X, y, cv=10,return_times=True)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.plot(figsize=(20, 5))
plt.title("Validation Curve with SVM")
plt.fill_between(train_sizes,train_scores_mean - train_scores_std,
train_scores_mean + train_scores_std,alpha=0.1,color="r",)
plt.fill_between(train_sizes,test_scores_mean - test_scores_std,test_scores_mean + test_scores_std,alpha=0.1,
color="g",)
plt.plot(
train_sizes, train_scores_mean, "o-", color="r", label="Training score")
plt.plot(
train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
)
plt.legend(loc="best")
plt.show()


train_scores, test_scores = validation_curve(svm.SVC(), X, y, param_name="C", param_range=[0.1,1,10,100,1000],cv=5,scoring="accuracy",n_jobs=-1)

#
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
#
# Plot the model scores (accuracy) against the paramater range
#
param_range=[0.1,1,10,100,1000]
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with SVM")
plt.xlabel(r"C")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(
    param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw
)
plt.fill_between(
    param_range,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2,
    color="darkorange",
    lw=lw,
)
plt.semilogx(
    param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw
)
plt.fill_between(
    param_range,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.2,
    color="navy",
    lw=lw,
)
plt.legend(loc="best")
plt.show()

# grid cv search svm
param_grid = {'C': [0.1,1,10,100,1000], 'gamma': [0.001,0.05,0.02,1,10]} 
grid = GridSearchCV(svm.SVC(), param_grid,verbose = 3,return_train_score=True)
grid.fit(X_train, y_train)
# print best parameter after tuning
print(grid.best_params_)

a=pd.DataFrame(grid.best_params_,index=[0])
a.to_csv('svmpara.csv')
# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)
grid_predictions = grid.predict(X_test)
# print classification report
report = classification_report(y_test, grid_predictions)
print(report)
print('     ')



# grid cv search mlp
mlp_gs = neural_network.MLPClassifier(max_iter=100)
parameter_space = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
clf1 = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, verbose = 3)
clf1.fit(X_train, y_train)
# print best parameter after tuning
print(clf1.best_params_)
# print how our model looks after hyper-parameter tuning
print(clf1.best_estimator_)
clf_predictions = clf1.predict(X_test)
# print classification report
print(classification_report(y_test, clf_predictions))
print('     ')
# grid cv search knn
leaf_size = list(range(20,50))
n_neighbors = list(range(35,100))
p=[1,2]
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
clf = GridSearchCV(neighbors.KNeighborsClassifier(), hyperparameters, verbose = 3)
clf.fit(X_train, y_train)
# print best parameter after tuning
print(clf.best_params_)
# print how our model looks after hyper-parameter tuning
print(clf.best_estimator_)
clf_predictions = clf.predict(X_test)
# print classification report
print(classification_report(y_test, clf_predictions))
print('     ')





