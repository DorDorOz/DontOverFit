import numpy as np
import pandas as pd
from xgboost.sklearn import  XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn import  metrics
from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression, SGDClassifier, Lasso
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import StackingCVClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from lightgbm.sklearn import LGBMClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import seaborn as sn
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE, RFECV
import scipy.sparse
from sklearn.decomposition import PCA


train = pd.read_csv("D:/Python/data/DontOverFit/train.csv")
train_target = train[['target']]
train_values = pd.DataFrame({'var_33': train['33'], 'var_65': train['65']})

testSub = pd.read_csv('D:/Python/data/DontOverFit/test.csv').drop("id", axis = 'columns')
testSub_values = pd.DataFrame({'var_33': testSub['33'], 'var_65': testSub['65']})

X_train, X_test, y_train, y_test = train_test_split(train_values, train_target, test_size = 0.2, random_state = 0)


#########################################################################################
##LogisticRegression
lr_clf_1 = LogisticRegression(C = 0.1, 
                              max_iter = 1000, 
                              class_weight = 'balanced', 
                              penalty = 'l1', 
                              solver = 'liblinear',
                              random_state = 0)
lr_clf_1.fit(X_train, y_train.values.ravel())
print('AUC: ', roc_auc_score(y_test, lr_clf_1.predict_proba(X_test)[:, 1]))

#########################################################################################
##KNN
knn_clf_1 = KNeighborsClassifier(n_neighbors = 10)
knn_clf_1.fit(train_values, train_target.values.ravel())

#########################################################################################
##KNN
svc_clf_1 = SVC(kernel = 'rbf', C = 0.1, gamma = 0.1, probability = True)
svc_clf_1.fit(train_values, train_target.values.ravel())

#########################################################################################
##Lasso
lasso_clf_1 = Lasso(alpha = 0.1, tol = 0.01, random_state = 0, selection = 'random')
lasso_clf_1.fit(train_values, train_target.values.ravel())

#########################################################################################
##RF
rf_clf_1 = RandomForestClassifier(n_estimators = 1000, max_depth = 2, random_state = 0)
rf_clf_1.fit(train_values, train_target.values.ravel())

#########################################################################################
##DT
dt_clf_1 = DecisionTreeClassifier(max_depth = 2, random_state = 0)
dt_clf_1.fit(train_values, train_target.values.ravel())
#########################################################################################
ens_clf_1 = pd.DataFrame(np.column_stack((lr_clf_1.predict_proba(testSub_values)[:,1],
                              knn_clf_1.predict_proba(testSub_values)[:,1],
                              svc_clf_1.predict_proba(testSub_values)[:,1],
                              lasso_clf_1.predict(testSub_values),
                              rf_clf_1.predict_proba(testSub_values)[:, 1],
                              dt_clf_1.predict_proba(testSub_values)[:, 1],
                              ((lr_clf_1.predict_proba(testSub_values)[:,1]+
                              knn_clf_1.predict_proba(testSub_values)[:,1]+
                              svc_clf_1.predict_proba(testSub_values)[:,1]+
                              lasso_clf_1.predict(testSub_values)+
                              rf_clf_1.predict_proba(testSub_values)[:, 1]+
                              dt_clf_1.predict_proba(testSub_values)[:, 1])/6))))
ens_clf_1.columns = ['lr_clf_1', 'knn_clf_1', 'svc_clf_1', 
                     'lasso_clf_1', 'rf_clf_1', 'dt_clf_1', 'clf_1_means']

model_1_ens = pd.concat([pd.DataFrame(testSub_Ids), pd.DataFrame(lr_clf_1.predict_proba(testSub_values)[:,1])], axis = 1)
model_1_ens.columns = ['id', 'target']
model_1_ens.to_csv(r'D:/Python/data/DontOverFit/baseModels-onlyLR.csv', index = False)

































