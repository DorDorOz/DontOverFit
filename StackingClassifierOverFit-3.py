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
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import StackingCVClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import VotingClassifier
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


train = pd.read_csv('D:/Python/data/DontOverFit/train.csv').drop("id", axis = 'columns')
targets = train['target']
train.drop('target', axis = 'columns', inplace = True)

test = pd.read_csv('D:/Python/data/DontOverFit/test.csv').drop("id", axis = 'columns')

RANDOM_SEED = 0
lgbm_1 = LGBMClassifier(objective = 'binary', max_depth = 2, num_leaves = 16, random_state = 0,
                        feature_fraction = 0.8, bagging_fraction = 0.8, bagging_freq = 4,
                        min_child_weight = 10, min_child_samples = 6, n_estimators = 100)
xgb_1 = XGBClassifier(objective = 'binary:logistic', max_depth = 2, random_state = 0, min_child_weight = 10, subsample = 0.8)
rf_1 = RandomForestClassifier(n_estimators = 100, max_depth = 4, bootstrap = False, class_weight = 'balanced',
                              criterion = 'gini', max_features = 'auto', min_samples_leaf = 1,
                              min_samples_split = 6, min_impurity_decrease = 0)
dt_1 = DecisionTreeClassifier(max_depth = 32, random_state = 0, min_impurity_decrease = 0, max_features = 'sqrt')
svc_1 = SVC(kernel = 'rbf', C = 0.1, gamma = 'auto', probability = True)
gnb_1 = GaussianNB()
knn_1 = KNeighborsClassifier(n_neighbors = 30)
sgd_1 = SGDClassifier(max_iter = 1000, tol = 1e-3, loss = 'log')
lr_1 = LogisticRegression(C = 0.1, max_iter = 2000, class_weight = 'balanced', penalty = 'l1', solver = 'liblinear')
lr_2 = LogisticRegression(C = 5, max_iter = 2000, class_weight = 'balanced', penalty = 'l2', solver = 'sag')
meta_lr = LogisticRegression(C = 2, max_iter = 2000, class_weight = 'balanced', penalty = 'l2', solver = 'sag')

np.random.seed(0)
sclf = StackingCVClassifier(classifiers = [lgbm_1, xgb_1, rf_1, dt_1, knn_1, svc_1, gnb_1, sgd_1, lr_1, lr_2], 
                            use_probas = True,
                            use_features_in_secondary = False,
                            meta_classifier = meta_lr,
                            cv = 5)
num_folds = 5
folds = RepeatedStratifiedKFold(n_splits = num_folds, n_repeats = 5, random_state = 0)
#folds = StratifiedKFold(n_splits = num_folds, random_state = 0)

test_result_non_transform = np.zeros(len(test))
auc_score_non_transform = 0
cond_non_transform = 0

test_result_noise = np.zeros(len(test))
auc_score_noise = 0
cond_noise = 0

test_result_over = np.zeros(len(test))
auc_score_over = 0
cond_over = 0



for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, targets)):
    ##base
    X_train, y_train = train.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid, y_valid = train.iloc[val_idx], targets.iloc[val_idx]
    sclf.fit(X_train.values, y_train.values)
    y_pred = sclf.predict_proba(X_valid.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'Non_Transform:', auc)
    auc_score_non_transform += auc
    preds_non_transform = sclf.predict_proba(test.values)
    test_result_non_transform += preds_non_transform[:, 1]
    cond_non_transform += 1
    
    ##noise
    np.random.seed(0)
    noise = np.random.normal(0.01, 0.01, train.shape)
    train_noise = train + noise
    X_train, y_train = train_noise.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid, y_valid = train_noise.iloc[val_idx], targets.iloc[val_idx]
    sclf.fit(X_train.values, y_train.values)
    y_pred = sclf.predict_proba(X_valid.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'Noise:', auc)
    auc_score_noise += auc
    preds_noise = sclf.predict_proba(test.values)
    test_result_noise += preds_noise[:, 1]
    cond_noise += 1
    
    ##overSampled*2+noise
    X_train, y_train = train.iloc[trn_idx], targets.iloc[trn_idx]
    X_train = X_train.append(X_train).reset_index(drop = True)
    y_train = y_train.append(y_train).reset_index(drop = True)
    
    np.random.seed(0)
    noise = np.random.normal(0.01, 0.01, X_train.shape)
    X_train_over_noise = X_train + noise
    
    X_valid, y_valid = train.iloc[val_idx], targets.iloc[val_idx]
    sclf.fit(X_train_over_noise.values, y_train.values)
    y_pred = sclf.predict_proba(X_valid.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'OverNoise:', auc)
    auc_score_over += auc
    preds_over = sclf.predict_proba(test.values)
    test_result_over += preds_over[:, 1]
    cond_over += 1

  
auc_score_non_transform = auc_score_non_transform / cond_non_transform
print("AUC score_non_transform: ", auc_score_non_transform)
test_result_non_transform = test_result_non_transform / cond_non_transform

auc_score_noise = auc_score_noise / cond_noise
print("AUC score_noise: ", auc_score_noise)
test_result_noise = test_result_noise / cond_noise

auc_score_over = auc_score_over / cond_over
print("AUC score_over: ", auc_score_over)
test_result_over = test_result_over / cond_over


ens_DF = pd.DataFrame({'Base' : test_result_non_transform, 
                       'Noise' : test_result_noise,
                       'Over-2': test_result_over})

ens_DF_mean = pd.DataFrame((test_result_non_transform + 
                            test_result_over +
                            test_result_noise) / 3)

print((auc_score_noise + auc_score_non_transform + auc_score_over) / 3)

model_1_ens_lr = pd.concat([pd.DataFrame(testSub_Ids), pd.DataFrame(ens_DF_mean)], axis = 1)
model_1_ens_lr.columns = ['id', 'target']
model_1_ens_lr.to_csv(r'D:/Python/data/DontOverFit/StackingClassifier-3.csv', index = False)





    
    



