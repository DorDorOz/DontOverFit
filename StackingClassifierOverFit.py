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
train = train[results_Single_AUC_group['Col']]

test = pd.read_csv('D:/Python/data/DontOverFit/test.csv').drop("id", axis = 'columns')
test = test[results_Single_AUC_group['Col']]

RANDOM_SEED = 0
lgbm_1 = LGBMClassifier(objective = 'binary', max_depth = 2, num_leaves = 8, random_state = 0)
lgbm_2 = LGBMClassifier(objective = 'binary', max_depth = 2, num_leaves = 6, random_state = 0)
lgbm_3 = LGBMClassifier(objective = 'binary', max_depth = 2, num_leaves = 4, random_state = 0)
xgb_1 = XGBClassifier(objective = 'binary:logistic', max_depth = 2, random_state = 0)
xgb_2 = XGBClassifier(objective = 'binary:logistic', max_depth = 3, random_state = 0)
xgb_3 = XGBClassifier(objective = 'binary:logistic', max_depth = 4, random_state = 0)
rf_1 = RandomForestClassifier(n_estimators = 100, max_depth = 2, random_state = 0)
rf_2 = RandomForestClassifier(n_estimators = 100, max_depth = 3, random_state = 0)
rf_3 = RandomForestClassifier(n_estimators = 100, max_depth = 4, random_state = 0)
dt_1 = DecisionTreeClassifier(max_depth = 2, random_state = 0)
dt_2 = DecisionTreeClassifier(max_depth = 4, random_state = 0)
dt_3 = DecisionTreeClassifier(max_depth = 6, random_state = 0)
svc_1 = SVC(kernel = 'rbf', C = 10, gamma = 1, probability = True)
gnb_1 = GaussianNB()
knn_1 = KNeighborsClassifier(n_neighbors = 10)
knn_2 = KNeighborsClassifier(n_neighbors = 20)
knn_3 = KNeighborsClassifier(n_neighbors = 30)
knn_4 = KNeighborsClassifier(n_neighbors = 40)
sgd_1 = SGDClassifier(max_iter = 1000, tol = 1e-3, loss = 'log')
lr_1 = LogisticRegression(C = 0.1, max_iter = 2000, class_weight = 'balanced', penalty = 'l1', solver = 'liblinear')
lr_2 = LogisticRegression(C = 2, max_iter = 2000, class_weight = 'balanced', penalty = 'l2', solver = 'sag')
lr_3 = LogisticRegression(C = 2, max_iter = 2000, class_weight = 'balanced', penalty = 'l2', solver = 'lbfgs')
lr_4 = LogisticRegression(C = 0.1, max_iter = 2000, class_weight = 'balanced', penalty = 'l1', solver = 'saga')
lr_5 = LogisticRegression(C = 2, max_iter = 2000, class_weight = 'balanced', penalty = 'l2', solver = 'newton-cg')
lr_6 = LogisticRegression(C = 2, max_iter = 2000, class_weight = 'balanced', penalty = 'l2', solver = 'sag')
lr_7 = LogisticRegression(C = 2, max_iter = 2000, class_weight = 'balanced', penalty = 'l2', solver = 'lbfgs')
meta_lr = LogisticRegression(C = 4, max_iter = 2000, class_weight = 'balanced', penalty = 'l2', solver = 'sag')



data_Robust = preprocessing.RobustScaler().fit_transform(np.concatenate((train, test), axis = 0))
train_Robust = pd.DataFrame(data_Robust[:250])
test_Robust = pd.DataFrame(data_Robust[250:])

data_MinMax = preprocessing.MinMaxScaler().fit_transform(np.concatenate((train, test), axis = 0))
train_MinMax = pd.DataFrame(data_MinMax[:250])
test_MinMax = pd.DataFrame(data_MinMax[250:])

data_Standard = preprocessing.StandardScaler().fit_transform(np.concatenate((train, test), axis = 0))
train_Standard = pd.DataFrame(data_Standard[:250])
test_Standard = pd.DataFrame(data_Standard[250:])

pca = PCA(n_components = 50)
train_Standard_PCA = pd.DataFrame(pca.fit_transform(train_Standard))
test_Standard_PCA = pd.DataFrame(pca.fit_transform(test_Standard))

train_Robust_PCA = pd.DataFrame(pca.fit_transform(train_Robust))
test_Robust_PCA = pd.DataFrame(pca.fit_transform(test_Robust))

train_MinMax_PCA = pd.DataFrame(pca.fit_transform(train_MinMax))
test_MinMax_PCA = pd.DataFrame(pca.fit_transform(test_MinMax))

np.random.seed(0)
sclf = StackingClassifier(classifiers = [lr_1, lr_2, lr_3, lr_4, lr_5, lr_6, lr_7,
                                           lgbm_1, lgbm_2, lgbm_3, xgb_1, xgb_2, xgb_3,
                                           rf_1, rf_2, rf_3, dt_1, dt_2, dt_3, knn_1, knn_2, knn_3, knn_4,
                                           svc_1, gnb_1, sgd_1, meta_lr], 
                            use_probas = True,
                            use_features_in_secondary = True,
                            meta_classifier = meta_lr)
num_folds = 5
folds = KFold(n_splits = num_folds, random_state = 0)
test_result_non_transform = np.zeros(len(test))
test_result_std_Scaled = np.zeros(len(test))
test_result_Robust = np.zeros(len(test))
test_result_Robust_PCA = np.zeros(len(test))
test_result_MinMax = np.zeros(len(test))
test_result_MinMax_PCA = np.zeros(len(test))
test_result_std_Scaled_PCA = np.zeros(len(test))

auc_score_non_transform = 0
auc_score_std_Scaled = 0
auc_score_Robust = 0
auc_score_Robust_PCA = 0
auc_score_MinMax = 0
auc_score_MinMax_PCA = 0
auc_score_std_Scaled_PCA = 0

cond_non_transform = 0
cond_std_Scaled = 0
cond_Robust = 0
cond_Robust_PCA = 0
cond_MinMax = 0
cond_MinMax_PCA = 0
cond_std_Scaled_PCA = 0

auc_criteria = 0.8

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, targets)):
    ##non_transformed
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
    ##stdScaled
    X_train_Standard, y_train = train_Standard.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid_Standard, y_valid = train_Standard.iloc[val_idx], targets.iloc[val_idx]
    sclf.fit(X_train_Standard.values, y_train.values)
    y_pred = sclf.predict_proba(X_valid_Standard.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'Std_Scaled:', auc)
    auc_score_std_Scaled += auc
    preds_std_Scaled = sclf.predict_proba(test_Standard.values)
    test_result_std_Scaled += preds_std_Scaled[:, 1]
    cond_std_Scaled += 1
    ##stdScaled_PCA
    X_train_Standard_PCA, y_train = train_Standard_PCA.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid_Standard_PCA, y_valid = train_Standard_PCA.iloc[val_idx], targets.iloc[val_idx]
    sclf.fit(X_train_Standard_PCA.values, y_train.values)
    y_pred = sclf.predict_proba(X_valid_Standard_PCA.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'Std_Scaled_PCA:', auc)
    auc_score_std_Scaled_PCA += auc
    preds_std_Scaled_PCA = sclf.predict_proba(test_Standard_PCA.values)
    test_result_std_Scaled_PCA += preds_std_Scaled_PCA[:, 1]
    cond_std_Scaled_PCA += 1
    ##Robust
    X_train_Robust, y_train = train_Robust.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid_Robust, y_valid = train_Robust.iloc[val_idx], targets.iloc[val_idx]
    sclf.fit(X_train_Robust.values, y_train.values)
    y_pred = sclf.predict_proba(X_valid_Robust.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'Rob_Scaled:', auc)
    auc_score_Robust += auc
    preds_Rob = sclf.predict_proba(test_Robust.values)
    test_result_Robust += preds_Rob[:, 1]
    cond_Robust += 1
    ##Robust_PCA
    X_train_Robust_PCA, y_train = train_Robust_PCA.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid_Robust_PCA, y_valid = train_Robust_PCA.iloc[val_idx], targets.iloc[val_idx]
    sclf.fit(X_train_Robust_PCA.values, y_train.values)
    y_pred = sclf.predict_proba(X_valid_Robust_PCA.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'Rob_Scaled_PCA:', auc)
    auc_score_Robust_PCA += auc
    preds_Rob_PCA = sclf.predict_proba(test_Robust_PCA.values)
    test_result_Robust_PCA += preds_Rob_PCA[:, 1]
    cond_Robust_PCA += 1
    ##MinMax
    X_train_MinMax, y_train = train_MinMax.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid_MinMax, y_valid = train_MinMax.iloc[val_idx], targets.iloc[val_idx]
    sclf.fit(X_train_MinMax.values, y_train.values)
    y_pred = sclf.predict_proba(X_valid_MinMax.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'MM_Scaled:', auc)
    auc_score_MinMax += auc
    preds_mm = sclf.predict_proba(test_MinMax.values)
    test_result_MinMax += preds_mm[:, 1]
    cond_MinMax += 1
    ##MinMax_PCA
    X_train_MinMax_PCA, y_train = train_MinMax_PCA.iloc[trn_idx], targets.iloc[trn_idx]
    X_valid_MinMax_PCA, y_valid = train_MinMax_PCA.iloc[val_idx], targets.iloc[val_idx]
    sclf.fit(X_train_MinMax_PCA.values, y_train.values)
    y_pred = sclf.predict_proba(X_valid_MinMax_PCA.values)
    auc = roc_auc_score(y_valid, y_pred[:, 1])
    print('Fold: ', fold_ + 1, 'MM_Scaled_PCA:', auc)
    auc_score_MinMax_PCA += auc
    preds_mm_PCA = sclf.predict_proba(test_MinMax_PCA.values)
    test_result_MinMax_PCA += preds_mm_PCA[:, 1]
    cond_MinMax_PCA += 1
  
auc_score_non_transform = auc_score_non_transform / cond_non_transform
print("AUC score_non_transform: ", auc_score_non_transform)
test_result_non_transform = test_result_non_transform / cond_non_transform

auc_score_std_Scaled = auc_score_std_Scaled / cond_std_Scaled
print("AUC score_std_Scaled: ", auc_score_std_Scaled)
test_result_std_Scaled = test_result_std_Scaled / cond_std_Scaled

auc_score_std_Scaled_PCA = auc_score_std_Scaled_PCA / cond_std_Scaled_PCA
print("AUC score_std_Scaled_PCA: ", auc_score_std_Scaled_PCA)
test_result_std_Scaled_PCA = test_result_std_Scaled_PCA / cond_std_Scaled_PCA

auc_score_Robust = auc_score_Robust / cond_Robust
print("AUC score_Robust: ", auc_score_Robust)
test_result_Robust = test_result_Robust / cond_Robust

auc_score_Robust_PCA = auc_score_Robust_PCA / cond_Robust_PCA
print("AUC score_Robust_PCA: ", auc_score_Robust_PCA)
test_result_Robust_PCA = test_result_Robust_PCA / cond_Robust_PCA

auc_score_MinMax = auc_score_MinMax / cond_MinMax
print("AUC score_MinMax: ", auc_score_MinMax)
test_result_MinMax = test_result_MinMax / cond_MinMax

auc_score_MinMax_PCA = auc_score_MinMax_PCA / cond_MinMax_PCA
print("AUC score_MinMax: ", auc_score_MinMax_PCA)
test_result_MinMax_PCA = test_result_MinMax_PCA / cond_MinMax_PCA

print("All-AUC", (auc_score_MinMax + 
                  auc_score_MinMax_PCA + 
                  auc_score_Robust + 
                  auc_score_Robust_PCA + 
                  auc_score_std_Scaled + 
                  auc_score_std_Scaled_PCA + 
                  auc_score_non_transform) / 7)



test_result_non_transform = pd.DataFrame(test_result_non_transform)
test_result_std_Scaled = pd.DataFrame(test_result_std_Scaled)
test_result_std_Scaled_PCA = pd.DataFrame(test_result_std_Scaled_PCA)
test_result_Robust = pd.DataFrame(test_result_Robust)
test_result_Robust_PCA = pd.DataFrame(test_result_Robust_PCA)
test_result_MinMax = pd.DataFrame(test_result_MinMax)
test_result_MinMax_PCA = pd.DataFrame(test_result_MinMax_PCA)
final_Ens = pd.DataFrame((test_result_non_transform + 
                          test_result_std_Scaled + 
                          test_result_std_Scaled_PCA + 
                          test_result_Robust + 
                          test_result_Robust_PCA + 
                          test_result_MinMax + 
                          test_result_MinMax_PCA) / 7)



model_1_ens_lr = pd.concat([pd.DataFrame(testSub_Ids), pd.DataFrame(final_Ens)], axis = 1)
model_1_ens_lr.columns = ['id', 'target']
model_1_ens_lr.to_csv(r'D:/Python/data/DontOverFit/9215.csv', index = False)






