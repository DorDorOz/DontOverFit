import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import  XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn import  metrics
from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import seaborn as sn
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE, RFECV
import scipy.sparse
from sklearn.decomposition import PCA
import gc

#df_train = pd.read_csv("D:/Python/data/DontOverFit/train.csv")
#
#target_count = df_train.target.value_counts()
#print('Class 0:', target_count[0])
#print('Class 1:', target_count[1])
#print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
#
#target_count.plot(kind = 'bar', title = 'Count (target)');


#
#pca = PCA() 
#  
#pca.fit_transform(train) 
#plt.plot(pca.explained_variance_ratio_)
#plt.xlabel('number of components')
#plt.ylabel('cumulative explained variance')
#plt.show()


train = pd.read_csv("D:/Python/data/DontOverFit/train.csv")
X_train = train[train.columns.difference(['target', 'id'])]
y_train = train[['target']]

testsub = pd.read_csv("D:/Python/data/DontOverFit/test.csv")
X_testsub = testsub[testsub.columns.difference(['id'])]
testsub_Ids = testsub[['id']]

sc = preprocessing.StandardScaler()  
X_train = sc.fit_transform(X_train)  
testsub = sc.transform(X_testsub)  

pca = PCA(ncomponents = 150)
pca.fit_transform(train)
print(pca.explained_variance_ratio_)

#print('lr_clf_1 AUC: ', roc_auc_score(y_test, lr_clf_1.predict_proba(X_test)[:,1]))

pca = PCA().fit(train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

pca = PCA().fit(train)
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

#data = RobustScaler().fit_transform(np.concatenate((X_train, test), axis = 0))
#train_X = data[:250]
#test = data[250:]


#X_train_bins = pd.DataFrame({'var_33': train['33']})
#X_test_bins = pd.DataFrame({'var_33': testSub['33']})
#comb_bins = pd.DataFrame(np.concatenate((X_train_bins, X_test_bins), axis = 0))
#comb_bins.columns = ['var_33']
#
#cut_array = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
#cut_labels = ['33_m4', '33_m3', '33_m2', '33_m1', '33_zero', '33_p1', '33_p2', '33_p3', '33_p4']
#comb_bins['bins'] = pd.cut(comb_bins['var_33'], bins = cut_array, labels = cut_labels)
#comb_bins = pd.concat([comb_bins, pd.get_dummies(comb_bins['bins'])], axis = 1)

#X_train_col = pd.DataFrame(np.column_stack((X_train_col, pd.DataFrame(PCA().fit_transform(X_train_transform)))))   
#X_train_col = pd.DataFrame(np.column_stack((X_train_col, pd.DataFrame(MinMaxScaler().fit_transform(X_train_transform)))))
#X_train_col = pd.DataFrame(np.column_stack((X_train_col, pd.DataFrame(StandardScaler().fit_transform(X_train_transform)))))
#X_train_col = pd.DataFrame(np.column_stack((X_train_col, pd.DataFrame(RobustScaler().fit_transform(X_train_transform)))))

##X_train_col = pd.DataFrame(PCA().fit_transform(X_train_col))
##X_train_col = X_train_col.assign(isPos_33 = np.where(X_train[['33']] > 0 , 1, 0))
#X_train_col = X_train_col.assign(isPos_65 = np.where(X_train[['65']] > 0 , 1, 0))
#X_train_col = X_train_col.assign(isPos_217 = np.where(X_train[['217']] > 0 , 1, 0))
#X_train_col = X_train_col.assign(isPos_117 = np.where(X_train[['117']] > 0 , 1, 0))

###################################################################################################
#trainExtract = pd.read_csv("D:/Python/data/DontOverFit/targetExtract.csv")
#testsub = pd.read_csv("D:/Python/data/DontOverFit/test.csv")
#X_test_values = testsub[testsub.columns.difference(['id'])]
#X_test_values_target = pd.DataFrame(np.column_stack((X_test_values, trainExtract['target'])))
#
#maxDF = X_test_values_target[X_test_values_target.iloc[:,300] >= 0.85]
#minDF = X_test_values_target[X_test_values_target.iloc[:, 300] < 0.15]
#minTarDF = minDF.loc[:, ['Log_1', 'Log_2', 'Log_3', 'Log_4', 'Log_5', 
#                                 'Log_6', 'Log_7', 'Log_8', 'Log_9', 'Log_10', 'Log_Mean']].min(axis = 1)
#maxTarDF = maxDF.loc[:, ['Log_1', 'Log_2', 'Log_3', 'Log_4', 'Log_5', 
#                                 'Log_6', 'Log_7', 'Log_8', 'Log_9', 'Log_10', 'Log_Mean']].max(axis = 1)
#maxDFIds = maxDF['id']
#minDFIds = minDF['id']
#minMaxSub = pd.DataFrame(np.row_stack((pd.DataFrame(np.column_stack((minDFIds, minTarDF))), 
#                                       pd.DataFrame(np.column_stack((maxDFIds, maxTarDF))))))
#minMaxSub.columns = ['id', 'target']
#minMaxSub.id = minMaxSub.id.astype(int)
#minMaxSub.to_csv(r'D:/Python/data/DontOverFit/MinMaxSub.csv', index = False)

#sn.kdeplot(train.iloc[:,227], shade = True)
#
#
#pos_target = train.loc[train.target == 1]
#neg_target = train.loc[train.target == 0]
#
#testsub.iloc[:,227].describe()
#pos_target.iloc[:,227].describe()
#neg_target.iloc[:,227].describe()


X_train_sel = X_train[['33']]
X_train_sel['isPos'] = np.where(X_train_sel[['33']] > 0 , 1, 0)


pd.DataFrame(np.column_stack((train['33'], train['34'])))



np.random.seed(0)
pd.DataFrame(np.random.normal(0, 0.001, 250))



##EDA
numerical_features = train.columns[2:]
print('Distributions - Histograms columns')
plt.figure(figsize=(30, 200))
for i, col in enumerate(numerical_features):
    plt.subplot(50, 6, i + 1)
    plt.hist(train[col]) 
    plt.title(col)
gc.collect()


print('Distributions columns')
plt.figure(figsize=(30, 200))
for i, col in enumerate(numerical_features):
    plt.subplot(50, 6, i + 1)
    plt.hist(train[train["target"] == 0][col], alpha=0.5, label='0', color='b')
    plt.hist(train[train["target"] == 1][col], alpha=0.5, label='1', color='r')    
    plt.title(col)
gc.collect()



high = pd.read_csv("D:/Python/data/DontOverFit/854.csv")
low = pd.read_csv("D:/Python/data/DontOverFit/853.csv")


avgEns = pd.DataFrame({'id': high['id'],
                       'target': ((high['target'] * 0.8) + (low['target'] * 0.2))})
avgEns.to_csv(r'D:/Python/data/DontOverFit/highlow-2.csv', index = False)



