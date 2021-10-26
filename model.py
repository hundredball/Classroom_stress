import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, SparsePCA
from sklearn.model_selection import KFold, LeaveOneOut, RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE 
from imblearn.pipeline import Pipeline

class MyModel:
    
    def __init__(self, model_name, preprocess_method, imbalanced_method, X_train, Y_train, X_val, Y_val):
        
        self.model = self._select_model(model_name)
        self.preprocess_method = 
        
        
    def _select_model(self, model_name):
        
        if model_name in ['svm','xgboost','naive','random_forest']:
            if model_name == 'svm':
                clf = svm.SVC()
            elif model_name == 'xgboost':
                clf = xgb.XGBClassifier()
            elif model_name == 'naive':
                clf = GaussianNB()
            elif model_name == 'random_forest':
                clf = RandomForestClassifier(max_depth=2)
                
        return clf