import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation

def model_evaluation(X, y, model, label):
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    print('{0} Accuracy: {1:.4f}'.format(label, accuracy))
    cm = confusion_matrix(y, predictions)
    cm = cm.astype('float') / cm.sum(axis=1)
    df_cm = pd.DataFrame(cm, index = sorted(y.unique()), columns = sorted(y.unique()))
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='.3f', cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def get_feature_importance(model, X_train):
    feature_importance = model.feature_importances_
    indices = np.argsort(feature_importance)[::-1]
    feature_importance = [feature_importance[i] for i in indices]
    predictors = X_train.columns
    predictors = [predictors[i] for i in indices]
    predictors_index = range(1, len(predictors) + 1)
    num_features = X_train.shape[1]
    #num_features = len(predictors) if num_features is None else num_features
    plt.figure(figsize=(20,20))
    plt.bar(predictors_index[:num_features], feature_importance[:num_features], align = 'center')
    plt.xticks(predictors_index[:num_features], predictors[:num_features], rotation = 'vertical')
    plt.show()

# load and clean data
train_df = pd.read_csv('train_w_cluster.csv')
test_df = pd.read_csv('test_w_cluster.csv')
train_df['vote_count'] = train_df['vote_count'].apply(lambda x: int(x.replace('[', '').replace(']', '')))

# create dummy columns and remove string variables
columns_to_remove = ['overview', 'tagline', 'plot', 'plot outline', 'mpaa_rating_text', 'id']
dummy_columns = ['mpaa_rating', 'overview_cluster', 'tagline_cluster', 'plot_cluster', 'plot outline_cluster', 'mpaa_rating_text_cluster']
for column in dummy_columns:
    train_df = pd.concat([train_df, pd.get_dummies(train_df[column], prefix = column, prefix_sep = '_')], axis=1)
    test_df = pd.concat([test_df, pd.get_dummies(test_df[column], prefix = column, prefix_sep = '_')], axis=1)
train_df = train_df.drop(columns_to_remove + dummy_columns, axis = 1)
test_df = test_df.drop(columns_to_remove + dummy_columns, axis = 1)

print(train_df.shape)
print(test_df.shape)

X_train = train_df.drop('genre', axis = 1)
y_train = train_df['genre']
X_test = test_df.drop('genre', axis = 1)
y_test = test_df['genre']

# base_model = XGBClassifier()
# base_model.fit(X_train, y_train)
# model_evaluation(X_train, y_train, base_model, 'Train')

# model_a = XGBClassifier()
# XGB_PARAMETERS = {'max_depth': range(3,10,2), 'min_child_weight': range(0,6,2)}
# model_a_cv = GridSearchCV(model_a, XGB_PARAMETERS, scoring = "f1_weighted", cv = 5, , verbose = 100)
# model_a_cv.fit(X_train, y_train)

# print model_a_cv.grid_scores_, model_a_cv.best_params_, model_a_cv.best_score_

# print "Model b"
# model_b = XGBClassifier()
# model_a_best = {'max_depth': 5, 'min_child_weight': 0}
# XGB_PARAMETERS = {k:[v] for k,v in model_a_best.items()}.copy()
# XGB_PARAMETERS.update({'gamma': [i/10.0 for i in range(0,5)]})
# print XGB_PARAMETERS
# model_b_cv = GridSearchCV(model_b, XGB_PARAMETERS, scoring = "f1_weighted", cv = 5, verbose = 100)
# model_b_cv.fit(X_train, y_train)

# print model_b_cv.grid_scores_, model_b_cv.best_params_, model_b_cv.best_score_

# print "Model c"
# model_c = XGBClassifier()
# XGB_PARAMETERS = {k:[v] for k,v in model_b_cv.best_params_.items()}.copy()
# XGB_PARAMETERS.update({'subsample':[i/10.0 for i in range(6,10)],
#                        'colsample_bytree':[i/10.0 for i in range(6,10)]})
# print XGB_PARAMETERS
# model_c_cv = GridSearchCV(model_c, XGB_PARAMETERS, scoring = "f1_weighted", cv = 5, verbose = 100)
# model_c_cv.fit(X_train, y_train)

# print model_c_cv.grid_scores_, model_c_cv.best_params_, model_c_cv.best_score_

# print "Model d"
# model_d = XGBClassifier()
# XGB_PARAMETERS = {k:[v] for k,v in model_c_cv.best_params_.items()}.copy()
# XGB_PARAMETERS.update({'reg_alpha':[1e-5, 1e-3, 0.1, 1, 10, 100]})
# print XGB_PARAMETERS
# model_d_cv = GridSearchCV(model_d, XGB_PARAMETERS, scoring = "f1_weighted", cv = 5, verbose = 100)
# model_d_cv.fit(X_train, y_train)

# print model_d_cv.grid_scores_, model_d_cv.best_params_, model_d_cv.best_score_

# print "Model e"
# model_e = XGBClassifier()
# XGB_PARAMETERS = {k:[v] for k,v in model_d_cv.best_params_.items()}.copy()
# XGB_PARAMETERS.update({'learning_rate':[0.1], 'n_estimators':[5000]})
# print XGB_PARAMETERS
# model_e_cv = GridSearchCV(model_d, XGB_PARAMETERS, scoring = "f1_weighted", cv = 5, verbose = 100)
# model_e_cv.fit(X_train, y_train)

# print model_e_cv.grid_scores_, model_e_cv.best_params_, model_e_cv.best_score_

model_final = XGBClassifier(
    learning_rate = 0.1,
    n_estimators = 5000,
    colsample_bytree = 0.9,
    gamma = 0.2,
    max_depth = 5,
    min_child_weight = 0,
    reg_alpha =0.1,
    subsample = 0.6)
model_final.fit(X_train, y_train)
model_evaluation(X_train, y_train, model_final.best_estimator_, 'Train')
model_evaluation(X_test, y_test, model_final.best_estimator_, 'Test')
get_feature_importance(model_cv.best_estimator_, X_train)