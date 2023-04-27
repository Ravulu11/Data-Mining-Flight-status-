#!/usr/bin/env python
# coding: utf-8

# # Prediction of Flight Status in Top 25 Busiest domestic routes of UNITED STATES
# ## Introduction
# ### Authored by:
# #### Team Name : ELITE
# Team Members: Sindhura Alla, Medha Alla, Ravindra Kumar Velidandi, Sai Mithil Sagi, Venkata Saipavan Lahar Sudrosh Kumar Atchutha, Sanjana Thinderu,
# ### Description of the analysis
# In this project, we are using a dataset containing information about flight data from Jan2021- August2022   
# Our prediction task is to predict the flight status of the top 25 busiest airports. We are using the input variables OPERATINGAIRLINE, SEASON, DAYOFMONTH, DAYOFWEEK, HOLIDAY, ORIGIN, DESTINATION, DEPTIMEBLK,ARRTIMEBLK, DISTANCEGROUP, DEPDELAYMINUTESHIST7D, CANCELLEDHIST7D, DIVERTEDHIST7D, DEPDELAYMINUTESHIST30D, CANCELLEDHIST30D, DIVERTEDHIST30D.
# 
# We are using the following classifiers:
# * Decision Tree 
# * Random Forest
# * AdaBoost
# * GradientDescent
# * XGBoost.
# * MLP classifier
# * KNN
# 
# we have plotted a correlation plot to understand the relationship between the factors and target.
# The important factor which we are considering for our model is "F1 score". Since the accuracy and precision have an equal importance in our model and the data set is imbalanced, instead of focusing on Accuracy we are considering F1 score.

# #### Importing required packages

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from IPython.display import Image
from sklearn.neural_network import MLPClassifier


# In[2]:


random_seed = 9
np.random.seed(random_seed)


# ## Step 2 - Preliminary (Business) Problem Scoping
# We are developing a multi classifier models to identify  the flight status

# ## Step 3 - Loading, cleaning and preparing data

# In[3]:


flight_data_df = pd.read_csv('/content/drive/MyDrive/FlightStatus/FilteredColumnsData/BusyRoutesDataFinalPartial.csv')
flight_data_df.head(5)


# ### 3.1-Data Exploration

# In[ ]:


print(flight_data_df.columns)
print(flight_data_df.describe())
print(flight_data_df.info())


# ###3.2-Cleaning Column names:

# In[4]:


flight_data_df.columns = [col.strip().upper().replace(' ','_') for col in flight_data_df.columns]
flight_data_df.columns


# In[41]:


routes = {}
for origin in flight_data_df["ORIGIN"].unique():
  routes[origin] = flight_data_df[flight_data_df['ORIGIN'] == origin]['DESTINATION'].unique().tolist()
print(routes)
print(flight_data_df['FLIGHTSTATUS'].unique())


# ### 3.3-Dropping Unwanted Columns:

# In[47]:


# Origin and destination is already in columns as UID, so corresponding names columns should be removed
flight_data_df = flight_data_df.drop(columns=['ORIGINCITYNAME','DESTCITYNAME'])


# ### 3.4-Checking for null values if any:

# In[ ]:


flight_data_df.isnull().sum()


# ### 3.5-Categorizing the category columns:

# In[48]:


flight_data_df['OPERATINGAIRLINE'] = flight_data_df['OPERATINGAIRLINE'].astype('category')
flight_data_df['SEASON'] = flight_data_df['SEASON'].astype('category')
flight_data_df['DAYOFMONTH'] = flight_data_df['DAYOFMONTH'].astype('category')
flight_data_df['DAYOFWEEK'] = flight_data_df['DAYOFWEEK'].astype('category')
flight_data_df['ORIGIN'] = flight_data_df['ORIGIN'].astype('category')
flight_data_df['DESTINATION'] = flight_data_df['DESTINATION'].astype('category')
flight_data_df['DISTANCEGROUP'] = flight_data_df['DISTANCEGROUP'].astype('category')
flight_data_df['DEPTIMEBLK'] = flight_data_df['DEPTIMEBLK'].astype('category')
flight_data_df['ARRTIMEBLK'] = flight_data_df['ARRTIMEBLK'].astype('category')
flight_data_df['FLIGHTSTATUS'] = flight_data_df['FLIGHTSTATUS'].astype('category')
flight_data_df.dtypes


# ### 3.6-Encoding Category Columns:

# In[49]:


LE =LabelEncoder()
flight_data_df['OPERATINGAIRLINE'] = LE.fit_transform(flight_data_df['OPERATINGAIRLINE'])
flight_data_df['SEASON'] = LE.fit_transform(flight_data_df['SEASON'])
flight_data_df['DAYOFMONTH'] = LE.fit_transform(flight_data_df['DAYOFMONTH'])
flight_data_df['DAYOFWEEK'] = LE.fit_transform(flight_data_df['DAYOFWEEK'])
flight_data_df['ORIGIN'] = LE.fit_transform(flight_data_df['ORIGIN'])
flight_data_df['DESTINATION'] = LE.fit_transform(flight_data_df['DESTINATION'])
flight_data_df['DISTANCEGROUP'] = LE.fit_transform(flight_data_df['DISTANCEGROUP'])
flight_data_df['DEPTIMEBLK'] = LE.fit_transform(flight_data_df['DEPTIMEBLK'])
flight_data_df['ARRTIMEBLK'] = LE.fit_transform(flight_data_df['ARRTIMEBLK'])
flight_data_df['FLIGHTSTATUS'] = LE.fit_transform(flight_data_df['FLIGHTSTATUS'])
flight_data_df.head(20)


# ## Step 4 - Splitting data into Train and Test sets
# #### Creating the training set and the test set with a 70/30 split.
#  

# In[ ]:


# constructing datasets for analysis
target = 'FLIGHTSTATUS'
predictors = list(flight_data_df.columns)
predictors.remove(target)
X = flight_data_df[predictors]
Y = flight_data_df[target]


# In[ ]:


train_X,test_X, train_Y,test_Y = train_test_split(X,Y, test_size=0.3, random_state=random_seed)


# ## Step 5 - Training our models using Classifiers

# ### 5.1 - Decision Tree

# #### Non pruned decision tree

# In[ ]:


classTree = DecisionTreeClassifier(random_state=random_seed)
_ = classTree.fit(train_X, train_Y)


# In[ ]:


fig = plt.figure(figsize=(300,60), dpi=100)
_ = plot_tree(classTree, 
                feature_names=flight_data_df.columns,  
                class_names=['Cancelled', 'LongDelay', 'OnTime', 'ShortDelay'],
                fontsize=8
             )


# ####  Creating an initial 'wide' range of possible Hyperparameter values for Decision Tree and Random Forest Classifiers.

# In[ ]:


criterion = ['gini', 'entropy']
max_depth = [int(x) for x in np.linspace(1, 500, 50)]
min_samples_split = [int(x) for x in np.linspace(2, 500, 50)]
min_samples_leaf = [int(x) for x in np.linspace(1, 100, 50)]
max_leaf_nodes = [int(x) for x in np.linspace(2, len(test_Y), 50)]
min_impurity_decrease = [x for x in np.arange(0.0, 0.01, 0.0001).round(5)]
param_grid_random = { 'criterion': criterion,
                      'max_depth': max_depth,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf' : min_samples_leaf,
                      'max_leaf_nodes' : max_leaf_nodes,
                      'min_impurity_decrease' : min_impurity_decrease,
                     }


# In[ ]:


from sklearn.metrics import f1_score, make_scorer
f1micro = make_scorer(f1_score , average='micro')


# #### Pruning Decision Tree using Randomizedsearch CV

# In[ ]:


dtree_default = DecisionTreeClassifier(random_state=random_seed)
best_random_dtree_model = RandomizedSearchCV(
        estimator=DecisionTreeClassifier(random_state=random_seed), 
       scoring=f1micro , 
        param_distributions=param_grid_random, 
        n_iter = 1_000,error_score='raise', 
        cv=5,  ##cross validation,K 
        verbose=0, 
        n_jobs = -1
    )
best_random_dtree_model = best_random_dtree_model.fit(train_X, train_Y)


# In[ ]:


random_dtree_search_best_params = best_random_dtree_model.best_params_
print('Best parameters found: ', random_dtree_search_best_params)


# #### Testing the performance of the DecisionTree obtained by RandomizedCV search parameters

# In[ ]:


y_pred = best_random_dtree_model.predict(test_X)

print("************************************")
print(f"{'F1 Score: ':18}{f1_score(test_Y, y_pred,average='micro')}")
print("************************************")
print(f"{'Accuracy Score: ':18}{accuracy_score(test_Y, y_pred)}")
print(f"{'Recall Score:':18}{recall_score(test_Y, y_pred,average='weighted')}")
print(f"{'Precision Score: ':18}{precision_score(test_Y, y_pred,average='weighted')}")
print("************************************")


# #### GRID search for best Decision Tree model

# In[ ]:


# Limited gridsearch to small universe because of high data points and TPU crash for bigger universe
param_grid = { 'min_samples_split': [365,366,367,368],       
              'min_samples_leaf': [83,84,85,86],
              'min_impurity_decrease': [0.009,0.0001,0.002,0.003],
              'max_leaf_nodes':[7575,7576,7577,7578],  
              'max_depth': [70,71,72,73],
              'criterion': ['entropy']
              }

best_tree_grid_search_model = GridSearchCV(estimator=DecisionTreeClassifier(random_state=random_seed), 
                                    scoring=f1micro, param_grid=param_grid, cv=10, verbose=0,  n_jobs = -1)
best_tree_grid_search_model = best_tree_grid_search_model.fit(train_X, train_Y)


# In[ ]:


print('Best parameters found: ', best_tree_grid_search_model.best_params_)


# #### Testing the performance of the DecisionTree obtained by Grid search parameters

# In[ ]:


y_pred = best_tree_grid_search_model.predict(test_X)

print("************************************")
print(f"{'F1 Score: ':18}{f1_score(test_Y, y_pred,average='micro')}")
print("************************************")
print(f"{'Accuracy Score: ':18}{accuracy_score(test_Y, y_pred)}")
print(f"{'Recall Score:':18}{recall_score(test_Y, y_pred,average='weighted')}")
print(f"{'Precision Score: ':18}{precision_score(test_Y, y_pred,average='weighted')}")
print("************************************")


# ### 5.2 - Random Forest

# #### Randomized search on RandomForest classifier

# In[ ]:


randomtree_default = RandomForestClassifier(random_state=random_seed)

best_random_forest_model = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=random_seed), 
        scoring=f1micro, 
        param_distributions=param_grid_random, 
        n_iter = 1_000, 
        cv=5, 
        verbose=0, 
        n_jobs = -1,
        random_state=random_seed
    )
best_random_forest_model = best_random_forest_model.fit(train_X, train_Y)


# In[ ]:


random_forest_search_best_params = best_random_forest_model.best_params_
print('Best parameters found: ', random_forest_search_best_params)


# #### Testing the performance of RandomForest obtained by Randomized searchCV parameters

# In[ ]:


y_pred = best_random_forest_model.predict(test_X)

print("************************************")
print(f"{'F1 Score: ':18}{f1_score(test_Y, y_pred,average='micro')}")
print("************************************")
print(f"{'Accuracy Score: ':18}{accuracy_score(test_Y, y_pred)}")
print(f"{'Recall Score:':18}{recall_score(test_Y, y_pred,average='weighted')}")
print(f"{'Precision Score: ':18}{precision_score(test_Y, y_pred,average='weighted')}")
print("************************************")


# #### GRID search for best Random Forest model

# In[ ]:


# Limited gridsearch to small universe because of high data points and TPU crash for bigger universe
param_grid = { 'min_samples_split': [60,61,62,63,64],       
              'min_samples_leaf': [25,26,27,28],
              'min_impurity_decrease': [0.0,0.0001],
              'max_leaf_nodes':[51505,51506,51507,51508],  
              'max_depth': [161,162,163,164,165],
              'criterion': ['entropy']
              }

best_grid_forest_model = GridSearchCV(estimator=RandomForestClassifier(random_state=random_seed), 
                                    scoring=f1micro, param_grid=param_grid, cv=5, verbose=0,  n_jobs = -1)
best_grid_forest_model = best_grid_forest_model.fit(train_X, train_Y)


# In[ ]:


random_forest_grid_search_best_params = best_grid_forest_model.best_params_
print('Best parameters found: ', random_forest_grid_search_best_params)


# #####  Testing the performance of RandomForest obtained by Grid search parameters

# In[ ]:


y_pred = best_grid_forest_model.predict(test_X)
print("************************************")
print(f"{'F1 Score: ':18}{f1_score(test_Y, y_pred,average='micro')}")
print("************************************")
print(f"{'Accuracy Score: ':18}{accuracy_score(test_Y, y_pred)}")
print(f"{'Recall Score:':18}{recall_score(test_Y, y_pred,average='weighted')}")
print(f"{'Precision Score: ':18}{precision_score(test_Y, y_pred,average='weighted')}")
print("************************************")


# ### 5.3 -  ADABoost

# #### Randomized search for ADABoost

# In[ ]:


n_estimators = [int(x) for x in np.linspace(1, 100, 25)]
learning_rate = [float(x) for x in np.linspace(1, 10, 10)]
param_grid_Adaboost = { 'n_estimators': n_estimators,
                      'learning_rate': learning_rate,
                     }
Adaboost_default = AdaBoostClassifier(random_state=random_seed)

best_random_adaboost_model = RandomizedSearchCV(
        estimator=AdaBoostClassifier(random_state=random_seed), 
        scoring=f1micro, 
        param_distributions=param_grid_Adaboost, 
        n_iter = 5_000, 
        cv=10, 
        verbose=0, 
        n_jobs = -1,
        random_state=random_seed
    )
best_random_adaboost_model = best_random_adaboost_model.fit(train_X, train_Y)


# In[ ]:


random_search_best_ada_params = best_random_adaboost_model.best_params_
print('Best parameters found: ', random_search_best_ada_params)


# ##### Testing the performance of the ADABoost selected parametres obtained by Randomized searchCV

# In[ ]:


y_pred = best_random_adaboost_model.predict(test_X)

print("************************************")
print(f"{'F1 Score: ':18}{f1_score(test_Y, y_pred,average='micro')}")
print("************************************")
print(f"{'Accuracy Score: ':18}{accuracy_score(test_Y, y_pred)}")
print(f"{'Recall Score:':18}{recall_score(test_Y, y_pred,average='weighted')}")
print(f"{'Precision Score: ':18}{precision_score(test_Y, y_pred,average='weighted')}")
print("************************************")


# #### Grid search for ADABoost best model

# In[ ]:


# Limited gridsearch to small universe because of high data points and high runtime on TPU (> 8hrs)
param_grid = {'n_estimators': [90,91,92,93],       
              'learning_rate': [0.99,1,0.98,1.1,1.2,0.97]}
grid_adaboostmodel = GridSearchCV(estimator=AdaBoostClassifier(random_state=random_seed),
                                   scoring=f1micro, param_grid=param_grid, cv=5, verbose=0,  n_jobs = -1)
grid_adaboostmodel = grid_adaboostmodel.fit(train_X, train_Y)


# In[ ]:


grid_adaboost_best_params = grid_adaboostmodel.best_params_
print('Best parameters found: ', grid_adaboost_best_params)


# ##### Testing the performance of the ADABoost selected parametres obtained by GridsearchCV

# In[ ]:


y_pred = grid_adaboostmodel.predict(test_X)

print("************************************")
print(f"{'F1 Score: ':18}{f1_score(test_Y, y_pred,average='micro')}")
print("************************************")
print(f"{'Accuracy Score: ':18}{accuracy_score(test_Y, y_pred)}")
print(f"{'Recall Score:':18}{recall_score(test_Y, y_pred,average='weighted')}")
print(f"{'Precision Score: ':18}{precision_score(test_Y, y_pred,average='weighted')}")
print("************************************")


# ### 5.4 - Gradient Boosting

# #### Randomized search for GradientBoost

# In[ ]:


n_estimators = [int(x) for x in np.linspace(1, 100, 5)]
learning_rate = [float(x) for x in np.linspace(1, 10, 1)]
max_depth = [int(x) for x in np.linspace(1, 500, 5)]
param_grid_gradiant = { 'n_estimators': n_estimators,
                      'learning_rate': learning_rate,
                      'max_depth':max_depth
                     }
best_random_gradiant_model = RandomizedSearchCV(
        estimator=GradientBoostingClassifier(random_state=random_seed), 
        scoring= f1micro, 
        param_distributions=param_grid_gradiant, 
        n_iter = 1_000, 
        cv=2, 
        verbose=0, 
        n_jobs = -1,
        random_state=random_seed
    )
best_random_gradiant_model = best_random_gradiant_model.fit(train_X, train_Y)


# In[ ]:


random_search_best_gradiant_params = best_random_gradiant_model.best_params_
print('Best parameters found: ', random_search_best_gradiant_params)


# ##### Testing the performance of the Gradient boost selected parametres obtained by Randomized searchCV

# In[ ]:


y_pred = best_random_gradiant_model.predict(test_X)

print("************************************")
print(f"{'F1 Score: ':18}{f1_score(test_Y, y_pred,average='micro')}")
print("************************************")
print(f"{'Accuracy Score: ':18}{accuracy_score(test_Y, y_pred)}")
print(f"{'Recall Score:':18}{recall_score(test_Y, y_pred,average='weighted')}")
print(f"{'Precision Score: ':18}{precision_score(test_Y, y_pred,average='weighted')}")
print("************************************")


# #### Grid Search for Gradient Boost

# In[ ]:


# Limited gridsearch to small universe because of high data points and high runtime on TPU (> 8hrs)
param_grid = {'n_estimators': [98,99,100,101],       
              'learning_rate': [1.0,0.09,0.08,1.1], 
              'max_depth': [1,2,3]
              }
              
best_grid_gradiant_model = GridSearchCV(estimator=GradientBoostingClassifier(random_state=random_seed), 
                                    scoring=f1micro, param_grid=param_grid, cv=10, verbose=0,  n_jobs = -1)
best_grid_gradiant_model = best_grid_gradiant_model.fit(train_X, train_Y)


# In[ ]:


grid_gradiant_best_params = best_grid_gradiant_model .best_params_
print('Best parameters found: ', grid_gradiant_best_params)


# ##### Testing the performance of the Gradient boost selected parametres obtained by GridSearch CV

# In[ ]:


y_pred = best_grid_gradiant_model.predict(test_X)

print("************************************")
print(f"{'F1 Score: ':18}{f1_score(test_Y, y_pred,average='micro')}")
print("************************************")
print(f"{'Accuracy Score: ':18}{accuracy_score(test_Y, y_pred)}")
print(f"{'Recall Score:':18}{recall_score(test_Y, y_pred,average='weighted')}")
print(f"{'Precision Score: ':18}{precision_score(test_Y, y_pred,average='weighted')}")
print("************************************")


# ### 5.5 - XGBoost

# #### Randomized search for XGBoost

# In[ ]:


n_estimators=[int(x) for x in np.linspace(1, 500, 5)]
min_child_weight=[4,5]
gamma=[i/10.0 for i in range(3,6)]
subsample=[i/10.0 for i in range(6,11)]
colsample_bytree = [i/10.0 for i in range(6,11)]
max_depth= [int(x) for x in np.linspace(1, 500, 50)]
objective= ['reg:squarederror', 'reg:tweedie']
booster= ['gbtree', 'gblinear']
eval_metric= ['rmse']
eta= [i/10.0 for i in range(3,6)]
param_grid_random = {
    'n_estimators':n_estimators,
    'min_child_weight':min_child_weight, 
    'gamma':gamma,  
    'subsample':subsample,
    'colsample_bytree':colsample_bytree, 
    'max_depth': max_depth,
    'objective': objective,
    'booster': booster,
    'eval_metric': eval_metric,
    'eta': eta,
}


# In[ ]:


XGB_C = XGBClassifier(random_state=random_seed)
best_random_XGB_model = RandomizedSearchCV(
        estimator= XGB_C, 
        scoring=f1micro, 
        param_distributions=param_grid_random, 
        n_iter = 10, 
        cv=5, 
        verbose=0, 
        n_jobs = -1, 
        random_state=random_seed
    )
best_random_XGB_model = best_random_XGB_model.fit(train_X.values, train_Y.values)


# In[ ]:


best_random_search_xgb_params = best_random_XGB_model.best_params_
print('Best parameters found: ', best_random_search_xgb_params)


# ##### Testing the performance of the XGBoost selected parametres obtained by Randomized searchCV

# In[ ]:


y_pred = best_random_XGB_model.predict(test_X.values)

print("************************************")
print(f"{'F1 Score: ':18}{f1_score(test_Y, y_pred,average='micro')}")
print("************************************")
print(f"{'Accuracy Score: ':18}{accuracy_score(test_Y, y_pred)}")
print(f"{'Recall Score:':18}{recall_score(test_Y, y_pred,average='weighted')}")
print(f"{'Precision Score: ':18}{precision_score(test_Y, y_pred,average='weighted')}")
print("************************************")


# #### Grid Search for XGBoost

# In[ ]:


# Limited gridsearch to small universe because of high data points and high runtime on TPU (> 2hrs for below universe)
param_grid = {
              'n_estimators': [1,2,3],       
              'min_child_weight': [3,4],
              'max_depth': [356,357],
              'gamma': [0.2,0.3],
              'eval_metric': ['rmse'],
              'eta': [0.5],
              'colsample_bytree': [0.8,0.9],
              'booster': ['gblinear'],
              'subsample': [0.9],
              'objective': ['reg:tweedie'],
              }
              
best_grid_xgb_model = GridSearchCV(estimator=XGBClassifier(random_state=random_seed), 
                                  scoring=f1micro, param_grid=param_grid, cv=5, verbose=0, n_jobs = -1)


# In[ ]:


best_grid_xgb_model = best_grid_xgb_model.fit(train_X.values, train_Y.values)


# ##### Testing the performance of XGBoost selected parametres by GridsearchCV

# In[ ]:


y_pred = best_grid_xgb_model.predict(test_X.values)

print("************************************")
print(f"{'F1 Score: ':18}{f1_score(test_Y, y_pred,average='micro')}")
print("************************************")
print(f"{'Accuracy Score: ':18}{accuracy_score(test_Y, y_pred)}")
print(f"{'Recall Score:':18}{recall_score(test_Y, y_pred,average='weighted')}")
print(f"{'Precision Score: ':18}{precision_score(test_Y, y_pred,average='weighted')}")
print("************************************")


# ### 5.6 - Neural Networks (MLP Classifier)

# In[ ]:


get_ipython().run_cell_magic('time', '', "# Classification with 3 hidden layers\nan1 = MLPClassifier(hidden_layer_sizes=(20,10,5), solver='adam', max_iter=1000)\nN1  = an1.fit(train_X, train_Y)")


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ny_pred_m1 = N1.predict(test_X)')


# In[ ]:


print('********Neural Model-1*********')
print("************************************")
print(f"{'F1 Score: ':18}{f1_score(test_Y, y_pred_m1,average='micro')}")
print("************************************")
print(f"{'Accuracy Score: ':18}{accuracy_score(test_Y, y_pred_m1)}")
print(f"{'Recall Score:':18}{recall_score(test_Y, y_pred_m1,average='weighted')}")
print(f"{'Precision Score: ':18}{precision_score(test_Y, y_pred_m1,average='weighted')}")
print("************************************")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Classification with six layers and high number of hidden nodes\nan2 = MLPClassifier(hidden_layer_sizes=(500,250,150,250,50,10), solver='adam', max_iter=1000)\nN2  = an1.fit(train_X, train_Y)")


# In[ ]:


y_pred_m1 = N2.predict(test_X)


# In[ ]:


print('********Neural Model-2*********')
print("************************************")
print(f"{'F1 Score: ':18}{f1_score(test_Y, y_pred_m1,average='micro')}")
print("************************************")
print(f"{'Accuracy Score: ':18}{accuracy_score(test_Y, y_pred_m1)}")
print(f"{'Recall Score:':18}{recall_score(test_Y, y_pred_m1,average='weighted')}")
print(f"{'Precision Score: ':18}{precision_score(test_Y, y_pred_m1,average='weighted')}")
print("************************************")


# ### 5.7 - KNearest Neighbours

# #### Randomized search on KNN

# In[ ]:


n_neighbors = [int(x) for x in np.linspace(1,20,5)]
weights = ['uniform','distance']
metric = ['euclidean','manhattan','chebyshev','seuclidean','minkowski'] 
p = [int(x) for x in np.linspace(1,10,5)]
leaf_size = [int(x) for x in np.linspace(1,20,5)]

random_grid_knn = {
    'n_neighbors': n_neighbors,
    'weights': weights,
    'metric': metric,
    'p':p,
    'leaf_size': leaf_size
}


# In[ ]:


knn = KNeighborsClassifier() 
knn_random = RandomizedSearchCV(estimator = knn, 
                                random_state = random_seed,
                                n_jobs = -1,
                                param_distributions = random_grid_knn,
                                n_iter = 5,
                                cv=3,
                                verbose = 2)
knn_random.fit(train_X, train_Y)


# In[ ]:


random_search_best_params_knn = knn_random.best_params_
print('Best parameters found: ', random_search_best_params_knn)


# In[ ]:


validation_predictions = knn_random.predict(test_X)


# In[ ]:


print('********KNN*********')
print("************************************")
print(f"{'F1 Score: ':18}{f1_score(test_Y, validation_predictions,average='micro')}")
print("************************************")
print(f"{'Accuracy Score: ':18}{accuracy_score(test_Y, validation_predictions)}")
print(f"{'Recall Score:':18}{recall_score(test_Y, validation_predictions,average='weighted')}")
print(f"{'Precision Score: ':18}{precision_score(test_Y, validation_predictions,average='weighted')}")
print("************************************")


# ### Step 6 - Correlation Plot

# In[50]:


import seaborn as sns
import matplotlib.pyplot as plt
corr_df= flight_data_df.drop(columns=['ORIGIN','DESTINATION'])
sns.set_theme(style="darkgrid")

# Compute the correlation matrix
corr = corr_df.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 15))

sns.heatmap(corr, annot=True, fmt=".2f", cmap='viridis')


# ### Summary of results:

# By performing Randomized searchCV and Gridsearch CV we have tested the selected parametres for the best models and obtained the following results:
# *   Decision Tree:  F1 Score: 0.8422853601692061 
# *   Random Forest:   F1 Score: 0.8434304651820717 
# *   ADABoost:  F1 Score: 0.8428377049401178 
# *   GradiantBoost:  F1 Score: 0.843174500532137
# *   XGBoost:  F1 Score: 0.8426895148796293
# *   Neural Network: F1 Score: 0.8428781204111602
# *   KNN: F1 Score: 0.840345417559175
# 
# > From the correlation plot we found that the ArrivalTimeBlk, DepTimeBlK,DepdelayHist7d, DepdelayHist30d, CancelledHist7d, CancelledHist30d and season have high negative correlation on the Flightstatus compared to others. 
# 
# > This indicates that flight is tend to get cancelled more for end of the day flights or winter flights. Also if past delay or cancellations are high then flight is tended to delays or cancellations
# 
# 
# *Note : Due to huge dataset, we have performed Grid searchCV on limited parameters. Eventhough we have opted for Google Colab to run our analysis, we noticed cases where grid search took more than 12hours to do full run. Due to time constraints, we opted to decrease our grid search universe for the reasons above. Hence it's possible that bigger parameters universe could improve our f1-score*
# 

# In[ ]:




