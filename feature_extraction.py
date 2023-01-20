#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:41:49 2023

@author: kst
"""
from shamir import Shamir
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
import seaborn as sns
from genDataMult1 import genData
N=100000
p=np.array([1,2,3,4,5])
A=[0,1]
H=[2,3,4]
SS = Shamir(len(p),2,0,10**6,p)
dts = genData(N,p,A,H,SS)

###########################################
## split data
###########################################
dts_train, dts_test = model_selection.train_test_split(dts, 
                      test_size=0.3)

# ## print info
# print("X_train shape:", dts_train.drop("Y",axis=1).shape, "| X_test shape:", dts_test.drop("Y",axis=1).shape)
# print("y_train mean:", round(np.mean(dts_train["Y"]),2), "| y_test mean:", round(np.mean(dts_test["Y"]),2))
# print(dts_train.shape[1], "features:", dts_train.drop("Y",axis=1).columns.to_list())

###########################################
# Normalize data
###########################################
# scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
# X = scaler.fit_transform(dts_train.drop("Y", axis=1))
# dts_scaled= pd.DataFrame(X, columns=dts_train.drop("Y", axis=1).columns, index=dts_train.index)
# dts_scaled["Y"] = dts_train["Y"]
# dts_scaled.head()

###########################################
#Correlation matrix
###########################################
# def corr_mat(corr_matrix):
#     for col in corr_matrix.columns:
#         if corr_matrix[col].dtype == "O":
#               corr_matrix[col] = corr_matrix[col].factorize(sort=True)[0]
#     corr_matrix = corr_matrix.corr(method="pearson")
#     sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=True, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5)
#     plt.title("pearson correlation")

# plt.figure('Sim')
# corr_mat(dts.loc[dts['Y'] == 0].drop('Y', axis = 1))
# plt.figure('View')
# corr_mat(dts.loc[dts['Y'] == 1].drop('Y', axis = 1))
# plt.figure('Both')
# corr_mat(dts)

###########################################
### LASSO regularization feature selection
###########################################
# X = dts_train.drop("Y", axis=1).values
# y = dts_train["Y"].values
# feature_names = dts_train.drop("Y", axis=1).columns
# ## Anova
# selector = feature_selection.SelectKBest(score_func=  
#                 feature_selection.f_classif, k=9).fit(X,y)
# anova_selected_features = feature_names[selector.get_support()]

# ## Lasso regularization
# selector = feature_selection.SelectFromModel(estimator= 
#               linear_model.LogisticRegression(C=1, penalty="l1", 
#               solver='liblinear'), max_features=9).fit(X,y)
# lasso_selected_features = feature_names[selector.get_support()]
 
# ## Plot
# dtf_features = pd.DataFrame({"features":feature_names})
# dtf_features["anova"] = dtf_features["features"].apply(lambda x: "anova" if x in anova_selected_features else "")
# dtf_features["num1"] = dtf_features["features"].apply(lambda x: 1 if x in anova_selected_features else 0)
# dtf_features["lasso"] = dtf_features["features"].apply(lambda x: "lasso" if x in lasso_selected_features else "")
# dtf_features["num2"] = dtf_features["features"].apply(lambda x: 1 if x in lasso_selected_features else 0)
# dtf_features["method"] = dtf_features[["anova","lasso"]].apply(lambda x: (x[0]+" "+x[1]).strip(), axis=1)
# dtf_features["selection"] = dtf_features["num1"] + dtf_features["num2"]
# sns.barplot(y="features", x="selection", hue="method", data=dtf_features.sort_values("selection", ascending=False), dodge=False)



###########################################
### Randomforest feature selection
###########################################
X = dts_train.drop("Y", axis=1).values
y = dts_train["Y"].values
feature_names = dts_train.drop("Y", axis=1).columns.tolist()
## Importance
model = ensemble.RandomForestClassifier(n_estimators=100,
                      criterion="entropy", random_state=0)
model.fit(X,y)
importances = model.feature_importances_
## Put in a pandas dtf
dts_importances = pd.DataFrame({"IMPORTANCE":importances, 
            "VARIABLE":feature_names}).sort_values("IMPORTANCE", 
            ascending=False)
dts_importances['cumsum'] = dts_importances['IMPORTANCE'].cumsum(axis=0)
dts_importances = dts_importances.set_index("VARIABLE")
    
## Plot
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False)
fig.suptitle("Features Importance", fontsize=20)
ax[0].title.set_text('variables')

dts_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(
                kind="barh", legend=False, ax=ax[0]).grid(axis="x")
ax[0].set(ylabel="")
ax[1].title.set_text('cumulative')
dts_importances[["cumsum"]].plot(kind="line", linewidth=4, 
                                  legend=False, ax=ax[1])
ax[1].set(xlabel="", xticks=np.arange(len(dts_importances)), 
          xticklabels=dts_importances.index)
plt.xticks(rotation=70)
plt.grid(axis='both')
plt.show()