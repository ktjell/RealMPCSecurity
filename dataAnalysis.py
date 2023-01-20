#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 12:41:18 2023

@author: kst
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

import statsmodels.formula.api as smf
import statsmodels.api as sm

from genDataset import genData

dts = genData(20000)
y = 'Y'
x = 'e'


###########################################
### Histogram and box plot of a variable or feature.
###########################################
def hist_box(x):
    fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False)
    fig.suptitle(x, fontsize=20)
    ### distribution
    ax[0].title.set_text('distribution')
    variable = dts[x].fillna(dts[x].mean())
    breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
    variable = variable[ (variable > breaks[0]) & (variable < 
                        breaks[10]) ]
    sns.distplot(variable, hist=True, kde=True, kde_kws={"shade": True}, ax=ax[0])
    des = dts[x].describe()
    ax[0].axvline(des["25%"], ls='--')
    ax[0].axvline(des["mean"], ls='--')
    ax[0].axvline(des["75%"], ls='--')
    ax[0].grid(True)
    des = round(des, 2).apply(lambda x: str(x))
    box = '\n'.join(("min: "+des["min"], "25%: "+des["25%"], "mean: "+des["mean"], "75%: "+des["75%"], "max: "+des["max"]))
    ax[0].text(0.95, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right", bbox=dict(boxstyle='round', facecolor='white', alpha=1))
    ### boxplot 
    ax[1].title.set_text('outliers (log scale)')
    tmp_dts = pd.DataFrame(dts[x])
    # tmp_dts[x] = np.log(tmp_dtf[x])
    tmp_dts.boxplot(column=x, ax=ax[1])
    plt.show()

###########################################
### Bivariate distributions
###########################################
def biv_dist(num,cat):
    fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False)
    fig.suptitle(cat+"   vs   "+num, fontsize=20)
                
    ### distribution
    ax[0].title.set_text('density')
    for i in dts[cat].unique():
        sns.distplot(dts[dts[cat]==i][num], hist=False, label=i, ax=ax[0])
    ax[0].grid(True)
    ### stacked
    ax[1].title.set_text('bins')
    breaks = np.quantile(dts[num], q=np.linspace(0,1,11))
    tmp = dts.groupby([cat, pd.cut(dts[num], breaks, duplicates='drop')]).size().unstack().T
    tmp = tmp[dts[cat].unique()]
    tmp["tot"] = tmp.sum(axis=1)
    for col in tmp.drop("tot", axis=1).columns:
          tmp[col] = tmp[col] / tmp["tot"]
    tmp.drop("tot", axis=1).plot(kind='bar', stacked=True, ax=ax[1], legend=False, grid=True)
    ### boxplot   
    # ax[2].title.set_text('outliers')
    # sns.catplot(x=cat, y=num, data=dtf, kind="box", ax=ax[2])
    # ax[2].grid(True)
    plt.show()

###########################################
### ANOVA testing for correlation
###########################################
def anov(x,y):
    model = smf.ols(y+' ~ '+x, data=dts).fit()
    table = sm.stats.anova_lm(model)
    p = table["PR(>F)"][0]
    coeff, p = None, round(p, 3)
    conclusion = "Correlated" if p < 0.05 else "Non-Correlated"
    print("Anova F: the variables are", conclusion, "(p-value: "+str(p)+")")


# hist_box(x)
# biv_dist(x,y)
# anov(x, y)