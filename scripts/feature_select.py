#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   feature_select.py
@Time    :   2021/05/23 14:46:37
@Author  :   Shanto Roy 
@Version :   1.0
@Contact :   sroy10@uh.edu
@License :   (C)Copyright 2020-2021, Shanto Roy
@Desc    :   This Script helps to visualize importance of selected features
'''


import pandas as pd
import streamlit as st
from numpy import set_printoptions
import plotly.express as px
import matplotlib.pyplot as plt



def feature_importance_plot(model,names):
    importance = {}

    for i,j in zip(names, list(model.feature_importances_)):
        importance[i] = j

    feature_importance = dict(sorted(importance.items(), key=lambda item: item[1], reverse=True))
    plot_df = pd.DataFrame(feature_importance.items(), columns=["Features", "Importance"])
    fig = px.bar(plot_df,
                    x = "Features",
                    y = "Importance")
    st.plotly_chart(fig)


# Feature Importance with Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

def random_forest_classifier(X,Y,col_names):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, Y)
    
    feature_importance_plot(model,col_names)

    
    
# Feature Importance with Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier

def extra_tree_classifier(X,Y,col_names):
    model = ExtraTreesClassifier(n_estimators=100)
    model.fit(X, Y)
    
    feature_importance_plot(model,col_names)
    


from xgboost import XGBClassifier

def xgboost(X,Y,col_names):
    model = XGBClassifier(random_state = 0)
    model.fit(X, Y)

    feature_importance_plot(model,col_names)

    


# primary interface for the App
def st_feature_selection():
    df = pd.read_csv("temp_data/test.csv")
    # drop object/string containing columns
    df_without_obj = df.select_dtypes(exclude=['object'])
    # add the label column once again
    df = pd.concat([df_without_obj, df["os"]], axis=1)

    consider_features = st.sidebar.selectbox(
        'Choose No. of Target Features', ["All", "Select Features"])

    if consider_features == "All":
        col_names = list(df.columns)
    if consider_features == "Select Features":
        col_names = []
        feature_list = list(df.columns)
        for col_name in feature_list:
            check_box = st.sidebar.checkbox(col_name)
            if check_box:
                col_names.append(col_name)
    

    df = df[col_names]
    st.write(df)

    # considering the last column as class labels
    array = df.values
    X = array[:,0:len(col_names)-1]
    Y = array[:,len(col_names)-1]

    select_method = st.sidebar.selectbox(
        'Select Feature Selection Method', ["Random Forest", "ExtraTree", "XGBoost"])


    if select_method == "Random Forest":
        try:
            random_forest_classifier(X,Y,col_names)
        except Exception as e:
            st.write(e)

    if select_method == "ExtraTree":
        try:
            extra_tree_classifier(X,Y,col_names)
        except Exception as e:
            st.write(e)

    if select_method == "XGBoost":
        try:
            xgboost(X,Y,col_names)
        except Exception as e:
            st.write(e)