#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   app.py
@Time    :   2021/05/16 18:11:21
@Author  :   Shanto Roy 
@Version :   1.0
@Contact :   sroy10@uh.edu
@License :   (C)Copyright 2020-2021, Shanto Roy
@Desc    :   None
'''


import streamlit as st
import pandas as pd
import cufflinks as cf
import plotly
import plotly.graph_objs as go
import json
import ast
import os
import time

import sys
sys.path.append('scripts/')

from visualization import st_data_visualization
from feature_select import st_feature_selection
from classification import st_classification

def try_read_df(f, f_name):
    filename, file_extension = os.path.splitext(f_name)
    try:
        if file_extension.startswith(".xls"):
            return pd.read_excel(f)
        elif file_extension.startswith(".csv"):
            return pd.read_csv(f)
        else:
            st.write("File Type did not match")
    except Exception as e:
        st.write(e)


def main():
    # SideBar Settings
    st.sidebar.title("Control Panel")
    st.sidebar.info(
            "Test"
        )

    # app functionalities
    primary_function = st.sidebar.selectbox(
        'Choose App Functionality', ["Upload CSV File", "Data Visualization", \
                    "Data Cleanup", "Feature Selection", "Classification"])

    if primary_function == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader("Upload a CSV/Excel file", accept_multiple_files=False,\
                                                                        type=("csv", "xls"))

        if uploaded_file is not None:
            data = try_read_df(uploaded_file, uploaded_file.name)
            st.write("Here are the first ten rows of the File")
            st.table(data.head(10))
            file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,\
                                                        "FileSize":uploaded_file.size}
            st.sidebar.write(file_details)
            with open(os.path.join("temp_data", "test.csv"), "wb") as f:
                f.write(uploaded_file.getbuffer())

    if primary_function == "Data Visualization":
        st_data_visualization()
    if primary_function == "Feature Selection":
        st_feature_selection()
    if primary_function == "Classification":
        st_classification()

if __name__ == '__main__':
    main()