#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   visualization.py
@Time    :   2021/05/16 18:12:14
@Author  :   Shanto Roy 
@Version :   1.0
@Contact :   sroy10@uh.edu
@License :   (C)Copyright 2020-2021, Shanto Roy
@Desc    :   None
'''


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import cufflinks as cf


class one_feature:
    def __init__(self, df, x_col_name):
        self.df = df 
        self.x_col_name = x_col_name


    def bar_plot(self):
        #labels
        key = self.df[self.x_col_name].value_counts().keys().tolist()
        #values
        val = self.df[self.x_col_name].value_counts().values.tolist()
        trace = go.Bar(x = key, y=val,\
                marker=dict(color=val,colorscale='Viridis',showscale=True),text = val)
        data=[trace]
        fig = go.Figure(data=data)
        st.plotly_chart(fig)


    def pi_plot(self):
        #labels
        key = self.df[self.x_col_name].value_counts().keys().tolist()
        #values
        val = self.df[self.x_col_name].value_counts().values.tolist()
        trace = go.Pie(labels=key, 
                values=val, 
                marker=dict(colors=['red']), 
                # Seting values to 
                hoverinfo="value"
              )
        data = [trace]
        fig = go.Figure(data = data)
        st.plotly_chart(fig)

    # def histogram_plot(self):
    #     fig = px.histogram(
    #             data_frame = self.df,
    #             x = self.x_col_name
    #         )
    #     st.plotly_chart(fig)

    def histogram_plot(self):
        # defining data
        trace = go.Histogram(x=self.df[self.x_col_name],nbinsx=40,histnorm='percent')
        data = [trace]
        fig = go.Figure(data = data)
        st.plotly_chart(fig)



class two_features:
    def __init__(self, df, x_col_name, y_col_name):
        self.df = df 
        self.x_col_name = x_col_name
        self.y_col_name = y_col_name

    def box_plot(self):
        fig = px.box(self.df, x = self.x_col_name, y = self.y_col_name)
        st.plotly_chart(fig)

    def violin_plot(self):
        fig = px.violin(self.df, x = self.x_col_name, y = self.y_col_name)
        st.plotly_chart(fig)

    def scatter_plot(self):
        fig = px.scatter(self.df, x = self.x_col_name, y = self.y_col_name, color = self.y_col_name, \
                            color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig)

    def bar_plot(self):
        self.df = self.df.groupby([self.x_col_name,self.y_col_name]).size().reset_index(name='quantity')
        fig = px.bar(self.df,
                    x = self.x_col_name,
                    y = 'quantity',
                    color = self.y_col_name,
                    barmode = 'stack')
        st.plotly_chart(fig)

    


    



class three_features:
    def __init__(self, df, x_col_name, y_col_name, category_col_name):
        self.df = df 
        self.x_col_name = x_col_name
        self.y_col_name = y_col_name
        self.category_col_name = category_col_name

    def scatter_plot(self):
        fig = px.scatter(self.df, x=self.x_col_name, y=self.y_col_name, \
                            color=self.category_col_name)
        st.plotly_chart(fig)

    def line_plot(self):
        fig = px.line(
                data_frame=self.df,
                x = self.x_col_name,
                y = self.y_col_name,
                color = self.category_col_name
            )
        st.plotly_chart(fig)




def st_data_visualization():
    # original saved database -> test.csv
    df = pd.read_csv("temp_data/test.csv")
    # for code testing -> 5000_sales_records.csv
    # df = pd.read_csv("temp_data/5000_sales_records.csv")
    column_list = df.columns.values.tolist()

    target_feature_no = st.sidebar.selectbox(
        'Choose No. of Target Features', ["One", "Two", "Three", "All"])
    
    if target_feature_no == 'One':
        st.sidebar.write("Choose One Column")
        x_col_name = st.sidebar.selectbox('Select X column', column_list)

        plot_list = ["bar", "pi", "histogram"]
        plot_type = st.sidebar.selectbox('Select Plot Type', plot_list)

        plot = one_feature(df, x_col_name)
        if plot_type == "bar":
            plot.bar_plot()
        if plot_type == "pi":
            plot.pi_plot()
        if plot_type == "histogram":
            plot.histogram_plot()
        

    if target_feature_no == 'Two':    
        st.sidebar.write("Choose Two Columns for Viewing Relationships")
        x_col_name = st.sidebar.selectbox('Select X column', column_list)
        y_col_name = st.sidebar.selectbox('Select Y column', column_list)

        plot_list = ["box", "violin", "scatter", "bar"]
        plot_type = st.sidebar.selectbox('Select Plot Type', plot_list)

        plot = two_features(df, x_col_name, y_col_name)
        if plot_type == "box":
            plot.box_plot()
        if plot_type == "violin":
            plot.violin_plot()
        if plot_type == "scatter":
            plot.scatter_plot()
        if plot_type == "bar":
            plot.bar_plot()
        



    if target_feature_no == 'Three':    
        st.sidebar.write("Choose Two Columns for Viewing Relationships")
        x_col_name = st.sidebar.selectbox('Select X column', column_list)
        y_col_name = st.sidebar.selectbox('Select Y column', column_list)

        st.sidebar.write("Choose Category Column")
        category_col_name = st.sidebar.selectbox('Select Category', column_list)

        plot_list = ["scatter", "line"]
        plot_type = st.sidebar.selectbox('Select Plot Type', plot_list)

        plot = three_features(df, x_col_name, y_col_name, category_col_name)
        if plot_type == "scatter":
            plot.scatter_plot()
        if plot_type == "line":
            plot.line_plot()