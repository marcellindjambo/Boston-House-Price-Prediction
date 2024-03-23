# Real Estate Prices Prediction Model at Boston

## Real Estate Prices Prediction Model at Boston

## Project Overview

The purpose of this project is to develop a machine learning model to predict housing prices in the city of Boston. The project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology, which is a widely used approach for data mining and machine learning projects.

## Data Understanding

The housing_data.csv file contains information on housing in Boston. The data includes various characteristics of housing such as crime rate per capita, proportion of residential land over 25,000 sq.ft., etc. The goal is to understand the meaning of each variable and its potential impact on housing prices.

## Data Preparation

Data preparation involves cleaning, transforming, and preparing the data for model training. The steps include converting data to numeric format, standardization, and dividing the data into training and testing sets.

## Modeling

In this phase, several machine learning models are built and evaluated for their ability to predict housing prices. Models include RandomForestRegressor, DecisionTreeRegressor and KNeighborsRegressor. Hyperparameters are optimized using a grid search.

## Evaluation

Models are evaluated using appropriate metrics such as performance score. Models that perform best are selected for predicting housing prices.

## Deployment

A Streamlit application is created to allow users to predict housing prices in Boston using the developed model.

- To test app:
    streamlit run streamlit_app.py