# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:38:42 2024

@author: mzing
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import tempfile
import streamlit as st
import json
import requests


train_original = pd.read_csv('dataset/train.csv')
test_original = pd.read_csv('dataset/test.csv')

full_data = pd.concat([train_original, test_original], sort=False).reset_index(drop=True)

def data_split(df, test_size):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


train_original, test_original = data_split(full_data, 0.2)

train_copy = train_original.copy()
test_copy = test_original.copy()

####################### Classes used to preprocess the data ##############################

class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_drop=None):
        """
        Custom transformer to drop specified features from a DataFrame.

        Parameters:
            feature_to_drop (list or None): List of feature names to drop.
        """
        self.feature_to_drop = feature_to_drop or []

    def fit(self, X, y=None):
        """
        No fitting required. Returns self.

        Parameters:
            X (pd.DataFrame): Input data (ignored).
            y (pd.Series or None): Target labels (ignored).

        Returns:
            self
        """
        return self

    def transform(self, X):
        """
        Drops specified features from the DataFrame.

        Parameters:
            X (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Transformed data with specified features dropped.
        """
        missing_features = set(self.feature_to_drop) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")
        return X.drop(columns=self.feature_to_drop)
        

class OneHotWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_enc_ft=None):
        """
        Custom transformer to perform one-hot encoding on specified features in a DataFrame.

        Parameters:
            one_hot_enc_ft (list or None): List of feature names to one-hot encode.
        """
        self.one_hot_enc_ft = one_hot_enc_ft or []
        self.one_hot_enc = OneHotEncoder(handle_unknown='ignore')

    def fit(self, X, y=None):
        """
        Fit the one-hot encoder to the specified features.

        Parameters:
            X (pd.DataFrame): Input data.
            y (pd.Series or None): Target labels (ignored).

        Returns:
            self
        """
        self.one_hot_enc.fit(X[self.one_hot_enc_ft])
        self.feat_names_one_hot_enc = self.one_hot_enc.get_feature_names_out(self.one_hot_enc_ft)
        return self

    def transform(self, X):
        """
        Applies one-hot encoding to specified features and concatenates with the remaining features.

        Parameters:
            X (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Transformed data with one-hot encoded features.
        """
        missing_features = set(self.one_hot_enc_ft) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")

        one_hot_enc_df = pd.DataFrame(self.one_hot_enc.transform(X[self.one_hot_enc_ft]).toarray(), columns=self.feat_names_one_hot_enc, index=X.index)
        rest_of_features = [ft for ft in X.columns if ft not in self.one_hot_enc_ft]
        df_concat = pd.concat([one_hot_enc_df, X[rest_of_features]], axis=1)
        return df_concat


        
class MinMaxWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler_ft=None):
        """
        Custom transformer to apply Min-Max scaling to specified features in a DataFrame.

        Parameters:
            min_max_scaler_ft (list or None): List of feature names to scale.
        """
        self.min_max_scaler_ft = min_max_scaler_ft or []

    def fit(self, X, y=None):
        """
        No fitting required. Returns self.

        Parameters:
            X (pd.DataFrame): Input data (ignored).
            y (pd.Series or None): Target labels (ignored).

        Returns:
            self
        """
        return self

    def transform(self, X):
        """
        Applies Min-Max scaling to specified features in the DataFrame.

        Parameters:
            X (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Transformed data with specified features scaled.
        """
        missing_features = set(self.min_max_scaler_ft) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")

        min_max_enc = MinMaxScaler()
        X[self.min_max_scaler_ft] = min_max_enc.fit_transform(X[self.min_max_scaler_ft])
        return X
              
        
# Create the pipeline
pipeline = Pipeline([
    ('drop features', DropFeatures(feature_to_drop=['SL.', 'AGE', 'KEY COMPLAINTS -CODE', 'BODY WEIGHT', 'BODY HEIGHT', 'HR PULSE', 'BP -HIGH', 'BP-LOW', 'RR',
    'PAST MEDICAL HISTORY CODE', 'HB', 'UREA', 'CREATININE', 'STATE AT THE TIME OF ARRIVAL', 'IMPLANT USED (Y/N)', 'HB_LEVEL', 'UREA_LEVEL', 'BMI',
    'CREATININE_LEVEL'])),
    ('one hot encoder', OneHotWithFeatNames(one_hot_enc_ft=['GENDER', 'MARITAL STATUS', 'MODE OF ARRIVAL', 'TYPE OF ADMSN', 'AGE_GROUP', 'BP_Cat'])),
    ('min max scaler', MinMaxWithFeatNames(min_max_scaler_ft=['TOTAL LENGTH OF STAY',
    'LENGTH OF STAY - ICU', 'LENGTH OF STAY- WARD', 'COST OF IMPLANT', 'BMI_VALUE'])),
])

############################# Streamlit interface ######################################


st.markdown("""
# 🏥 Hospital Treatment Cost Prediction using Machine Learning

Curious about your potential hospital expenses? You're in the right place! Simply provide the requested information and hit the **Predict** button. Our advanced ML model will estimate your treatment costs. 💰💉🌡️
""")


# Gender input
st.write("""
## Gender
""")
input_gender = st.radio('Select Gender', ['M', 'F'])


# Marital Status input
st.write("""
## Marital Status
""")
input_marital_status = st.radio('Select Marital Status', ['Married', 'Unmarried'])



# age group input
st.write("""
## Age Group
""")
input_age_group = st.radio('Select Age Group', ['Child', 'Youngadult', 'Adult', 'Old'],
                           help="age <=10 - Child, >10 age <=25 - Youngadult, >26 age <=50 - Adult, age >=50 - Old.")


# Complaints Code dropdown
st.write("""
## BP Category
""")
input_bp_cat_key = st.selectbox('Select Key Complaints Code', ('Normal','Hypertension Stage 1','Hypertension Stage 2','Hypertensive Crisis','Elevated'))


# bmi input number
st.write("""
## BMI
""")
input_bmi = st.slider('Enter BMI', value=23.3,
                      min_value=9.7, max_value=404.4, step=0.1)


# arrival mode input
st.write("""
## Arrival Mode
""")
input_arrival_mode = st.radio('Select Mode of Arrival', ['Walked in', 'Ambulance', 'Transferred'])



# type of admission input
st.write("""
## Type of Admission
""")
input_admission = st.radio('Select Type of Admission', ['Elective', 'Emergency'])


# total stay input number
st.write("""
## Total Length of Stay
""")
input_total_length_of_stay = st.slider('Select Total Length of Stay', value=12,
                      min_value=3, max_value=50, step=1)


# icu stay input number
st.write("""
## Length of Stay - ICU
""")
input_length_of_stay_icu = st.slider('Select Length of Stay - ICU', value=3,
                      min_value=0, max_value=30, step=1)


# ward stay input number
st.write("""
## Length of Stay - Ward
""")
input_length_of_stay_ward = st.slider('Select Length of Stay - Ward', value=8,
                      min_value=0, max_value=22, step=1)



# implant cost input number
st.write("""
## Implant Cost
""")
input_cost_of_implant = st.number_input('Enter Cost of Implant',
                      min_value=0)

# list of all the inputs
profile_to_predict = [0, 0.0, input_gender, input_marital_status, '', 0.0, 0, 0, 0.0, 0.0, 0, '',
0.0, 0.0, 0.0, input_arrival_mode, '', input_admission, 0.0, 
input_total_length_of_stay, input_length_of_stay_icu, input_length_of_stay_ward, '', input_cost_of_implant, input_age_group, '', '', input_bmi, '', input_bp_cat_key, '']


# Convert to dataframe with column names
profile_to_predict_df = pd.DataFrame(
    [profile_to_predict], columns=train_copy.columns)


# add the profile to predict as a last row in the train data
train_copy_with_profile_to_pred = pd.concat(
    [train_copy, profile_to_predict_df])


# whole dataset prepared
profile_to_pred_prep = pipeline.fit_transform(
    train_copy_with_profile_to_pred).tail(1).drop(columns=['TOTAL COST TO HOSPITAL'])



# load the model from the temporary file
def load_model():
    try:
        model = joblib.load("gb_model.sav")
        return model
    except FileNotFoundError:
        st.error("Model file 'gb_model.sav' not found. Please check the file path.")
        return None

# Predict button
if st.button('Predict'):
    with st.spinner('Predicting...'):
        loaded_model = load_model()
        if loaded_model:
            prediction = loaded_model.predict(profile_to_pred_prep)
            formatted_prediction = f"${prediction[0]:,.2f}"  # Format as currency
            st.markdown(f"**The predicted hospital treatment pricing is: {formatted_prediction}**", unsafe_allow_html=True)