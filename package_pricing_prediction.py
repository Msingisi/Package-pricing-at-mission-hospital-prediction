# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 22:11:14 2024

@author: mzing
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import joblib
import streamlit as st

train_original = pd.read_csv('dataset/train.csv')

test_original = pd.read_csv('dataset/test.csv')

full_data = pd.concat([train_original, test_original], axis=0)


def data_split(df, test_size):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


train_original, test_original = data_split(full_data, 0.2)

train_copy = train_original.copy()
test_copy = test_original.copy()


def value_cnt_norm_cal(df,feature):
    ftr_value_cnt = df[feature].value_counts()
    ftr_value_cnt_norm = df[feature].value_counts(normalize=True) * 100
    ftr_value_cnt_concat = pd.concat([ftr_value_cnt, ftr_value_cnt_norm], axis=1)
    ftr_value_cnt_concat.columns = ['Count', 'Frequency (%)']
    return ftr_value_cnt_concat

####################### Classes used to preprocess the data ##############################

class MissingValueImputer(BaseEstimator, TransformerMixin):
    def __init__(self, mode_imputed_ft=None, median_imputed_ft=None):

        self.mode_imputed_ft = mode_imputed_ft or []
        self.median_imputed_ft = median_imputed_ft or []

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        missing_features = set(self.mode_imputed_ft + self.median_imputed_ft) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")

        # Impute missing values with mode
        for ft in self.mode_imputed_ft:
            the_mode = X[ft].mode()[0]
            X[ft] = X[ft].fillna(the_mode)

        # Impute missing values with median
        for ft in self.median_imputed_ft:
            the_median = X[ft].median()
            X[ft] = X[ft].fillna(the_median)

        return X
    
class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_drop=None):

        self.feature_to_drop = feature_to_drop or []

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        missing_features = set(self.feature_to_drop) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")
        return X.drop(columns=self.feature_to_drop)
    
class MinMaxWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler_ft=None):

        self.min_max_scaler_ft = min_max_scaler_ft or []

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        missing_features = set(self.min_max_scaler_ft) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")

        min_max_enc = MinMaxScaler()
        X[self.min_max_scaler_ft] = min_max_enc.fit_transform(X[self.min_max_scaler_ft])
        return X
    
class OneHotWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_enc_ft=None):
        self.one_hot_enc_ft = one_hot_enc_ft or []
        self.one_hot_enc = OneHotEncoder(handle_unknown='ignore')

    def fit(self, X, y=None):
        self.one_hot_enc.fit(X[self.one_hot_enc_ft])
        self.feat_names_one_hot_enc = self.one_hot_enc.get_feature_names_out(self.one_hot_enc_ft)
        return self

    def transform(self, X):
        missing_features = set(self.one_hot_enc_ft) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")

        one_hot_enc_df = pd.DataFrame(self.one_hot_enc.transform(X[self.one_hot_enc_ft]).toarray(), columns=self.feat_names_one_hot_enc, index=X.index)
        rest_of_features = [ft for ft in X.columns if ft not in self.one_hot_enc_ft]
        df_concat = pd.concat([one_hot_enc_df, X[rest_of_features]], axis=1)
        return df_concat
    
class OrdinalEncoderWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_enc_ft=None):
        self.ordinal_enc_ft = ordinal_enc_ft or []
        self.ordinal_enc = OrdinalEncoder()

    def fit(self, X, y=None):
        self.ordinal_enc.fit(X[self.ordinal_enc_ft])
        return self

    def transform(self, X):
        missing_features = set(self.ordinal_enc_ft) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")

        ordinal_enc_df = pd.DataFrame(self.ordinal_enc.transform(X[self.ordinal_enc_ft]), columns=self.ordinal_enc_ft, index=X.index)
        rest_of_features = [ft for ft in X.columns if ft not in self.ordinal_enc_ft]
        df_concat = pd.concat([ordinal_enc_df, X[rest_of_features]], axis=1)
        return df_concat
    
class SkewnessHandler(BaseEstimator, TransformerMixin):
    def __init__(self, col_with_skewness=None):

        self.col_with_skewness = col_with_skewness or []

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        missing_features = set(self.col_with_skewness) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in DataFrame: {missing_features}")

        X[self.col_with_skewness] = np.cbrt(X[self.col_with_skewness])
        return X
    
class DropUncommonComplaint(BaseEstimator, TransformerMixin):
    def __init__(self, complaint_list=None):
 
        self.complaint_list = complaint_list or []

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        if 'KEY COMPLAINTS -CODE' in X.columns:
            X = X[~X['KEY COMPLAINTS -CODE'].isin(self.complaint_list)]
            return X
        else:
            print("KEY COMPLAINTS -CODE feature is not in the dataframe")
            return X
        
# Create the pipeline
pipeline = Pipeline([
    ('missing values imputer', MissingValueImputer(mode_imputed_ft=['KEY COMPLAINTS -CODE'], median_imputed_ft=['HB','UREA','CREATININE','BP -HIGH','BP-LOW'])),
    ('drop features', DropFeatures(feature_to_drop=['BP_Cat','IMPLANT USED (Y/N)','AGE_GROUP','LOG_TOTAL_COST_TO_HOSPITAL','SL.','BODY WEIGHT','BODY HEIGHT',
                                                   'PAST MEDICAL HISTORY CODE','BMI','STATE AT THE TIME OF ARRIVAL',
                                                    'CREATININE_LEVEL','HB_LEVEL','UREA_LEVEL'])),
    ('skewness handler', SkewnessHandler(col_with_skewness=['RR','CREATININE','BP -HIGH','BP-LOW','TOTAL LENGTH OF STAY','LENGTH OF STAY - ICU','BMI_VALUE','UREA','COST OF IMPLANT'])),
    ('min max scaler', MinMaxWithFeatNames(min_max_scaler_ft=['RR','CREATININE','BP -HIGH','BP-LOW','AGE','HR PULSE','HB','LENGTH OF STAY- WARD','UREA','BMI_VALUE','TOTAL LENGTH OF STAY','COST OF IMPLANT','LENGTH OF STAY - ICU'])),
    ('drop uncommon complaint', DropUncommonComplaint(complaint_list=['PM-VSD','Other nervous','CAD-SVD','CAD-VSD','Other general'])),
    ('one hot encoder', OneHotWithFeatNames(one_hot_enc_ft=['GENDER','MODE OF ARRIVAL','MARITAL STATUS','TYPE OF ADMSN','KEY COMPLAINTS -CODE']))
])


############################# Streamlit interface ############################

primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"


st.markdown("""
    <style>
    h1 {
        font-size: 24px;
        text-align: center;
        text-transform: uppercase;
        color: #F63366;
    }
    </style>
    """, unsafe_allow_html=True)
    
# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #F0F2F6;
    }
    .stDataFrame {
        border: 2px solid #F63366;
    }
    </style>
    """, unsafe_allow_html=True)
    

st.markdown("""
# 🏥 Hospital Treatment Pricing Prediction using Machine Learning

Curious about your potential hospital expenses? You're in the right place! Simply provide the requested information and hit the **Predict** button. Our advanced ML model will estimate your treatment costs. 💰💉🌡️
""")

# Age input slider
st.write("""
## Age
""")
input_age = st.slider('Select patient age', value=28, min_value=0, max_value=88, step=1)


#Gender input
st.write("""
## Gender
""")
input_gender = st.radio('Select patient gender',['Male','Female'], index=0)


#Mariral status input
st.write("""
## Marital status
""")
input_marital_status = st.radio('Select patient marital status',['Married','Unmarried'], index=0)


# Key complaints dropdown
st.write("""
## Key complaints
""")
complaints = ['Other heart', 'CAD-DVD', 'RHD', 'CAD-TVD', 'ACHD', 'Other tertalogy', 'Other respiratory', 'OS-ASD']
input_complaints = st.selectbox(
    'Select patient key Complaints Code', complaints)


# pulse input slider
st.write("""
## Heart rate pulse
""")
input_heart_rate = st.slider('Select patient Heart rate pulse', value=92, min_value=41, max_value=155, step=1)


# BP High input slider
st.write("""
## High BP
""")
input_bp_high = st.slider('Select patient BP High', value=115, min_value=70, max_value=215, step=1)


# BP Low input slider
st.write("""
## Low BP
""")
input_bp_low = st.slider('Select patient BP Low', value=72, min_value=39, max_value=140, step=1)


# rr input slider
st.write("""
## Respiratory rate 
""")
input_rr = st.slider('Select patient respiratory rate ', value=24,
                      min_value=12, max_value=42, step=1)


# hemoglobin input slider
st.write("""
## Hemoglobin
""")
input_hb = st.slider('Select patient hemoglobin count', value=12.9,
                      min_value=5.0, max_value=25.7, step=0.1)


# Urea input slider
st.write("""
## Urea levels
""")
input_urea = st.slider('Select patient urea levels', value=27,
                      min_value=2, max_value=143, step=1)


# creatinine input slider
st.write("""
## Creatinine levels
""")
input_creatiine = st.slider('Select patient creatinine levels',
                      min_value=0.1, max_value=5.2, step=0.1)


# Mode of arrival dropdown
st.write("""
## Mode of arrival
""")
mode = ['Walked in', 'Ambulance', 'Transferred']
input_mode = st.selectbox(
    'Select arrival mode', mode)


# Type of admission input
st.write("""
## Type of admission
""")
input_admission = st.radio('Select type of admission',['Elective','Emergency'], index=0)


# total stay input number
st.write("""
## Total length of stay
""")
input_stay = st.slider('Select total length of stay', value=12,
                      min_value=3, max_value=50, step=1)


# icu stay input number
st.write("""
## Length of stay in icu
""")
input_icu = st.slider('Select length of stay - ICU', value=3,
                      min_value=0, max_value=30, step=1)


# ward stay input number
st.write("""
## Length of Stay in ward
""")
input_ward = st.slider('Select length of stay - Ward', value=8,
                      min_value=0, max_value=22, step=1)


# implant cost input number
st.write("""
## Implant Cost
""")
input_implant = st.number_input('Enter cost of implant',
                      min_value=0)



# implant cost input number
st.write("""
## Body mass index
""")
input_bmi = st.number_input('Enter patient BMI',
                      min_value=9.72)




st.markdown('##')
st.markdown('##')


# list of all the inputs
profile_to_predict = [0, input_age, input_gender, input_marital_status, input_complaints, 0.0, 0, input_heart_rate, input_bp_high, input_bp_low, input_rr, '',
input_hb, input_urea, input_creatiine, input_mode, '', input_admission, 0.0, 
input_stay, input_icu, input_ward, '', input_implant, '', '', '', input_bmi, '', '', '', 0.0]


# Convert to dataframe with column names
profile_to_predict_df = pd.DataFrame(
    [profile_to_predict], columns=train_copy.columns)


# add the profile to predict as a last row in the train data
train_copy_with_profile_to_pred = pd.concat(
    [train_copy, profile_to_predict_df], ignore_index=True)


# whole dataset prepared
profile_to_pred_prep = pipeline.fit_transform(
    train_copy_with_profile_to_pred).tail(1).drop(columns=['TOTAL COST TO HOSPITAL'])


# load the model from the temporary file
def load_model():
    try:
        model = joblib.load("extra_trees_model.sav")
        return model
    except FileNotFoundError:
        st.error("Model file 'extra_trees_model.sav' not found. Please check the file path.")
        return None

# Predict button
if st.button('Predict'):
    with st.spinner('Predicting...'):
        loaded_model = load_model()
        if loaded_model:
            prediction = loaded_model.predict(profile_to_pred_prep)
            formatted_prediction = f"${prediction[0]:,.0f}"  # Format as currency
            st.markdown(f"**The predicted hospital treatment pricing is: {formatted_prediction}**", unsafe_allow_html=True)