import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


train_original = pd.read_csv('dataset/train.csv')

test_original = pd.read_csv('dataset/test.csv')

full_data = pd.concat([train_original, test_original], axis=0)


def data_split(df, test_size):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


train_original, test_original = data_split(full_data, 0.2)

train_copy = train_original.copy()
test_copy = test_original.copy()

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


# Apply the pipeline to the DataFrame
pd.options.mode.chained_assignment = None
train_copy_prep = pipeline.fit_transform(train_copy)

test_copy_prep = pipeline.fit_transform(test_copy)


X_cost_amt_train_prep, y_cost_amt_train_prep = train_copy_prep.loc[:, train_copy_prep.columns != 'TOTAL COST TO HOSPITAL'], train_copy_prep['TOTAL COST TO HOSPITAL']
X_cost_amt_test_prep, y_cost_amt_test_prep = test_copy_prep.loc[:, test_copy_prep.columns != 'TOTAL COST TO HOSPITAL'], test_copy_prep['TOTAL COST TO HOSPITAL']

X_cost_amt_train_prep.shape
X_cost_amt_test_prep.shape


####################### Classes used to preprocess the data ##############################

# Create function to create line plot of predicted values vs actual values
def graph_test_accuracy(y_test, y_pred, mse):
    plt.figure(figsize=(6,4))
    x_ax = range(len(y_test)) # determine range of x-axis based of len of y_test
    plt.plot(x_ax, y_test, linewidth=1, label="original", alpha=0.9)
    plt.plot(x_ax, y_pred, linewidth=1.1, label="predicted", alpha=0.9)
    plt.title(f"y-test and y-predicted data (mae = {mae:.2f})")
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(fancybox=True, shadow=True, loc='best')
    plt.grid(True)
    plt.show()

 # add scatterplot that plots variance between predicted and actual values   
    plt.figure(figsize=(6,4))   
    variance=y_test-prediction # create dataframe of the difference between actual values and prediction
    sns.scatterplot(variance)
    plt.axhline(y = 0, color = 'r', linestyle = '--') # horizantal line that indicates where there is 0 variance
    plt.title('Difference between Predictions and Actual')
    plt.ylabel('')
    plt.xlabel('') 
    plt.show()


# initialize Extra trees regression  with default parameters
model_extra_trees=ExtraTreesRegressor(max_depth=20, min_samples_split=2,
                                      min_samples_leaf=1, n_estimators=800, random_state=42)

# fit model with training data
model_extra_trees.fit(X_cost_amt_train_prep, y_cost_amt_train_prep)

# make predictions and calculate mae
prediction=model_extra_trees.predict(X_cost_amt_test_prep)
mae=mean_absolute_error(y_cost_amt_test_prep, prediction)
mae
graph_test_accuracy(y_cost_amt_test_prep, prediction, mae)

joblib.dump(model_extra_trees, "extra_trees_model.sav")

#########################################################################################

# initialize Random forest regression  with default parameters
model_random_forest=RandomForestRegressor(max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=1000,
                      random_state=42)

# fit model with training data
model_random_forest.fit(X_cost_amt_train_prep, y_cost_amt_train_prep)

# make predictions and calculate mae
prediction=model_random_forest.predict(X_cost_amt_test_prep)
mae=mean_absolute_error(y_cost_amt_test_prep, prediction)
mae
graph_test_accuracy(y_cost_amt_test_prep, prediction, mae)

joblib.dump(model_random_forest, "random_forest_model.sav")

#########################################################################################
# Averaging two models (Extra trees regressor, Random Forest Regressor)

# initialize Extra trees regression  with default parameters
model_extra_trees=ExtraTreesRegressor(max_depth=20, min_samples_split=2,
                                      min_samples_leaf=1, n_estimators=800, random_state=42)
model_random_forest=RandomForestRegressor(max_depth=20, min_samples_leaf=2, min_samples_split=10, n_estimators=1000,
                      random_state=42)

# Train all models
model_extra_trees.fit(X_cost_amt_train_prep, y_cost_amt_train_prep)
model_random_forest.fit(X_cost_amt_train_prep, y_cost_amt_train_prep)

prediction_1=model_extra_trees.predict(X_cost_amt_test_prep)
prediction_2=model_random_forest.predict(X_cost_amt_test_prep)

pred_final = (prediction_1 + prediction_2)/2
avg_mae = mean_absolute_error(y_cost_amt_test_prep, pred_final)
avg_mae

graph_test_accuracy(y_cost_amt_test_prep, pred_final, avg_mae)
#########################################################################