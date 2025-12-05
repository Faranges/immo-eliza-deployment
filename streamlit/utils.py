# Custom Class for XGBPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

class FullXGBPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, log_target=True, random_state=888):
        self.log_target = log_target
        self.random_state = random_state
        
        # Placeholders for fitted transformers
        self.num_imputer = None
        self.state_encoder = None
        self.label_encoders = {}
        self.model = None
        self.feature_cols = None

    def fit(self, X, y):
        # 1) Handle log transform
        if self.log_target:
            self.y_log_mean = np.log1p(y)
            y_fit = np.log1p(y)
        else:
            y_fit = y

        X_train = X.copy()

        # 2) Impute numerical columns
        num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        self.num_imputer = SimpleImputer(strategy="mean")
        X_train[num_cols] = self.num_imputer.fit_transform(X_train[num_cols])

        # 3) Impute categorical 'state_of_building'
        X_train['state_of_building'] = X_train['state_of_building'].fillna('unknown')

        # 4) Ordinal encode 'state_of_building'
        state_order = [["unknown","To demolish","Under construction","To restore",
                        "To renovate","To be renovated","Normal","Fully renovated",
                        "Excellent","New"]]
        self.state_encoder = OrdinalEncoder(categories=state_order)
        X_train['state_of_building_oe'] = self.state_encoder.fit_transform(
            X_train[['state_of_building']]
        ).flatten()

        # 5) Label encode other categorical columns
        cat_cols = ['type', 'subtype', 'province']
        for col in cat_cols:
            mapping = {cat: idx for idx, cat in enumerate(X_train[col].astype(str).unique())}
            X_train[col + '_le'] = X_train[col].astype(str).map(mapping)
            self.label_encoders[col] = mapping


        # 6) Drop original categorical columns
        X_train = X_train.drop(columns=['type','subtype','state_of_building','province'])

        self.feature_cols = X_train.columns.tolist()

        # 7) Fit XGBoost
        self.model = XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=8,
            subsample=1.0,
            colsample_bytree=0.6,
            random_state=self.random_state,
            n_jobs=-1,
            min_child_weight=5
        )
        self.model.fit(X_train, y_fit)
        return self

    def transform(self, X):
        X_trans = X.copy()
        # 1) Impute numeric
        num_cols = X_trans.select_dtypes(include=["int64","float64"]).columns.tolist()
        X_trans[num_cols] = self.num_imputer.transform(X_trans[num_cols])

        # 2) Categorical 'state_of_building'
        X_trans['state_of_building'] = X_trans['state_of_building'].fillna('unknown')
        X_trans['state_of_building_oe'] = self.state_encoder.transform(
            X_trans[['state_of_building']]
        ).flatten()

        # 3) Label encode others
        
        for col, mapping in self.label_encoders.items():
            X_trans[col + '_le'] = X_trans[col].astype(str).map(mapping).fillna(-1).astype(int)

        X_trans = X_trans.drop(columns=['type','subtype','state_of_building','province'])
        return X_trans[self.feature_cols]

    

    def predict(self, X):
        X_trans = self.transform(X)
        y_pred = self.model.predict(X_trans)
        if self.log_target:
            return np.expm1(y_pred)  # inverse log1p
        else:
            return y_pred

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        metrics = {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred)
        }
        return metrics

