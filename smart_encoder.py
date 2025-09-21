import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class SmartCategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, model_type='tree', ordinal_map=None, target_col=None,
                 max_categories=10, n_splits=5, smoothing=10, verbose=True,
                 bool_strategy='label'):
        self.model_type = model_type
        self.ordinal_map = ordinal_map or {}
        self.target_col = target_col
        self.max_categories = max_categories
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.verbose = verbose
        self.bool_strategy = bool_strategy
        self.encoders = {}
        self.encoding_strategy = {}
        self.report = {}
        self.global_mean = None

    def fit(self, X, y=None):
        X = X.copy()

        # Convert categorical to string
        for col in X.select_dtypes(include='category').columns:
            X[col] = X[col].astype(str)

        # Convert bool to string if needed
        for col in X.select_dtypes(include='bool').columns:
            if self.bool_strategy != 'ignore':
                X[col] = X[col].astype(str)

        if self.target_col and y is None:
            y = X[self.target_col]

        if y is not None:
            self.global_mean = y.mean()

        # Handle object and bool columns
        for col in X.select_dtypes(include=['object']).columns:
            n_unique = X[col].nunique()
            strategy = None

            if col in self.ordinal_map:
                strategy = 'ordinal'

            elif n_unique == 2 and self.bool_strategy in ['label', 'onehot']:
                strategy = self.bool_strategy
                if strategy == 'label':
                    le = LabelEncoder()
                    le.fit(X[col].astype(str))
                    self.encoders[col] = le

            elif self.model_type == 'linear' and n_unique <= self.max_categories:
                strategy = 'onehot'

            elif self.model_type == 'tree' and self.target_col:
                strategy = 'target_kfold'
                self.encoders[col] = self._fit_kfold_encoding(X[col].astype(str), y)

            elif self.model_type == 'tree':
                strategy = 'frequency'
                self.encoders[col] = X[col].value_counts()

            else:
                strategy = 'label'
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.encoders[col] = le

            if strategy:
                self.encoding_strategy[col] = strategy
                self.report[col] = {
                    'strategy': strategy,
                    'unique_categories': int(n_unique)
                }

        if self.verbose:
            self._print_report()

        return self

    def transform(self, X):
        X = X.copy()

        for col in X.select_dtypes(include='category').columns:
            X[col] = X[col].astype(str)

        for col in X.select_dtypes(include='bool').columns:
            if self.bool_strategy != 'ignore':
                X[col] = X[col].astype(str)

        for col, method in self.encoding_strategy.items():
            if method == 'ordinal':
                X[col] = X[col].map(self.ordinal_map[col])
            elif method == 'onehot':
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = X.drop(columns=[col])
                X = pd.concat([X, dummies], axis=1)
            elif method == 'frequency':
                X[col] = X[col].map(self.encoders[col]).fillna(0)
            elif method == 'target_kfold':
                X[col] = X[col].astype(str).map(self.encoders[col]).fillna(self.global_mean)
            elif method == 'label':
                X[col] = self.encoders[col].transform(X[col].astype(str))

        return X

    def _fit_kfold_encoding(self, col_series, y):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        temp_encoded = pd.Series(index=col_series.index, dtype=float)
        global_mean = y.mean()

        for train_idx, val_idx in kf.split(col_series):
            train_col = col_series.iloc[train_idx]
            train_y = y.iloc[train_idx]
            val_col = col_series.iloc[val_idx]

            df = pd.DataFrame({'col': train_col, 'target': train_y})
            agg = df.groupby('col')['target'].agg(['mean', 'count'])
            smooth = (agg['mean'] * agg['count'] + global_mean * self.smoothing) / (agg['count'] + self.smoothing)

            temp_encoded.iloc[val_idx] = val_col.map(smooth)

        final_df = pd.DataFrame({'col': col_series, 'target': y})
        agg = final_df.groupby('col')['target'].agg(['mean', 'count'])
        final_smooth = (agg['mean'] * agg['count'] + global_mean * self.smoothing) / (agg['count'] + self.smoothing)

        return final_smooth

    def _print_report(self):
        print("📋 Encoder Report :" + 80 * '=')
        for col, info in self.report.items():
            print(f" Column '{col}':")
            print(f" Encoder Method : {info['strategy']}")
            print(f" Number of Unique Categories: {info['unique_categories']}")
        print("End Report" + 80 * '=')
