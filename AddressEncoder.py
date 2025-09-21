from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

class NeighborhoodEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, location_col='neighborhood', target_col='price', year_col='year_built',
                 current_year=1404, n_splits=5, default_value='mean'):
        self.location_col = location_col
        self.target_col = target_col
        self.year_col = year_col
        self.current_year = current_year
        self.n_splits = n_splits
        self.default_value = default_value
        self.global_mean_ = None
        self.mapping_ = None
        self.encoded_ = None
        self.price_mean_ = None

    def fit(self, X, y=None):
        df = X.copy()
        if y is None:
            y = df[self.target_col]
        df['target'] = y.values

        # محاسبه سن ساختمان
        df['age'] = self.current_year - df[self.year_col]

        # ساخت نگاشت امن با KFold
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        encoded = pd.Series(index=df.index, dtype=float)

        for train_idx, val_idx in kf.split(df):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            # میانگین قیمت و سن ساختمان در داده‌های آموزش
            stats = train_df.groupby(self.location_col).agg({
                'target': 'mean',
                'age': 'mean'
            })

            # ترکیب قیمت و سن به‌صورت وزن‌دار یا ساده
            combined = stats['target'] - 0.1 * stats['age']  # وزن‌دهی قابل تنظیم

            encoded.iloc[val_idx] = val_df[self.location_col].map(combined)

        # نگاشت نهایی برای transform
        final_stats = df.groupby(self.location_col).agg({
            'target': 'mean',
            'age': 'mean'
        })
        self.mapping_ = final_stats['target'] - 0.1 * final_stats['age']
        self.global_mean_ = self.mapping_.mean()
        self.encoded_ = encoded.fillna(self.global_mean_)
        self.price_mean_ = final_stats['target']
        return self

    def transform(self, X):
        X = X.copy()
        if hasattr(self, 'encoded_') and len(X) == len(self.encoded_):
            X['neigh_encoded'] = self.encoded_.values
        else:
            encoded = X[self.location_col].map(self.mapping_)
            default = self.global_mean_ if self.default_value == 'mean' else 0
            X['neigh_encoded'] = encoded.fillna(default)
        return X[['neigh_encoded']]