import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

class ResidualOutlierAdvisor:
    def __init__(self, method='iqr', residual_threshold=2.5, strategy='weight', verbose=True):
        self.method = method  # 'iqr', 'zscore', 'model'
        self.residual_threshold = residual_threshold
        self.strategy = strategy  # 'drop', 'weight', 'cap'
        self.verbose = verbose
        self.feature_outliers = {}  # dict of feature-wise outlier masks
        self.residual_outliers = None
        self.weights = None

    def fit(self, X, residuals):
        df = X.copy()
        self.feature_outliers = {}

        # تحلیل آماری فیچرها
        for col in df.columns:
            if self.method == 'zscore':
                z = np.abs((df[col] - df[col].mean()) / df[col].std())
                self.feature_outliers[col] = z > 3.0
            elif self.method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                mask = (df[col] < Q1 - self.residual_threshold * IQR) | (df[col] > Q3 + self.residual_threshold * IQR)
                self.feature_outliers[col] = mask
            elif self.method == 'model':
                iso = IsolationForest(contamination=0.1, random_state=42)
                self.feature_outliers[col] = iso.fit_predict(df[[col]]) == -1

        # تحلیل ساختاری باقیمانده‌ها
        self.residual_outliers = np.abs(residuals) > self.residual_threshold * np.std(residuals)

        # ترکیب ماسک‌ها
        combined_mask = self.residual_outliers.copy()
        for mask in self.feature_outliers.values():
            combined_mask = combined_mask | mask

        if self.strategy == 'weight':
            self.weights = np.where(combined_mask, 0.5, 1.0)

        if self.verbose:
            print(f"🔍 Residual outliers: {np.sum(self.residual_outliers)}")
            for col, mask in self.feature_outliers.items():
                print(f"📊 Feature '{col}' outliers: {np.sum(mask)}")

    def transform(self, X, y=None):
        df = X.copy()
        combined_mask = self.residual_outliers.copy()
        for mask in self.feature_outliers.values():
            combined_mask = combined_mask | mask

        if self.strategy == 'drop':
            return df[~combined_mask], y[~combined_mask] if y is not None else None
        elif self.strategy == 'cap':
            for col in df.columns:
                upper = df[col][~combined_mask].quantile(0.99)
                lower = df[col][~combined_mask].quantile(0.01)
                df[col] = np.clip(df[col], lower, upper)
            return df, y
        elif self.strategy == 'weight':
            return df, y, self.weights
        else:
            return df, y

    def report(self):
        report_df = pd.DataFrame({col: mask for col, mask in self.feature_outliers.items()})
        report_df['residual_outlier'] = self.residual_outliers
        return report_df

    def plot_heatmap(self):
        report_df = self.report().astype(int)
        plt.figure(figsize=(10,6))
        sns.heatmap(report_df.T, cmap='Reds', cbar=False, linewidths=0.5)
        plt.title("Outlier Heatmap (Features + Residuals)")
        plt.xlabel("Samples")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.show()