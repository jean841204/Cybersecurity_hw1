import numpy as np
import pandas as pd
from typing import Tuple, Dict


class DataGenerator:
    """
    數據生成模組，用於創建線性回歸的合成數據
    """

    def __init__(self):
        self.x_data = None
        self.y_data = None
        self.true_slope = None
        self.true_intercept = None
        self.noise_level = None

    def generate_linear_data(self,
                           n_points: int = 100,
                           slope: float = 1.0,
                           intercept: float = 0.0,
                           noise_level: float = 1.0,
                           x_range: Tuple[float, float] = (-10, 10),
                           random_seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成線性回歸數據

        Parameters:
        -----------
        n_points : int
            數據點數量
        slope : float
            線性方程的斜率 (a)
        intercept : float
            線性方程的截距 (b)
        noise_level : float
            雜訊水平
        x_range : tuple
            X值的範圍
        random_seed : int
            隨機種子，確保結果可重現

        Returns:
        --------
        tuple : (x_data, y_data)
            生成的X和Y數據
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # 生成X值（等間距分佈）
        x_data = np.linspace(x_range[0], x_range[1], n_points)

        # 計算理論Y值
        y_theoretical = slope * x_data + intercept

        # 添加高斯雜訊
        noise = np.random.normal(0, noise_level, n_points)
        y_data = y_theoretical + noise

        # 存儲參數
        self.x_data = x_data
        self.y_data = y_data
        self.true_slope = slope
        self.true_intercept = intercept
        self.noise_level = noise_level

        return x_data, y_data

    def add_noise(self, y_data: np.ndarray, noise_level: float) -> np.ndarray:
        """
        為現有數據添加雜訊

        Parameters:
        -----------
        y_data : np.ndarray
            原始Y數據
        noise_level : float
            雜訊水平

        Returns:
        --------
        np.ndarray : 添加雜訊後的Y數據
        """
        noise = np.random.normal(0, noise_level, len(y_data))
        return y_data + noise

    def data_summary(self) -> Dict:
        """
        生成數據統計摘要

        Returns:
        --------
        dict : 包含統計信息的字典
        """
        if self.x_data is None or self.y_data is None:
            return {"error": "No data generated yet"}

        summary = {
            "data_points": len(self.x_data),
            "x_statistics": {
                "mean": float(np.mean(self.x_data)),
                "std": float(np.std(self.x_data)),
                "min": float(np.min(self.x_data)),
                "max": float(np.max(self.x_data)),
                "range": float(np.max(self.x_data) - np.min(self.x_data))
            },
            "y_statistics": {
                "mean": float(np.mean(self.y_data)),
                "std": float(np.std(self.y_data)),
                "min": float(np.min(self.y_data)),
                "max": float(np.max(self.y_data)),
                "range": float(np.max(self.y_data) - np.min(self.y_data))
            },
            "parameters": {
                "true_slope": self.true_slope,
                "true_intercept": self.true_intercept,
                "noise_level": self.noise_level
            },
            "correlation": float(np.corrcoef(self.x_data, self.y_data)[0, 1])
        }

        return summary

    def get_dataframe(self) -> pd.DataFrame:
        """
        將數據轉換為pandas DataFrame

        Returns:
        --------
        pd.DataFrame : 包含X和Y數據的DataFrame
        """
        if self.x_data is None or self.y_data is None:
            return pd.DataFrame()

        return pd.DataFrame({
            'X': self.x_data,
            'Y': self.y_data
        })

    def check_data_quality(self) -> Dict:
        """
        檢查數據品質

        Returns:
        --------
        dict : 數據品質報告
        """
        if self.x_data is None or self.y_data is None:
            return {"error": "No data generated yet"}

        # 檢查缺失值
        x_missing = np.isnan(self.x_data).sum()
        y_missing = np.isnan(self.y_data).sum()

        # 檢查異常值（使用IQR方法）
        def detect_outliers(data):
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            return outliers, lower_bound, upper_bound

        x_outliers, x_lower, x_upper = detect_outliers(self.x_data)
        y_outliers, y_lower, y_upper = detect_outliers(self.y_data)

        quality_report = {
            "missing_values": {
                "x_missing": int(x_missing),
                "y_missing": int(y_missing)
            },
            "outliers": {
                "x_outliers": int(x_outliers),
                "y_outliers": int(y_outliers),
                "x_bounds": (float(x_lower), float(x_upper)),
                "y_bounds": (float(y_lower), float(y_upper))
            },
            "data_validation": {
                "x_range_valid": bool(np.all(np.isfinite(self.x_data))),
                "y_range_valid": bool(np.all(np.isfinite(self.y_data))),
                "length_match": len(self.x_data) == len(self.y_data)
            }
        }

        return quality_report