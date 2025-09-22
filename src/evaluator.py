import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    模型評估模組，計算各種評估指標和診斷信息
    """

    def __init__(self):
        self.metrics_cache = {}

    def calculate_metrics(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         cache_key: str = None) -> Dict:
        """
        計算回歸模型評估指標

        Parameters:
        -----------
        y_true : np.ndarray
            真實值
        y_pred : np.ndarray
            預測值
        cache_key : str, optional
            緩存鍵值

        Returns:
        --------
        dict : 包含各種評估指標的字典
        """
        # 檢查緩存
        if cache_key and cache_key in self.metrics_cache:
            return self.metrics_cache[cache_key]

        # 確保數據格式正確
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()

        # 計算基本指標
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # 計算其他有用指標
        n = len(y_true)

        # 調整R²（考慮特徵數量，這裡假設為1個特徵）
        n_features = 1
        if n > n_features + 1:
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        else:
            adj_r2 = r2

        # 平均絕對百分比誤差（MAPE）
        mask = y_true != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.inf

        # 對稱平均絕對百分比誤差（SMAPE）
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

        # 殘差統計
        residuals = y_true - y_pred
        residual_std = np.std(residuals)
        residual_mean = np.mean(residuals)

        # 正規化均方根誤差（NRMSE）
        y_range = np.max(y_true) - np.min(y_true)
        nrmse = rmse / y_range if y_range != 0 else np.inf

        metrics = {
            "r2_score": float(r2),
            "adjusted_r2": float(adj_r2),
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "smape": float(smape),
            "nrmse": float(nrmse),
            "residual_stats": {
                "mean": float(residual_mean),
                "std": float(residual_std),
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals))
            },
            "data_stats": {
                "n_samples": int(n),
                "y_true_mean": float(np.mean(y_true)),
                "y_pred_mean": float(np.mean(y_pred)),
                "y_true_std": float(np.std(y_true)),
                "y_pred_std": float(np.std(y_pred))
            }
        }

        # 緩存結果
        if cache_key:
            self.metrics_cache[cache_key] = metrics

        return metrics

    def get_top_outliers(self,
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        x_values: np.ndarray,
                        n_outliers: int = 5) -> Dict:
        """
        Get top N outliers based on standardized residuals

        Parameters:
        -----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        x_values : np.ndarray
            X values
        n_outliers : int
            Number of top outliers to return

        Returns:
        --------
        dict : Dictionary containing outlier information
        """
        residuals = y_true - y_pred
        standardized_residuals = residuals / np.std(residuals) if np.std(residuals) != 0 else residuals
        outlier_scores = np.abs(standardized_residuals)

        # Get indices of top N outliers
        top_indices = np.argsort(outlier_scores)[-n_outliers:]

        return {
            'indices': top_indices.tolist(),
            'x_values': x_values[top_indices].tolist(),
            'y_values': y_true[top_indices].tolist(),
            'residuals': residuals[top_indices].tolist(),
            'scores': outlier_scores[top_indices].tolist()
        }

    def residual_analysis(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         x_values: np.ndarray = None) -> Dict:
        """
        殘差分析

        Parameters:
        -----------
        y_true : np.ndarray
            真實值
        y_pred : np.ndarray
            預測值
        x_values : np.ndarray, optional
            自變量值

        Returns:
        --------
        dict : 殘差分析結果
        """
        y_true = np.array(y_true).ravel()
        y_pred = np.array(y_pred).ravel()
        residuals = y_true - y_pred

        # 標準化殘差
        standardized_residuals = residuals / np.std(residuals) if np.std(residuals) != 0 else residuals

        # 正態性檢驗（Shapiro-Wilk）
        if len(residuals) >= 3:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan

        # Durbin-Watson統計量（檢驗自相關）
        def durbin_watson(residuals):
            diff = np.diff(residuals)
            return np.sum(diff**2) / np.sum(residuals**2)

        dw_stat = durbin_watson(residuals)

        # 異常值檢測（使用標準化殘差）
        outlier_threshold = 2.5
        outliers = np.abs(standardized_residuals) > outlier_threshold
        n_outliers = np.sum(outliers)

        analysis = {
            "residuals": residuals.tolist(),
            "standardized_residuals": standardized_residuals.tolist(),
            "normality_test": {
                "shapiro_stat": float(shapiro_stat) if not np.isnan(shapiro_stat) else None,
                "shapiro_p_value": float(shapiro_p) if not np.isnan(shapiro_p) else None,
                "is_normal": bool(shapiro_p > 0.05) if not np.isnan(shapiro_p) else None
            },
            "autocorrelation": {
                "durbin_watson": float(dw_stat),
                "interpretation": self._interpret_dw(dw_stat)
            },
            "outliers": {
                "count": int(n_outliers),
                "percentage": float(n_outliers / len(residuals) * 100),
                "indices": np.where(outliers)[0].tolist(),
                "threshold": float(outlier_threshold)
            },
            "homoscedasticity": self._test_homoscedasticity(residuals, y_pred)
        }

        if x_values is not None:
            x_values = np.array(x_values).ravel()
            analysis["residuals_vs_fitted"] = {
                "x_values": x_values.tolist(),
                "residuals": residuals.tolist()
            }

        return analysis

    def _interpret_dw(self, dw_stat: float) -> str:
        """解釋Durbin-Watson統計量"""
        if dw_stat < 1.5:
            return "Positive autocorrelation likely"
        elif dw_stat > 2.5:
            return "Negative autocorrelation likely"
        else:
            return "No significant autocorrelation"

    def _test_homoscedasticity(self, residuals: np.ndarray, fitted_values: np.ndarray) -> Dict:
        """檢驗同方差性"""
        # Breusch-Pagan檢驗的簡化版本
        # 計算殘差平方與擬合值的相關性
        residuals_squared = residuals ** 2
        correlation = np.corrcoef(fitted_values, residuals_squared)[0, 1]

        return {
            "correlation_fitted_residuals_sq": float(correlation) if not np.isnan(correlation) else 0.0,
            "homoscedastic": bool(abs(correlation) < 0.3) if not np.isnan(correlation) else True,
            "interpretation": "Homoscedastic" if abs(correlation) < 0.3 else "Heteroscedastic"
        }

    def model_diagnostics(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         x_values: np.ndarray = None,
                         model_params: Dict = None) -> Dict:
        """
        綜合模型診斷

        Parameters:
        -----------
        y_true : np.ndarray
            真實值
        y_pred : np.ndarray
            預測值
        x_values : np.ndarray, optional
            自變量值
        model_params : dict, optional
            模型參數

        Returns:
        --------
        dict : 綜合診斷結果
        """
        # 計算評估指標
        metrics = self.calculate_metrics(y_true, y_pred)

        # 殘差分析
        residual_analysis = self.residual_analysis(y_true, y_pred, x_values)

        # 模型適合度評估
        fit_assessment = self._assess_model_fit(metrics, residual_analysis)

        # 預測區間計算（簡化版本）
        prediction_intervals = self._calculate_prediction_intervals(y_true, y_pred)

        diagnostics = {
            "metrics": metrics,
            "residual_analysis": residual_analysis,
            "fit_assessment": fit_assessment,
            "prediction_intervals": prediction_intervals
        }

        if model_params:
            diagnostics["model_parameters"] = model_params

        return diagnostics

    def _assess_model_fit(self, metrics: Dict, residual_analysis: Dict) -> Dict:
        """評估模型適合度"""
        r2 = metrics["r2_score"]
        is_normal = residual_analysis["normality_test"]["is_normal"]
        is_homoscedastic = residual_analysis["homoscedasticity"]["homoscedastic"]
        outlier_percentage = residual_analysis["outliers"]["percentage"]

        # 綜合評估
        if r2 > 0.8 and is_normal and is_homoscedastic and outlier_percentage < 5:
            overall_quality = "Excellent"
        elif r2 > 0.6 and outlier_percentage < 10:
            overall_quality = "Good"
        elif r2 > 0.4:
            overall_quality = "Fair"
        else:
            overall_quality = "Poor"

        return {
            "overall_quality": overall_quality,
            "r2_interpretation": self._interpret_r2(r2),
            "assumptions_met": {
                "linearity": bool(r2 > 0.3),
                "normality": is_normal if is_normal is not None else True,
                "homoscedasticity": is_homoscedastic,
                "no_excessive_outliers": bool(outlier_percentage < 10)
            },
            "recommendations": self._generate_recommendations(r2, is_normal, is_homoscedastic, outlier_percentage)
        }

    def _interpret_r2(self, r2: float) -> str:
        """解釋R²值"""
        if r2 > 0.9:
            return "Very strong relationship"
        elif r2 > 0.7:
            return "Strong relationship"
        elif r2 > 0.5:
            return "Moderate relationship"
        elif r2 > 0.3:
            return "Weak relationship"
        else:
            return "Very weak relationship"

    def _generate_recommendations(self, r2: float, is_normal: bool, is_homoscedastic: bool, outlier_percentage: float) -> List[str]:
        """生成改進建議"""
        recommendations = []

        if r2 < 0.5:
            recommendations.append("Consider adding more features or using a non-linear model")

        if not is_normal:
            recommendations.append("Consider data transformation to improve normality of residuals")

        if not is_homoscedastic:
            recommendations.append("Consider weighted regression or data transformation for heteroscedasticity")

        if outlier_percentage > 10:
            recommendations.append("Investigate and consider removing outliers")

        if not recommendations:
            recommendations.append("Model appears to meet linear regression assumptions well")

        return recommendations

    def _calculate_prediction_intervals(self, y_true: np.ndarray, y_pred: np.ndarray, confidence: float = 0.95) -> Dict:
        """計算預測區間（簡化版本）"""
        residuals = y_true - y_pred
        residual_std = np.std(residuals)

        # 使用t分佈計算區間
        n = len(y_true)
        df = n - 2  # 自由度
        alpha = 1 - confidence
        t_value = stats.t.ppf(1 - alpha/2, df) if df > 0 else 1.96

        margin_of_error = t_value * residual_std

        return {
            "confidence_level": confidence,
            "margin_of_error": float(margin_of_error),
            "lower_bound": (y_pred - margin_of_error).tolist(),
            "upper_bound": (y_pred + margin_of_error).tolist(),
            "residual_std": float(residual_std)
        }

    def compare_models(self, model_results: List[Dict]) -> Dict:
        """
        比較多個模型的性能

        Parameters:
        -----------
        model_results : list
            包含多個模型結果的列表

        Returns:
        --------
        dict : 模型比較結果
        """
        if len(model_results) < 2:
            return {"error": "At least 2 models required for comparison"}

        comparison_metrics = ["r2_score", "rmse", "mae", "mape"]
        comparison = {}

        for metric in comparison_metrics:
            values = [result.get(metric, np.nan) for result in model_results]

            if metric == "r2_score":
                best_idx = np.nanargmax(values)  # R²越大越好
            else:
                best_idx = np.nanargmin(values)  # 其他指標越小越好

            comparison[metric] = {
                "values": values,
                "best_model_index": int(best_idx),
                "best_value": float(values[best_idx])
            }

        return comparison