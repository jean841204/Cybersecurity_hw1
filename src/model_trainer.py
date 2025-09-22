import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class LinearRegressionModel:
    """
    線性回歸模型類，封裝模型訓練和預測功能
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        初始化線性回歸模型

        Parameters:
        -----------
        test_size : float
            測試集比例
        random_state : int
            隨機種子
        """
        self.model = LinearRegression()
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.test_size = test_size
        self.random_state = random_state

        # 訓練數據
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # 標準化數據
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train_scaled = None
        self.y_test_scaled = None

        # 模型結果
        self.is_trained = False
        self.use_scaling = False

    def prepare_data(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    use_scaling: bool = False) -> Dict:
        """
        準備訓練數據

        Parameters:
        -----------
        X : np.ndarray
            特徵數據
        y : np.ndarray
            目標數據
        use_scaling : bool
            是否使用標準化

        Returns:
        --------
        dict : 數據準備結果
        """
        # 重塑數據維度
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # 分割訓練集和測試集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        self.use_scaling = use_scaling

        if use_scaling:
            # 標準化數據
            self.X_train_scaled = self.scaler_x.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler_x.transform(self.X_test)
            self.y_train_scaled = self.scaler_y.fit_transform(self.y_train)
            self.y_test_scaled = self.scaler_y.transform(self.y_test)
        else:
            self.X_train_scaled = self.X_train
            self.X_test_scaled = self.X_test
            self.y_train_scaled = self.y_train.ravel()
            self.y_test_scaled = self.y_test.ravel()

        return {
            "train_size": len(self.X_train),
            "test_size": len(self.X_test),
            "scaling_used": use_scaling,
            "x_shape": X.shape,
            "y_shape": y.shape
        }

    def train_model(self) -> Dict:
        """
        訓練線性回歸模型

        Returns:
        --------
        dict : 訓練結果
        """
        if self.X_train is None:
            return {"error": "Data not prepared. Call prepare_data() first."}

        # 訓練模型
        if self.use_scaling:
            self.model.fit(self.X_train_scaled, self.y_train_scaled.ravel())
        else:
            self.model.fit(self.X_train_scaled, self.y_train_scaled)

        self.is_trained = True

        # 獲取模型參數
        slope = float(self.model.coef_[0])
        intercept = float(self.model.intercept_)

        # 如果使用了標準化，需要轉換回原始尺度
        if self.use_scaling:
            # 轉換斜率和截距到原始尺度
            original_slope = slope * (self.scaler_y.scale_[0] / self.scaler_x.scale_[0])
            original_intercept = (intercept * self.scaler_y.scale_[0] +
                                self.scaler_y.mean_[0] -
                                original_slope * self.scaler_x.mean_[0])
        else:
            original_slope = slope
            original_intercept = intercept

        return {
            "slope": original_slope,
            "intercept": original_intercept,
            "scaled_slope": slope,
            "scaled_intercept": intercept,
            "scaling_used": self.use_scaling,
            "training_successful": True
        }

    def make_predictions(self, X_new: Optional[np.ndarray] = None) -> Dict:
        """
        生成預測結果

        Parameters:
        -----------
        X_new : np.ndarray, optional
            新的預測數據，如果為None則使用測試集

        Returns:
        --------
        dict : 預測結果
        """
        if not self.is_trained:
            return {"error": "Model not trained. Call train_model() first."}

        if X_new is None:
            # 使用測試集進行預測
            if self.use_scaling:
                y_pred_train = self.model.predict(self.X_train_scaled)
                y_pred_test = self.model.predict(self.X_test_scaled)

                # 轉換回原始尺度
                if self.y_train.ndim > 1:
                    y_pred_train = self.scaler_y.inverse_transform(
                        y_pred_train.reshape(-1, 1)
                    ).ravel()
                    y_pred_test = self.scaler_y.inverse_transform(
                        y_pred_test.reshape(-1, 1)
                    ).ravel()
            else:
                y_pred_train = self.model.predict(self.X_train_scaled)
                y_pred_test = self.model.predict(self.X_test_scaled)

            return {
                "train_predictions": y_pred_train,
                "test_predictions": y_pred_test,
                "train_actual": self.y_train.ravel(),
                "test_actual": self.y_test.ravel(),
                "train_features": self.X_train.ravel(),
                "test_features": self.X_test.ravel()
            }
        else:
            # 對新數據進行預測
            if X_new.ndim == 1:
                X_new = X_new.reshape(-1, 1)

            if self.use_scaling:
                X_new_scaled = self.scaler_x.transform(X_new)
                y_pred = self.model.predict(X_new_scaled)

                if self.y_train.ndim > 1:
                    y_pred = self.scaler_y.inverse_transform(
                        y_pred.reshape(-1, 1)
                    ).ravel()
            else:
                y_pred = self.model.predict(X_new)

            return {
                "predictions": y_pred,
                "features": X_new.ravel()
            }

    def get_model_equation(self) -> str:
        """
        獲取模型方程式

        Returns:
        --------
        str : 線性回歸方程式
        """
        if not self.is_trained:
            return "Model not trained yet"

        # 獲取原始尺度的參數
        train_result = self.train_model()
        slope = train_result["slope"]
        intercept = train_result["intercept"]

        if intercept >= 0:
            return f"y = {slope:.4f}x + {intercept:.4f}"
        else:
            return f"y = {slope:.4f}x - {abs(intercept):.4f}"

    def get_model_summary(self) -> Dict:
        """
        獲取模型摘要信息

        Returns:
        --------
        dict : 模型摘要
        """
        if not self.is_trained:
            return {"error": "Model not trained yet"}

        # 重新訓練以獲取參數（避免重複計算）
        train_result = self.train_model()

        return {
            "model_type": "Linear Regression",
            "equation": self.get_model_equation(),
            "parameters": {
                "slope": train_result["slope"],
                "intercept": train_result["intercept"]
            },
            "data_info": {
                "train_size": len(self.X_train) if self.X_train is not None else 0,
                "test_size": len(self.X_test) if self.X_test is not None else 0,
                "scaling_used": self.use_scaling
            },
            "sklearn_model": {
                "coef_": float(self.model.coef_[0]),
                "intercept_": float(self.model.intercept_)
            }
        }

    def predict_range(self, x_min: float, x_max: float, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        在指定範圍內生成預測線

        Parameters:
        -----------
        x_min : float
            X的最小值
        x_max : float
            X的最大值
        n_points : int
            預測點數量

        Returns:
        --------
        tuple : (x_range, y_predictions)
        """
        if not self.is_trained:
            return np.array([]), np.array([])

        x_range = np.linspace(x_min, x_max, n_points)
        predictions = self.make_predictions(x_range)

        return x_range, predictions["predictions"]