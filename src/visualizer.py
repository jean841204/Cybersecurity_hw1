import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Dict, Tuple, List, Optional
import streamlit as st


class RegressionVisualizer:
    """
    回歸模型視覺化模組，提供各種互動式圖表
    """

    def __init__(self, theme: str = "plotly_white"):
        """
        初始化視覺化器

        Parameters:
        -----------
        theme : str
            Plotly主題
        """
        self.theme = theme
        self.color_palette = {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "success": "#2ca02c",
            "danger": "#d62728",
            "warning": "#ff7f0e",
            "info": "#17a2b8",
            "light": "#f8f9fa",
            "dark": "#343a40"
        }

    def plot_scatter_with_regression(self,
                                   x_data: np.ndarray,
                                   y_data: np.ndarray,
                                   y_pred: np.ndarray = None,
                                   x_pred: np.ndarray = None,
                                   title: str = "Linear Regression Analysis",
                                   x_label: str = "X",
                                   y_label: str = "Y",
                                   show_equation: bool = True,
                                   equation: str = None,
                                   r2_score: float = None,
                                   outliers_info: Dict = None) -> go.Figure:
        """
        繪製散點圖與回歸線

        Parameters:
        -----------
        x_data : np.ndarray
            X數據
        y_data : np.ndarray
            Y數據
        y_pred : np.ndarray, optional
            預測的Y值
        x_pred : np.ndarray, optional
            預測線的X值
        title : str
            圖表標題
        x_label : str
            X軸標籤
        y_label : str
            Y軸標籤
        show_equation : bool
            是否顯示方程式
        equation : str
            回歸方程式
        r2_score : float
            R²分數

        Returns:
        --------
        plotly.graph_objects.Figure : 互動式圖表
        """
        fig = go.Figure()

        # 添加散點圖
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            name='Data Points',
            marker=dict(
                size=8,
                color=self.color_palette["primary"],
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            hovertemplate='<b>X</b>: %{x:.2f}<br><b>Y</b>: %{y:.2f}<extra></extra>'
        ))

        # 添加回歸線
        if y_pred is not None and x_pred is not None:
            fig.add_trace(go.Scatter(
                x=x_pred,
                y=y_pred,
                mode='lines',
                name='Regression Line',
                line=dict(
                    color=self.color_palette["danger"],
                    width=3
                ),
                hovertemplate='<b>Regression Line</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
            ))

        # 添加異常值標記
        if outliers_info is not None:
            outlier_x = outliers_info.get('x_values', [])
            outlier_y = outliers_info.get('y_values', [])
            if len(outlier_x) > 0 and len(outlier_y) > 0:
                fig.add_trace(go.Scatter(
                    x=outlier_x,
                    y=outlier_y,
                    mode='markers+text',
                    name='Top 5 Outliers',
                    marker=dict(
                        size=12,
                        color='red',
                        symbol='x',
                        line=dict(width=2)
                    ),
                    text=[f'O{i+1}' for i in range(len(outlier_x))],
                    textposition='top center',
                    textfont=dict(size=10, color='red'),
                    hovertemplate='<b>Outlier %{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
                ))

        # 設置標題和標籤
        title_text = title
        if show_equation and equation:
            title_text += f"<br><sub>{equation}"
            if r2_score is not None:
                title_text += f" | R² = {r2_score:.4f}"
            title_text += "</sub>"

        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title=x_label,
            yaxis_title=y_label,
            template=self.theme,
            hovermode='closest',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(t=80, r=20, b=60, l=60)
        )

        return fig

    def plot_residuals(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      x_values: np.ndarray = None,
                      title: str = "殘差分析") -> go.Figure:
        """
        繪製殘差圖

        Parameters:
        -----------
        y_true : np.ndarray
            真實值
        y_pred : np.ndarray
            預測值
        x_values : np.ndarray, optional
            X值（用於殘差vs擬合值圖）
        title : str
            圖表標題

        Returns:
        --------
        plotly.graph_objects.Figure : 殘差圖
        """
        residuals = y_true - y_pred

        # 創建子圖
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('殘差 vs 擬合值', '殘差分佈', '標準化殘差', 'Q-Q圖'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # 1. 殘差 vs 擬合值
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                name='殘差',
                marker=dict(
                    size=6,
                    color=self.color_palette["primary"],
                    opacity=0.7
                ),
                hovertemplate='擬合值: %{x:.2f}<br>殘差: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        # 添加零線
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

        # 2. 殘差分佈（直方圖）
        fig.add_trace(
            go.Histogram(
                x=residuals,
                name='殘差分佈',
                marker_color=self.color_palette["secondary"],
                opacity=0.7,
                nbinsx=20
            ),
            row=1, col=2
        )

        # 3. 標準化殘差
        standardized_residuals = residuals / np.std(residuals) if np.std(residuals) != 0 else residuals
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=standardized_residuals,
                mode='markers',
                name='標準化殘差',
                marker=dict(
                    size=6,
                    color=self.color_palette["success"],
                    opacity=0.7
                ),
                hovertemplate='擬合值: %{x:.2f}<br>標準化殘差: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )

        # 添加±2標準差線
        fig.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=-2, line_dash="dash", line_color="red", row=2, col=1)

        # 4. Q-Q圖（簡化版本）
        from scipy import stats
        z_scores = np.sort(standardized_residuals)
        normal_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(z_scores)))

        fig.add_trace(
            go.Scatter(
                x=normal_quantiles,
                y=z_scores,
                mode='markers',
                name='Q-Q圖',
                marker=dict(
                    size=6,
                    color=self.color_palette["warning"],
                    opacity=0.7
                )
            ),
            row=2, col=2
        )

        # 添加理論線
        fig.add_trace(
            go.Scatter(
                x=normal_quantiles,
                y=normal_quantiles,
                mode='lines',
                name='理論線',
                line=dict(color='red', dash='dash')
            ),
            row=2, col=2
        )

        # 更新布局
        fig.update_layout(
            title=dict(text=title, x=0.5),
            template=self.theme,
            showlegend=False,
            height=600
        )

        fig.update_xaxes(title_text="擬合值", row=1, col=1)
        fig.update_yaxes(title_text="殘差", row=1, col=1)
        fig.update_xaxes(title_text="殘差", row=1, col=2)
        fig.update_yaxes(title_text="頻率", row=1, col=2)
        fig.update_xaxes(title_text="擬合值", row=2, col=1)
        fig.update_yaxes(title_text="標準化殘差", row=2, col=1)
        fig.update_xaxes(title_text="理論分位數", row=2, col=2)
        fig.update_yaxes(title_text="樣本分位數", row=2, col=2)

        return fig

    def plot_predictions_vs_actual(self,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  title: str = "預測值 vs 實際值",
                                  show_perfect_line: bool = True) -> go.Figure:
        """
        繪製預測值vs實際值圖

        Parameters:
        -----------
        y_true : np.ndarray
            真實值
        y_pred : np.ndarray
            預測值
        title : str
            圖表標題
        show_perfect_line : bool
            是否顯示完美預測線

        Returns:
        --------
        plotly.graph_objects.Figure : 預測vs實際值圖
        """
        fig = go.Figure()

        # 添加散點圖
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            name='預測點',
            marker=dict(
                size=8,
                color=self.color_palette["primary"],
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            hovertemplate='實際值: %{x:.2f}<br>預測值: %{y:.2f}<extra></extra>'
        ))

        # 添加完美預測線
        if show_perfect_line:
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            perfect_line = np.linspace(min_val, max_val, 100)

            fig.add_trace(go.Scatter(
                x=perfect_line,
                y=perfect_line,
                mode='lines',
                name='完美預測線',
                line=dict(
                    color=self.color_palette["danger"],
                    width=2,
                    dash='dash'
                ),
                hoverinfo='skip'
            ))

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis_title="實際值",
            yaxis_title="預測值",
            template=self.theme,
            hovermode='closest',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig

    def plot_metrics_comparison(self,
                               metrics: Dict,
                               title: str = "模型評估指標") -> go.Figure:
        """
        繪製評估指標比較圖

        Parameters:
        -----------
        metrics : dict
            評估指標字典
        title : str
            圖表標題

        Returns:
        --------
        plotly.graph_objects.Figure : 指標比較圖
        """
        # 選擇要顯示的主要指標
        key_metrics = {
            'R² Score': metrics.get('r2_score', 0),
            'RMSE': metrics.get('rmse', 0),
            'MAE': metrics.get('mae', 0),
            'MAPE (%)': metrics.get('mape', 0)
        }

        # 為了更好的視覺效果，對指標進行標準化
        normalized_metrics = {}
        for key, value in key_metrics.items():
            if key == 'R² Score':
                normalized_metrics[key] = value  # R²已經在0-1之間
            elif key == 'MAPE (%)':
                normalized_metrics[key] = min(value / 100, 1)  # 轉換為0-1範圍
            else:
                # 對於RMSE和MAE，使用相對大小（需要更多上下文來標準化）
                normalized_metrics[key] = value

        fig = go.Figure()

        colors = [self.color_palette["primary"], self.color_palette["secondary"],
                 self.color_palette["success"], self.color_palette["warning"]]

        fig.add_trace(go.Bar(
            x=list(key_metrics.keys()),
            y=list(key_metrics.values()),
            marker_color=colors,
            text=[f'{v:.4f}' for v in key_metrics.values()],
            textposition='auto',
            hovertemplate='%{x}: %{y:.4f}<extra></extra>'
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            yaxis_title="指標值",
            template=self.theme,
            showlegend=False
        )

        return fig

    def plot_data_distribution(self,
                              x_data: np.ndarray,
                              y_data: np.ndarray,
                              title: str = "數據分佈分析") -> go.Figure:
        """
        繪製數據分佈圖

        Parameters:
        -----------
        x_data : np.ndarray
            X數據
        y_data : np.ndarray
            Y數據
        title : str
            圖表標題

        Returns:
        --------
        plotly.graph_objects.Figure : 數據分佈圖
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('X 分佈', 'Y 分佈', 'X-Y 散點圖', '相關性熱圖'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # X分佈
        fig.add_trace(
            go.Histogram(
                x=x_data,
                name='X分佈',
                marker_color=self.color_palette["primary"],
                opacity=0.7,
                nbinsx=20
            ),
            row=1, col=1
        )

        # Y分佈
        fig.add_trace(
            go.Histogram(
                x=y_data,
                name='Y分佈',
                marker_color=self.color_palette["secondary"],
                opacity=0.7,
                nbinsx=20
            ),
            row=1, col=2
        )

        # X-Y散點圖
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                name='X-Y關係',
                marker=dict(
                    size=6,
                    color=self.color_palette["success"],
                    opacity=0.6
                )
            ),
            row=2, col=1
        )

        # 相關性矩陣（簡化為單個值）
        correlation = np.corrcoef(x_data, y_data)[0, 1]
        fig.add_trace(
            go.Heatmap(
                z=[[1, correlation], [correlation, 1]],
                x=['X', 'Y'],
                y=['X', 'Y'],
                colorscale='RdBu',
                zmid=0,
                text=[[1, f'{correlation:.3f}'], [f'{correlation:.3f}', 1]],
                texttemplate='%{text}',
                textfont={"size": 14},
                showscale=True
            ),
            row=2, col=2
        )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            template=self.theme,
            showlegend=False,
            height=600
        )

        fig.update_xaxes(title_text="值", row=1, col=1)
        fig.update_yaxes(title_text="頻率", row=1, col=1)
        fig.update_xaxes(title_text="值", row=1, col=2)
        fig.update_yaxes(title_text="頻率", row=1, col=2)
        fig.update_xaxes(title_text="X", row=2, col=1)
        fig.update_yaxes(title_text="Y", row=2, col=1)

        return fig

    def create_interactive_regression_plot(self,
                                         x_data: np.ndarray,
                                         y_data: np.ndarray,
                                         model_results: Dict,
                                         show_confidence: bool = True) -> go.Figure:
        """
        創建完整的互動式回歸分析圖

        Parameters:
        -----------
        x_data : np.ndarray
            X數據
        y_data : np.ndarray
            Y數據
        model_results : dict
            模型結果
        show_confidence : bool
            是否顯示置信區間

        Returns:
        --------
        plotly.graph_objects.Figure : 完整的互動式圖表
        """
        fig = go.Figure()

        # 原始數據散點
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='markers',
            name='原始數據',
            marker=dict(
                size=8,
                color=self.color_palette["primary"],
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            hovertemplate='<b>原始數據</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))

        # 回歸線
        x_line = np.linspace(np.min(x_data), np.max(x_data), 100)
        slope = model_results.get('slope', 1)
        intercept = model_results.get('intercept', 0)
        y_line = slope * x_line + intercept

        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name='回歸線',
            line=dict(
                color=self.color_palette["danger"],
                width=3
            ),
            hovertemplate='<b>回歸線</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))

        # 預測點（如果有的話）
        if 'train_predictions' in model_results:
            train_pred = model_results['train_predictions']
            train_features = model_results['train_features']

            fig.add_trace(go.Scatter(
                x=train_features,
                y=train_pred,
                mode='markers',
                name='訓練預測',
                marker=dict(
                    size=6,
                    color=self.color_palette["success"],
                    opacity=0.8,
                    symbol='diamond'
                ),
                hovertemplate='<b>訓練預測</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
            ))

        if 'test_predictions' in model_results:
            test_pred = model_results['test_predictions']
            test_features = model_results['test_features']

            fig.add_trace(go.Scatter(
                x=test_features,
                y=test_pred,
                mode='markers',
                name='測試預測',
                marker=dict(
                    size=8,
                    color=self.color_palette["warning"],
                    opacity=0.8,
                    symbol='square'
                ),
                hovertemplate='<b>測試預測</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
            ))

        # 方程式和R²
        equation = f"y = {slope:.4f}x + {intercept:.4f}"
        r2 = model_results.get('r2_score', 0)

        fig.update_layout(
            title=dict(
                text=f"線性回歸分析<br><sub>{equation} | R² = {r2:.4f}</sub>",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="X",
            yaxis_title="Y",
            template=self.theme,
            hovermode='closest',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.8)"
            )
        )

        return fig