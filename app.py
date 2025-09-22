import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

# 添加src目錄到Python路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import DataGenerator
from model_trainer import LinearRegressionModel
from evaluator import ModelEvaluator
from visualizer import RegressionVisualizer

# 頁面配置
st.set_page_config(
    page_title="Linear Regression Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .sidebar-header {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """初始化session state"""
    if 'data_generated' not in st.session_state:
        st.session_state.data_generated = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'data_generator' not in st.session_state:
        st.session_state.data_generator = DataGenerator()
    if 'model' not in st.session_state:
        st.session_state.model = LinearRegressionModel()
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = ModelEvaluator()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = RegressionVisualizer()

def main():
    """主應用函數"""
    init_session_state()

    # 主標題
    st.markdown("""
    <div class="main-header">
        <h1>📊 Linear Regression Analysis System</h1>
        <p>Interactive Linear Regression Analysis Tool Based on CRISP-DM Process</p>
    </div>
    """, unsafe_allow_html=True)

    # 側邊欄控制面板
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2>🎛️ Parameter Control Panel</h2>
        </div>
        """, unsafe_allow_html=True)

        # Parameters
        st.subheader("Parameters")

        n_points = st.slider(
            "Number of data points (n)",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Number of data points to generate"
        )

        slope = st.slider(
            "Coefficient 'a' (y = ax + b + noise)",
            min_value=-10.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="Slope parameter in linear equation"
        )

        noise_variance = st.slider(
            "Noise Variance (var)",
            min_value=0.0,
            max_value=5000.0,
            value=1000.0,
            step=100.0,
            help="Variance of Gaussian noise N(0, variance)"
        )

        noise_level = np.sqrt(noise_variance)  # Convert variance to std dev
        intercept = 30.0  # Fixed intercept as per requirements
        test_size = 0.2  # Fixed test size
        use_scaling = False  # Fixed scaling
        random_seed = 42  # Fixed seed

        # 自動生成和訓練 (實時更新，無需按鈕)
        # 當參數改變時自動更新
        if not hasattr(st.session_state, 'last_params') or st.session_state.last_params != (n_points, slope, noise_variance):
            generate_and_train_automatically(slope, intercept, n_points, noise_level, random_seed, test_size, use_scaling)
            st.session_state.last_params = (n_points, slope, noise_variance)

    # 主內容區域 - 簡化版本
    if st.session_state.model_trained:
        # 直接顯示核心圖表
        show_simple_regression_plot()

        # 新增功能區域
        col1, col2 = st.columns(2)

        with col1:
            show_model_coefficients()

        with col2:
            show_top_outliers_table()
    else:
        st.info("👈 Please adjust parameters to see the regression plot")

def generate_and_train_automatically(slope, intercept, n_points, noise_level, random_seed, test_size, use_scaling):
    """自動生成數據並訓練模型"""
    try:
        # 生成數據
        x_data, y_data = st.session_state.data_generator.generate_linear_data(
            n_points=n_points,
            slope=slope,
            intercept=intercept,
            noise_level=noise_level,
            random_seed=random_seed
        )

        st.session_state.x_data = x_data
        st.session_state.y_data = y_data
        st.session_state.data_generated = True

        # 自動訓練模型
        st.session_state.model = LinearRegressionModel(
            test_size=test_size,
            random_state=42
        )

        # 準備數據
        st.session_state.model.prepare_data(
            st.session_state.x_data,
            st.session_state.y_data,
            use_scaling=use_scaling
        )

        # 訓練模型
        train_result = st.session_state.model.train_model()

        # 生成預測
        predictions = st.session_state.model.make_predictions()

        # 計算評估指標
        metrics = st.session_state.evaluator.calculate_metrics(
            np.concatenate([predictions['train_actual'], predictions['test_actual']]),
            np.concatenate([predictions['train_predictions'], predictions['test_predictions']])
        )

        # 存儲結果
        st.session_state.train_result = train_result
        st.session_state.predictions = predictions
        st.session_state.metrics = metrics
        st.session_state.model_trained = True

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def show_simple_regression_plot():
    """顯示簡化的回歸圖表"""
    # Get outliers for visualization
    all_actual = np.concatenate([st.session_state.predictions['train_actual'],
                               st.session_state.predictions['test_actual']])
    all_pred = np.concatenate([st.session_state.predictions['train_predictions'],
                             st.session_state.predictions['test_predictions']])
    all_x = np.concatenate([st.session_state.predictions['train_features'],
                          st.session_state.predictions['test_features']])

    outliers_info = st.session_state.evaluator.get_top_outliers(
        all_actual, all_pred, all_x, n_outliers=5
    )

    # Create regression line data
    x_line = np.linspace(np.min(st.session_state.x_data), np.max(st.session_state.x_data), 100)
    y_line = st.session_state.train_result['slope'] * x_line + st.session_state.train_result['intercept']

    # Simple regression plot
    fig = st.session_state.visualizer.plot_scatter_with_regression(
        st.session_state.x_data,
        st.session_state.y_data,
        y_pred=y_line,
        x_pred=x_line,
        title="Linear Regression with Top 5 Outliers",
        show_equation=True,
        equation=f"y = {st.session_state.train_result['slope']:.4f}x + {st.session_state.train_result['intercept']:.4f}",
        r2_score=st.session_state.metrics['r2_score'],
        outliers_info=outliers_info
    )

    st.plotly_chart(fig, use_container_width=True)

def show_model_coefficients():
    """顯示模型係數"""
    st.subheader("Model Coefficients")

    coefficients_df = pd.DataFrame({
        'Coefficient': ['Slope (a)', 'Intercept (b)', 'R² Score'],
        'True Value': [
            f"{st.session_state.data_generator.true_slope:.4f}",
            f"{st.session_state.data_generator.true_intercept:.4f}",
            "-"
        ],
        'Estimated Value': [
            f"{st.session_state.train_result['slope']:.4f}",
            f"{st.session_state.train_result['intercept']:.4f}",
            f"{st.session_state.metrics['r2_score']:.4f}"
        ]
    })

    st.dataframe(coefficients_df, use_container_width=True, hide_index=True)

def show_top_outliers_table():
    """顯示Top 5異常值表格"""
    st.subheader("Top 5 Outliers")

    # 獲取異常值數據
    all_actual = np.concatenate([st.session_state.predictions['train_actual'],
                               st.session_state.predictions['test_actual']])
    all_pred = np.concatenate([st.session_state.predictions['train_predictions'],
                             st.session_state.predictions['test_predictions']])
    all_x = np.concatenate([st.session_state.predictions['train_features'],
                          st.session_state.predictions['test_features']])

    outliers_info = st.session_state.evaluator.get_top_outliers(
        all_actual, all_pred, all_x, n_outliers=5
    )

    # 創建異常值表格
    outliers_df = pd.DataFrame({
        'Rank': [f"O{i+1}" for i in range(len(outliers_info['x_values']))],
        'X Value': [f"{x:.3f}" for x in outliers_info['x_values']],
        'Y Value': [f"{y:.3f}" for y in outliers_info['y_values']],
        'Residual': [f"{r:.3f}" for r in outliers_info['residuals']],
        'Score': [f"{s:.3f}" for s in outliers_info['scores']]
    })

    st.dataframe(outliers_df, use_container_width=True, hide_index=True)

def show_data_overview():
    """Show data overview"""
    st.subheader("📊 Data Overview and Exploratory Analysis")

    # 獲取數據摘要
    summary = st.session_state.data_generator.data_summary()
    quality_report = st.session_state.data_generator.check_data_quality()

    # 顯示統計信息
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Data Points",
            summary['data_points'],
            help="Total number of generated data points"
        )

    with col2:
        st.metric(
            "Correlation",
            f"{summary['correlation']:.4f}",
            help="Pearson correlation coefficient between X and Y"
        )

    with col3:
        st.metric(
            "True Slope",
            f"{summary['parameters']['true_slope']:.2f}",
            help="True slope used when generating data"
        )

    with col4:
        st.metric(
            "Noise Level",
            f"{summary['parameters']['noise_level']:.2f}",
            help="Standard deviation of added Gaussian noise"
        )

    # 數據視覺化
    col1, col2 = st.columns([2, 1])

    with col1:
        # 散點圖
        fig = st.session_state.visualizer.plot_scatter_with_regression(
            st.session_state.x_data,
            st.session_state.y_data,
            title="原始數據散點圖",
            x_label="X",
            y_label="Y"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 數據統計表
        st.subheader("統計摘要")

        stats_df = pd.DataFrame({
            'X': [
                summary['x_statistics']['mean'],
                summary['x_statistics']['std'],
                summary['x_statistics']['min'],
                summary['x_statistics']['max']
            ],
            'Y': [
                summary['y_statistics']['mean'],
                summary['y_statistics']['std'],
                summary['y_statistics']['min'],
                summary['y_statistics']['max']
            ]
        }, index=['平均值', '標準差', '最小值', '最大值'])

        st.dataframe(stats_df, use_container_width=True)

        # 數據品質報告
        st.subheader("數據品質")
        if quality_report['missing_values']['x_missing'] == 0 and quality_report['missing_values']['y_missing'] == 0:
            st.success("✅ 無缺失值")
        else:
            st.warning(f"⚠️ 發現缺失值: X({quality_report['missing_values']['x_missing']}), Y({quality_report['missing_values']['y_missing']})")

        outlier_percentage = (quality_report['outliers']['x_outliers'] + quality_report['outliers']['y_outliers']) / (2 * summary['data_points']) * 100
        if outlier_percentage < 5:
            st.success(f"✅ 異常值比例: {outlier_percentage:.1f}%")
        else:
            st.warning(f"⚠️ 異常值比例: {outlier_percentage:.1f}%")

    # 數據分佈分析
    st.subheader("數據分佈分析")
    fig_dist = st.session_state.visualizer.plot_data_distribution(
        st.session_state.x_data,
        st.session_state.y_data
    )
    st.plotly_chart(fig_dist, use_container_width=True)

def show_model_results():
    """Show model results"""
    # 回歸分析表格在最上面
    st.subheader("📊 Regression Analysis Results")

    # 回歸方程式
    equation = st.session_state.model.get_model_equation()
    st.info(f"📐 **Regression Equation**: {equation}")

    # 模型結果表格
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Parameters")
        params_df = pd.DataFrame({
            'Parameter': ['True Slope (a)', 'Estimated Slope', 'True Intercept (b)', 'Estimated Intercept'],
            'Value': [
                f"{st.session_state.data_generator.true_slope:.4f}",
                f"{st.session_state.train_result['slope']:.4f}",
                f"{st.session_state.data_generator.true_intercept:.4f}",
                f"{st.session_state.train_result['intercept']:.4f}"
            ]
        })
        st.dataframe(params_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Model Performance")
        metrics_df = pd.DataFrame({
            'Metric': ['R² Score', 'RMSE', 'MAE', 'MAPE (%)'],
            'Value': [
                f"{st.session_state.metrics['r2_score']:.4f}",
                f"{st.session_state.metrics['rmse']:.4f}",
                f"{st.session_state.metrics['mae']:.4f}",
                f"{st.session_state.metrics['mape']:.2f}"
            ]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    # 主要視覺化
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 回歸分析", "📊 殘差分析", "🔍 預測對比", "📈 評估指標"])

    with tab1:
        # Get outliers for visualization
        all_actual = np.concatenate([st.session_state.predictions['train_actual'],
                                   st.session_state.predictions['test_actual']])
        all_pred = np.concatenate([st.session_state.predictions['train_predictions'],
                                 st.session_state.predictions['test_predictions']])
        all_x = np.concatenate([st.session_state.predictions['train_features'],
                              st.session_state.predictions['test_features']])

        outliers_info = st.session_state.evaluator.get_top_outliers(
            all_actual, all_pred, all_x, n_outliers=5
        )

        # Create regression line data
        x_line = np.linspace(np.min(st.session_state.x_data), np.max(st.session_state.x_data), 100)
        y_line = st.session_state.train_result['slope'] * x_line + st.session_state.train_result['intercept']

        # Complete regression plot with outliers
        fig_regression = st.session_state.visualizer.plot_scatter_with_regression(
            st.session_state.x_data,
            st.session_state.y_data,
            y_pred=y_line,
            x_pred=x_line,
            title="Linear Regression Analysis with Top 5 Outliers",
            show_equation=True,
            equation=f"y = {st.session_state.train_result['slope']:.4f}x + {st.session_state.train_result['intercept']:.4f}",
            r2_score=st.session_state.metrics['r2_score'],
            outliers_info=outliers_info
        )
        st.plotly_chart(fig_regression, use_container_width=True)

        # 模型參數比較
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("真實參數 vs 估計參數")
            comparison_df = pd.DataFrame({
                '真實值': [
                    st.session_state.data_generator.true_slope,
                    st.session_state.data_generator.true_intercept
                ],
                '估計值': [
                    st.session_state.train_result['slope'],
                    st.session_state.train_result['intercept']
                ],
                '誤差': [
                    abs(st.session_state.data_generator.true_slope - st.session_state.train_result['slope']),
                    abs(st.session_state.data_generator.true_intercept - st.session_state.train_result['intercept'])
                ]
            }, index=['斜率', '截距'])
            st.dataframe(comparison_df, use_container_width=True)

        with col2:
            st.subheader("訓練/測試集信息")
            train_size = len(st.session_state.predictions['train_actual'])
            test_size = len(st.session_state.predictions['test_actual'])

            info_df = pd.DataFrame({
                '數量': [train_size, test_size, train_size + test_size],
                '比例': [f"{train_size/(train_size+test_size)*100:.1f}%",
                        f"{test_size/(train_size+test_size)*100:.1f}%", "100%"]
            }, index=['訓練集', '測試集', '總計'])
            st.dataframe(info_df, use_container_width=True)

    with tab2:
        # 殘差分析
        all_actual = np.concatenate([st.session_state.predictions['train_actual'],
                                   st.session_state.predictions['test_actual']])
        all_pred = np.concatenate([st.session_state.predictions['train_predictions'],
                                 st.session_state.predictions['test_predictions']])

        fig_residuals = st.session_state.visualizer.plot_residuals(
            all_actual, all_pred
        )
        st.plotly_chart(fig_residuals, use_container_width=True)

        # 殘差分析結果
        residual_analysis = st.session_state.evaluator.residual_analysis(
            all_actual, all_pred
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("正態性檢驗")
            normality = residual_analysis['normality_test']
            if normality['is_normal']:
                st.success("✅ 殘差符合正態分佈")
            else:
                st.warning("⚠️ 殘差偏離正態分佈")

            if normality['shapiro_p_value']:
                st.write(f"Shapiro-Wilk p值: {normality['shapiro_p_value']:.4f}")

        with col2:
            st.subheader("異常值檢測")
            outliers = residual_analysis['outliers']
            st.write(f"異常值數量: {outliers['count']}")
            st.write(f"異常值比例: {outliers['percentage']:.2f}%")

            if outliers['percentage'] < 5:
                st.success("✅ 異常值比例正常")
            else:
                st.warning("⚠️ 異常值比例偏高")

    with tab3:
        # 預測對比
        fig_pred_vs_actual = st.session_state.visualizer.plot_predictions_vs_actual(
            all_actual, all_pred
        )
        st.plotly_chart(fig_pred_vs_actual, use_container_width=True)

        # 預測精度分析
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("訓練集表現")
            train_metrics = st.session_state.evaluator.calculate_metrics(
                st.session_state.predictions['train_actual'],
                st.session_state.predictions['train_predictions']
            )
            st.write(f"R²: {train_metrics['r2_score']:.4f}")
            st.write(f"RMSE: {train_metrics['rmse']:.4f}")
            st.write(f"MAE: {train_metrics['mae']:.4f}")

        with col2:
            st.subheader("測試集表現")
            test_metrics = st.session_state.evaluator.calculate_metrics(
                st.session_state.predictions['test_actual'],
                st.session_state.predictions['test_predictions']
            )
            st.write(f"R²: {test_metrics['r2_score']:.4f}")
            st.write(f"RMSE: {test_metrics['rmse']:.4f}")
            st.write(f"MAE: {test_metrics['mae']:.4f}")

    with tab4:
        # 評估指標可視化
        fig_metrics = st.session_state.visualizer.plot_metrics_comparison(
            st.session_state.metrics
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

        # 詳細評估報告
        st.subheader("詳細評估報告")

        metrics_df = pd.DataFrame({
            '指標': ['R² Score', 'Adjusted R²', 'MSE', 'RMSE', 'MAE', 'MAPE (%)', 'SMAPE (%)'],
            '數值': [
                f"{st.session_state.metrics['r2_score']:.4f}",
                f"{st.session_state.metrics['adjusted_r2']:.4f}",
                f"{st.session_state.metrics['mse']:.4f}",
                f"{st.session_state.metrics['rmse']:.4f}",
                f"{st.session_state.metrics['mae']:.4f}",
                f"{st.session_state.metrics['mape']:.2f}",
                f"{st.session_state.metrics['smape']:.2f}"
            ],
            '說明': [
                '決定係數，越接近1越好',
                '調整後決定係數',
                '均方誤差，越小越好',
                '均方根誤差，越小越好',
                '平均絕對誤差，越小越好',
                '平均絕對百分比誤差',
                '對稱平均絕對百分比誤差'
            ]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()