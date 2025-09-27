# Linear Regression Interactive Analysis Tool

## 🌐 Demo Website
**Live Demo**: https://cybersecurityhw1-jeanchen.streamlit.app/

Try the interactive tool directly in your browser - no installation required!

## CRISP-DM流程規劃

### 1. Business Understanding（業務理解）
**目標：**
- 建立教育性質的線性回歸演示工具
- 幫助用戶理解線性回歸的基本概念
- 展示不同參數對模型性能的影響

**成功指標：**
- 系統能夠正確生成和分析線性數據
- 用戶界面直觀易用
- 回歸分析結果準確

### 2. Data Understanding（數據理解）
**數據來源：**
- 程式生成的合成數據
- 基於用戶設定的參數：y = ax + b + noise

**數據特徵：**
- X：自變量（連續數值）
- Y：因變量（連續數值，含雜訊）
- 數據點數量：用戶可調整（10-1000點）

**探索性數據分析：**
- 數據分佈視覺化
- 統計描述（均值、標準差、範圍）
- 相關性分析

### 3. Data Preparation（數據準備）
**數據生成流程：**
```
1. 生成X值序列（等間距或隨機）
2. 根據公式計算理論Y值：Y_theory = a*X + b
3. 添加高斯雜訊：Y_actual = Y_theory + noise
4. 數據標準化（可選）
```

**數據品質檢查：**
- 檢查缺失值
- 異常值檢測
- 數據範圍驗證

### 4. Modeling（建模）
**模型選擇：**
- 簡單線性回歸（Ordinary Least Squares）
- 使用scikit-learn LinearRegression

**模型訓練：**
```
1. 訓練集/測試集分割（80/20）
2. 模型擬合
3. 預測結果生成
```

**模型參數：**
- 回歸係數（斜率）
- 截距
- 相關係數（R²）

### 5. Evaluation（評估）
**評估指標：**
- R² Score（決定係數）
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

**視覺化評估：**
- 散點圖與回歸線
- 殘差分析圖
- 預測值vs實際值圖

### 6. Deployment（部署）
**技術架構：**
- 前端：Streamlit
- 後端：Python科學計算庫
- 部署：本地運行或雲端平台

**用戶界面設計：**
- 參數調整面板
- 實時結果顯示
- 互動式圖表

---

A simple and intuitive web application for exploring linear regression concepts with real-time parameter adjustment and outlier detection.

## 🌟 Features

- **Real-time Interactive Visualization**: Adjust parameters and see immediate changes
- **sklearn Linear Regression**: Professional machine learning implementation
- **Top 5 Outliers Detection**: Automatically identify and highlight anomalous data points
- **Model Coefficients Comparison**: Compare true vs estimated parameters
- **Simple Interface**: Just 3 parameter controls for easy use

## 🎯 What This Tool Does

This application generates synthetic linear data based on the equation:
```
y = ax + b + noise
```

Where:
- `a` is the slope coefficient (adjustable)
- `b` is the intercept (fixed at 30)
- `noise` follows Gaussian distribution N(0, variance)

## 🚀 Quick Start


### 💻 Local Development
```bash
# Clone or download the project
cd linear_regression_app

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

Open your browser and visit: `http://localhost:8501`

## 🎛️ How to Use

### Parameter Controls (Left Panel)
1. **Number of data points (n)**:
   - Range: 10 - 1000
   - Controls how many data points to generate

2. **Coefficient 'a' (y = ax + b + noise)**:
   - Range: -10 to 10
   - Controls the slope of the linear relationship

3. **Noise Variance (var)**:
   - Range: 0 - 5000
   - Controls data scatter (higher = more scattered)

### Main Display
- **Scatter Plot**: Blue dots showing generated data points
- **Regression Line**: Red line showing sklearn's fitted linear regression
- **Top 5 Outliers**: Red X markers labeled O1-O5 for most anomalous points

### Results Tables (Below Chart)
- **Model Coefficients**: Compare true vs estimated parameters
- **Top 5 Outliers**: Detailed information about anomalous data points

## 📊 Understanding the Output

### Model Coefficients Table
| Coefficient | True Value | Estimated Value |
|-------------|------------|-----------------|
| Slope (a) | Your setting | sklearn result |
| Intercept (b) | 30.0000 | sklearn result |
| R² Score | - | Model performance |

### Top 5 Outliers Table
| Rank | X Value | Y Value | Residual | Score |
|------|---------|---------|----------|-------|
| O1-O5 | X coord | Y coord | Error | Outlier strength |

## 🎓 Educational Use

This tool is perfect for:
- Understanding linear regression fundamentals
- Exploring the effect of noise on model performance
- Learning about outlier detection
- Visualizing how sample size affects model quality
- Comparing true vs estimated parameters

## ⚡ Real-time Interaction

- **Adjust any parameter** → Chart updates automatically
- **No buttons to click** → Immediate visual feedback
- **Interactive exploration** → Try different combinations

## 🔧 Technical Details

- **Frontend**: Streamlit
- **ML Library**: scikit-learn LinearRegression
- **Visualization**: Plotly for interactive charts
- **Outlier Detection**: Standardized residuals method

## 📁 Project Structure

```
linear_regression_app/
├── app.py                    # Main Streamlit application
├── src/                      # Core modules
│   ├── data_generator.py     # Synthetic data generation
│   ├── model_trainer.py      # sklearn model training
│   ├── evaluator.py          # Metrics and outlier detection
│   └── visualizer.py         # Interactive plotting
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🌐 Deployment

Ready for deployment on:
- **Streamlit Community Cloud** (recommended)
- **Heroku**
- **Railway**
- Any Python hosting platform

### GitHub + Streamlit Cloud Setup
1. Upload to GitHub repository
2. Connect to Streamlit Cloud
3. Set main file: `app.py`
4. Deploy automatically

## 🎯 Assignment Features

Perfect for coursework requiring:
- [x] sklearn linear regression implementation
- [x] Interactive parameter adjustment
- [x] Red regression line visualization
- [x] Top 5 outliers identification and labeling
- [x] Real-time chart updates
- [x] Model coefficient analysis

## 💡 Tips for Best Results

- **Low noise (0-500)**: Clear linear relationship, high R²
- **Medium noise (500-2000)**: Realistic data with some scatter
- **High noise (2000+)**: Challenging conditions, low R²
- **More data points**: Generally better parameter estimation
- **Different slopes**: See how angle affects outlier detection

## 🤔 Troubleshooting

**App won't start?**
- Check Python version (3.7+ required)
- Verify all dependencies installed: `pip install -r requirements.txt`

**Charts not updating?**
- Refresh the page (Ctrl+R)
- Check browser console for errors

**Performance issues?**
- Reduce number of data points
- Use modern browser (Chrome, Firefox, Safari)

---

**Created for**: Educational purposes and linear regression analysis
**License**: Open source
**Support**: Check GitHub issues page