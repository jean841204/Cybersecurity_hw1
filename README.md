# Linear Regression Interactive Analysis Tool

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

### Installation
```bash
# Clone or download the project
cd linear_regression_app

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Access
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