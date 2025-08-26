Real-Time Crpto Market Regime Classification with Deep Learning
=========================================================

A sophisticated, real-time pipeline that uses a multi-timeframe LSTM model to classify cryptocurrency market behavior (e.g., Trend, Range, Volatility Spike) for BTCUSDT, providing live predictions every minute.

### About This Project

This repository contains the complete code for a live inference pipeline that classifies the current market regime of BTCUSDT. The system fetches live data from Binance, processes it through a feature engineering pipeline that considers multiple timeframes (5m, 15m, 1h), and feeds the result into a trained LSTM (Long Short-Term Memory) neural network.

The primary goal is to provide a continuous, real-time signal that can be used to inform algorithmic trading strategies, risk management systems, or market analysis dashboards. For example, a trading algorithm might only take trend-following entries when the model predicts a Strong Trend regime, or tighten stop-losses when a Volatility Spike is predicted.

🔑 Key Features
---------------

*   **Real-Time Classification**: Runs continuously to provide minute-by-minute predictions of the current market state.
    
*   **Multi-Timeframe Analysis**: Enriches the primary 15m data with context from 5m (for immediate momentum) and 1h (for broader trend context) timeframes.
    
*   **Deep Learning Model**: Utilizes a Keras/TensorFlow LSTM model, ideal for learning from sequential time-series data.
    
*   **Robust Data Pipeline**: Features an intelligent data handler that fetches historical data by date range and seamlessly updates with the latest candle information in real-time.
    
*   **Two Operational Modes**:
    
    1.  **Scheduled Mode**: Runs predictions on fully closed candles at set intervals (e.g., every 5 minutes).
        
    2.  **Live Monitor Mode**: Provides "unofficial" predictions every minute by analyzing the in-progress candle.
        
*   **Extensible & Modular**: Code is organized into logical modules for data fetching, feature engineering, and inference, making it easy to adapt or extend.
    

⚙️ How It Works
---------------

The system operates through a streamlined, multi-stage process:

1.  **Data Prefill**: On startup, the pipeline fetches a 45-day historical dataset for the 5m, 15m, and 1h timeframes from the Binance API. This ensures all technical indicators have a sufficient "warm-up" period.
    
2.  **Live Data Update**: In each cycle, the pipeline fetches only the newest candle data since its last update, efficiently keeping the dataset current.
    
3.  **Timeframe Merging**: The 5m and 1h data are merged onto the primary 15m DataFrame. A merge\_asof operation aligns the timestamps, ensuring each 15-minute candle is enriched with the latest available data from the other timeframes.
    
4.  **Feature Engineering**: A comprehensive set of 36 features (12 for each timeframe) is calculated. These include indicators for trend, momentum, volatility, and price action, such as ema\_slope, adx, atr\_norm, and realized\_vol.
    
5.  **Scaling & Prediction**: The final feature set for the last 64 timesteps is scaled using the same scaler from training and fed into the LSTM model to generate a probability distribution across the six possible market regimes.
    
6.  **Output**: The predicted regime with the highest confidence is displayed, along with the probabilities for all other regimes.
    

🚀 Installation & Setup
-----------------------

Follow these steps to get the project running on your local machine.

**1\. Clone the Repository**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   git clone [https://github.com/akash-kumar5/Live-Market-Regime-Classifier](https://github.com/akash-kumar5/Live-Market-Regime-Classifier).git  cd YOUR_REPOSITORY_NAME   `

**2\. Create a Virtual Environment (Recommended)**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML``   python -m venv venv  source venv/bin/activate  # On Windows, use `venv\Scripts\activate`   ``

**3\. Install Dependencies**

The required libraries are listed in requirements.txt.

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install -r requirements.txt   `

_(Note: You will need to create a requirements.txt file by running pip freeze > requirements.txt in your terminal. Key libraries include tensorflow, pandas, scikit-learn, ta, requests, and schedule.)_

**4\. Place Model Artifacts**

Ensure your trained model and associated files are placed in the models/ directory:

*   lstm\_regime\_model.keras: The trained Keras model.
    
*   scaler.joblib: The Scikit-learn scaler object.
    
*   lstm\_model\_metadata.json: The metadata file containing feature lists and regime maps.
    

▶️ Usage
--------

The pipeline can be run in two distinct modes.

### Scheduled "Official" Predictions

This mode runs predictions at the start of every 5-minute candle, using only closed-candle data. This is the recommended mode for generating signals for a live trading algorithm.

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python main.py   `

### Live Minute-by-Minute Monitoring

This mode provides a continuous, "unofficial" prediction every minute by analyzing the live, in-progress candle. It's perfect for real-time observation and analysis.

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python live_inspect.py   `

🧠 Model Details
----------------

*   **Model Type**: LSTM (Long Short-Term Memory) Neural Network
    
*   **Time Steps**: 64 (The model looks at the features of the last 64 candles to make a prediction)
    
*   **Features**: 36 multi-timeframe features (see lstm\_model\_metadata.json for the full list).
    
*   **Predicted Regimes**:
    
    *   **Strong Trend**: Clear directional movement with strong momentum.
        
    *   **Weak Trend**: Directional movement, but with less momentum and more pullbacks.
        
    *   **Range**: Price is oscillating between clear support and resistance levels.
        
    *   **Squeeze**: Volatility has contracted to very low levels, often preceding a breakout.
        
    *   **Volatility Spike**: A sudden, sharp increase in price movement and volatility.
        
    *   **Choppy High-Vol**: High volatility but no clear direction; erratic price action.
        

💡 Future Improvements
----------------------

*   \[ \] **Strategy Backtesting**: Integrate a backtesting module to test the profitability of trading strategies based on the model's signals.
    
*   \[ \] **Cloud Deployment**: Deploy the pipeline to a cloud server (AWS, GCP) for 24/7 operation.
    
*   \[ \] **Real-Time Dashboard**: Create a web-based dashboard (e.g., using Dash or Streamlit) to visualize the live predictions and probabilities.
    
*   \[ \] **Multi-Asset Support**: Refactor the code to easily support and run classification for multiple cryptocurrency pairs simultaneously.
    
*   \[ \] **Model Retraining**: Implement a scheduled script to automatically retrain the model on new data to prevent model drift.
    

## Please ⭐ the repo. And follow for more similar projects.
