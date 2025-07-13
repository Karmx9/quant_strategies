# main.py
# --- This code is designed to be run on the QuantConnect platform ---

from AlgorithmImports import *
import numpy as np
import pandas as pd
# In a real system, you would use a library like TensorFlow/Keras.
# We will simulate the structure and output for backtesting efficiency.
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

class DeepLearningMultiFactorAlgorithm(QCAlgorithm):

    def Initialize(self):
        """
        Initial setup for the algorithm. Called once at the beginning.
        """
        self.SetStartDate(2022, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(1000000)

        # Define the S&P 500 universe
        self.sp500_symbol = self.AddEquity("SPY", Resolution.Hour).Symbol
        self.AddUniverse(self.Universe.ETF(self.sp500_symbol))
        
        # --- Algorithm Parameters ---
        self.momentum_lookback = 63
        self.mean_reversion_lookback = 21
        self.volatility_lookback = 21
        
        # --- Neural Network Simulation ---
        # In a real system, you would load pre-trained models here.
        # We will define their structure conceptually.
        self.price_model = self.CreatePricePredictionModel()
        self.risk_model = self.CreateRiskPredictionModel()
        
        # --- Scheduling ---
        # Schedule portfolio rebalancing to run monthly
        self.Schedule.On(self.DateRules.MonthStart(self.sp500_symbol), self.TimeRules.AfterMarketOpen(self.sp500_symbol, 5), self.RebalancePortfolio)
        # Schedule risk management to run weekly
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Friday), self.TimeRules.At(15, 0), self.ManagePortfolioRisk)
        
        self.current_constituents = set()

    def OnSecuritiesChanged(self, changes: SecurityChanges):
        """
        Event handler for when the constituents of our universe change.
        """
        for security in changes.AddedSecurities:
            self.current_constituents.add(security.Symbol)
        for security in changes.RemovedSecurities:
            if security.Symbol in self.current_constituents:
                self.current_constituents.remove(security.Symbol)
            if self.Portfolio[security.Symbol].Invested:
                self.Liquidate(security.Symbol, "Removed from S&P 500")

    def RebalancePortfolio(self):
        """
        Scheduled function to rebalance the portfolio based on the price prediction model.
        """
        self.Log(f"Starting monthly rebalance on {self.Time.date()}...")
        
        # 1. Get Predictions from our Price Prediction Model
        target_weights = self.get_price_model_targets()
        
        if not target_weights:
            self.Log("Price model did not provide target weights. Skipping rebalance.")
            return

        # 2. Execute Trades
        self.SetHoldings([PortfolioTarget(symbol, weight) for symbol, weight in target_weights.items()])
        self.Log("Rebalance based on price model complete.")

    def ManagePortfolioRisk(self):
        """
        Scheduled function to manage portfolio risk using the risk prediction model.
        """
        if not self.Portfolio.Invested:
            return

        # 1. Create feature vector for the risk model
        # The vector could include portfolio-level stats like total volatility, concentration, etc.
        portfolio_volatility = self.Portfolio.TotalPortfolioValue * self.Portfolio.TotalAbsoluteHoldingsCost * self.Portfolio.TotalHoldingsValue
        
        # In a real model, you'd have more features (correlation matrix, etc.)
        risk_features = [portfolio_volatility, len(self.Portfolio)]
        
        # 2. Get Prediction from our Risk Model
        # We simulate the model's prediction. Output is a risk score from 0 to 1.
        predicted_risk_score = self.risk_model_predict(risk_features)
        self.Log(f"Weekly Risk Check. Predicted Risk Score: {predicted_risk_score:.2f}")

        # 3. Take Action
        # If the predicted risk is too high, reduce portfolio leverage.
        if predicted_risk_score > 0.8: # High risk threshold
            self.Log(f"RISK ALERT: Predicted risk score {predicted_risk_score:.2f} exceeds threshold. Reducing leverage.")
            # Reduce total portfolio value by 25%
            self.SetHoldings(PortfolioTarget(self.sp500_symbol, 0.75)) # This is a simplified way to reduce overall exposure
        

    def get_price_model_targets(self):
        """
        This function simulates a neural network that combines factors to predict target weights.
        """
        if len(self.current_constituents) < 50: return {}

        symbols = list(self.current_constituents)
        history = self.History(symbols, self.momentum_lookback + 1, Resolution.Daily)
        if history.empty: return {}
        
        # --- Feature Engineering (Inputs to the Neural Network) ---
        features = {}
        for symbol in symbols:
            if symbol in history.index.levels[0]:
                hist_symbol = history.loc[symbol]['close']
                if len(hist_symbol) > self.momentum_lookback:
                    momentum = hist_symbol[-1] / hist_symbol[-self.momentum_lookback]
                    
                    lookback_data = hist_symbol[-self.mean_reversion_lookback:]
                    moving_avg = lookback_data.mean()
                    moving_std = lookback_data.std()
                    z_score = (lookback_data[-1] - moving_avg) / moving_std if moving_std > 0 else 0
                    
                    features[symbol] = {'Momentum': momentum, 'MeanReversion': z_score}

        if not features: return {}
        
        # --- Simulate NN Prediction ---
        # The NN would take these features and output a single score. We simulate this.
        df = pd.DataFrame.from_dict(features, orient='index')
        df.dropna(inplace=True)
        df['PredictionScore'] = df['Momentum'] - df['MeanReversion'] # Model learns to favor high momentum and low z-score
        
        sorted_by_score = df.sort_values(by='PredictionScore', ascending=False)
        
        # --- Strategy Execution ---
        num_stocks_to_trade = int(len(sorted_by_score) * 0.2)
        if num_stocks_to_trade == 0: return {}
        
        target_weights = {}
        long_weight = 0.8 / num_stocks_to_trade
        short_weight = -0.2 / num_stocks_to_trade
        
        for symbol in sorted_by_score.head(num_stocks_to_trade).index:
            target_weights[symbol] = long_weight
        for symbol in sorted_by_score.tail(num_stocks_to_trade).index:
            target_weights[symbol] = short_weight
            
        return target_weights

    # --- Conceptual Neural Network Definitions ---
    def CreatePricePredictionModel(self):
        # This is where you would define the architecture using Keras/TensorFlow
        # model = Sequential()
        # model.add(Dense(64, input_dim=2, activation='relu')) # 2 features: Momentum, MeanReversion
        # model.add(Dense(32, activation='relu'))
        # model.add(Dense(1, activation='linear')) # Output is a single prediction score
        # self.Compile(model)
        return "Conceptual Price Model" # Placeholder

    def CreateRiskPredictionModel(self):
        # This is where you would define the risk model architecture
        # model = Sequential()
        # model.add(Dense(32, input_dim=2, activation='relu')) # 2 features: Port. Vol, # Holdings
        # model.add(Dense(16, activation='relu'))
        # model.add(Dense(1, activation='sigmoid')) # Output is a risk score between 0 and 1
        # self.Compile(model)
        return "Conceptual Risk Model" # Placeholder

    def risk_model_predict(self, features):
        # In a real system, you'd use the loaded model: self.risk_model.predict(features)
        # We simulate the output based on the features for demonstration.
        # Simple simulation: risk increases with volatility and concentration.
        volatility_component = features[0]
        concentration_component = 1 / features[1] if features[1] > 0 else 1
        
        # Normalize and combine - this is a highly simplified simulation
        simulated_score = (volatility_component * 0.1 + concentration_component * 10) / 20
        return min(simulated_score, 1.0) # Cap score at 1.0

