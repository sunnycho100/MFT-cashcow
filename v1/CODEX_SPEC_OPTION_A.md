# Technical Specification: Option A - Hybrid Deep Learning + Reinforcement Learning Architecture

**Project:** Crypto Quantitative Trading System  
**Spec Version:** 2.0 (Option A)  
**Date:** February 13, 2026  
**Target:** Codex 5.3 Implementation  

---

## 🎯 Overview

Upgrade the existing crypto trading system with state-of-the-art deep learning models:

1. **Temporal Fusion Transformer (TFT)** - Multi-horizon price forecasting
2. **LSTM-CNN Hybrid** - Pattern recognition + sequential learning
3. **Sentiment Analyzer** - News/social media sentiment (optional/placeholder for now)
4. **PPO Reinforcement Learning Agent** - Learns optimal trading strategy
5. **Enhanced Ensemble** - Intelligent signal aggregation

**Key Requirements:**
- Must integrate with existing system (`config.yaml`, data fetcher, dashboard)
- Training should take 6-12 hours on M4 Mac Mini (utilize GPU/Neural Engine)
- Models save checkpoints and can resume training
- Produces trading signals compatible with current dashboard format
- Includes comprehensive logging for training progress

---

## 📁 File Structure

```
crypto-trader/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── temporal_fusion_transformer.py  # NEW
│   │   ├── lstm_cnn_hybrid.py              # NEW
│   │   ├── sentiment_analyzer.py           # NEW (placeholder)
│   │   ├── reinforcement_learning.py       # NEW
│   │   ├── enhanced_ensemble.py            # NEW
│   │   ├── mean_reversion.py              # KEEP (existing)
│   │   ├── momentum.py                     # KEEP (existing)
│   │   └── ml_model.py                     # KEEP (existing XGBoost)
│   └── ...
├── scripts/
│   ├── train_deep_models.py                # NEW - Main training script
│   ├── evaluate_models.py                  # NEW - Backtesting/evaluation
│   └── run_paper.py                        # UPDATE - Use new models
├── checkpoints/                             # NEW - Model weights
│   ├── tft/
│   ├── lstm_cnn/
│   └── rl_agent/
├── logs/
│   └── training/                           # NEW - Training logs
├── config.yaml                             # UPDATE - Add deep learning configs
├── requirements.txt                        # UPDATE - Add PyTorch, etc.
└── CODEX_SPEC_OPTION_A.md                 # This file
```

---

## 🔧 Dependencies to Add

Update `requirements.txt` with:

```
# Deep Learning
torch>=2.0.0
pytorch-forecasting>=1.0.0    # For TFT implementation
lightning>=2.0.0               # PyTorch Lightning for training
tensorboard>=2.15.0            # Training visualization

# Reinforcement Learning
stable-baselines3>=2.0.0      # PPO implementation
gymnasium>=0.29.0             # RL environment

# Additional utilities
transformers>=4.35.0          # For future sentiment analysis
sentencepiece>=0.1.99         # Tokenization
```

**Important:** Check M4 Mac compatibility - use MPS (Metal Performance Shaders) backend for GPU acceleration.

---

## 📊 Model 1: Temporal Fusion Transformer (TFT)

**File:** `src/models/temporal_fusion_transformer.py`

### Purpose
Multi-horizon forecasting for crypto prices. Predicts next 1h, 4h, 24h simultaneously with interpretable attention.

### Architecture
```python
class TemporalFusionTransformer:
    """
    Multi-horizon time-series forecasting using TFT architecture.
    
    Key features:
    - Variable selection network (learns which features matter)
    - LSTM encoder for historical context
    - Multi-head attention for long-term dependencies
    - Quantile regression for uncertainty estimation
    
    Input: Historical OHLCV + features (last 168 hours = 1 week)
    Output: Price predictions at [1h, 4h, 24h] horizons + confidence intervals
    """
```

### Key Parameters
- **Context length:** 168 hours (1 week of hourly data)
- **Prediction horizons:** [1, 4, 24] hours
- **Hidden size:** 128
- **Attention heads:** 4
- **Dropout:** 0.1
- **Quantiles:** [0.1, 0.5, 0.9] for uncertainty

### Training Configuration
- **Batch size:** 64
- **Learning rate:** 1e-3 (with ReduceLROnPlateau)
- **Epochs:** 100 (early stopping on validation loss)
- **Loss:** Quantile loss (handles uncertainty)
- **Optimizer:** Adam with weight decay 1e-5

### Expected Training Time
- **Initial training:** 3-4 hours on M4 Mac Mini
- **Per epoch:** ~2-3 minutes

### Integration
```python
def fit(self, data: Dict[str, pd.DataFrame]) -> None:
    """
    Train TFT on historical OHLCV data.
    
    Args:
        data: Dict of {pair: DataFrame} with OHLCV columns
    """
    
def predict(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Generate multi-horizon predictions.
    
    Returns:
        {
            'predictions': {
                '1h': {'mean': float, 'lower': float, 'upper': float},
                '4h': {...},
                '24h': {...}
            },
            'attention_weights': {...},  # Feature importance
            'confidence': float
        }
    """
    
def generate_signals(self, predictions: Dict) -> List[Dict]:
    """
    Convert predictions to trading signals.
    
    Logic:
    - BUY if 4h prediction > current price * 1.02 AND confidence > 0.6
    - SELL if 4h prediction < current price * 0.98 AND confidence > 0.6
    - HOLD otherwise
    """
```

### Feature Engineering
Use existing features from `ml_model.py` plus:
- Hour of day (cyclical encoding)
- Day of week (cyclical encoding)
- Rolling volatility (multiple windows)
- Volume profile
- Price momentum indicators

---

## 🧠 Model 2: LSTM-CNN Hybrid

**File:** `src/models/lstm_cnn_hybrid.py`

### Purpose
Pattern recognition + sequential learning. CNN extracts visual patterns from price charts, LSTM captures temporal dependencies.

### Architecture
```python
class LSTMCNNHybrid(nn.Module):
    """
    Hybrid deep learning model combining:
    - 1D CNN for pattern extraction (like candlestick patterns)
    - LSTM for sequential dependencies
    - Fully connected layers for final prediction
    
    Input: 
        - Raw OHLCV sequences (last 48 hours)
        - Extracted features (RSI, MACD, etc.)
    Output:
        - Direction probability (0-1, where >0.5 = bullish)
        - Confidence score
    """
    
    def __init__(self):
        # CNN branch (pattern extraction)
        self.cnn = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=3),  # 5 = OHLCV
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # LSTM branch (temporal)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        
        # Fusion + classifier
        self.fc = nn.Sequential(
            nn.Linear(128 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
```

### Key Parameters
- **Sequence length:** 48 hours
- **CNN filters:** [64, 128]
- **LSTM hidden:** 128
- **LSTM layers:** 2
- **Dropout:** 0.2-0.3

### Training Configuration
- **Batch size:** 128
- **Learning rate:** 1e-4
- **Epochs:** 50
- **Loss:** Binary cross-entropy with class weights (handle imbalanced data)
- **Optimizer:** AdamW

### Expected Training Time
- **Initial training:** 1-2 hours
- **Per epoch:** ~2 minutes

### Integration
```python
def fit(self, data: Dict[str, pd.DataFrame]) -> None:
    """Train on historical data with direction labels."""
    
def predict(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Returns:
        {
            'direction': 'BUY' | 'SELL' | 'HOLD',
            'probability': float,  # 0-1
            'confidence': float    # Model certainty
        }
    """
    
def generate_signals(self, predictions: Dict) -> List[Dict]:
    """
    Signal generation:
    - BUY if probability > 0.65 and confidence > 0.7
    - SELL if probability < 0.35 and confidence > 0.7
    - HOLD otherwise
    """
```

---

## 📰 Model 3: Sentiment Analyzer (Placeholder)

**File:** `src/models/sentiment_analyzer.py`

### Purpose
Analyze market sentiment from news/social media. **For initial implementation, use a simple placeholder that returns neutral sentiment.**

### Placeholder Implementation
```python
class SentimentAnalyzer:
    """
    Sentiment analysis for crypto news/social media.
    
    INITIAL IMPLEMENTATION: Placeholder that returns neutral.
    FUTURE: Integrate news APIs, Twitter, Reddit sentiment.
    """
    
    def __init__(self):
        self.enabled = False  # Disabled by default
        
    def fit(self, data: Dict[str, pd.DataFrame]) -> None:
        """No training needed for placeholder."""
        pass
        
    def predict(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Returns:
            {
                'sentiment': 'neutral',  # Always neutral for now
                'score': 0.5,
                'confidence': 0.5
            }
        """
        return {
            'sentiment': 'neutral',
            'score': 0.5,
            'confidence': 0.5
        }
    
    def generate_signals(self, predictions: Dict) -> List[Dict]:
        """No signals from placeholder."""
        return []
```

**Note:** This can be upgraded later with:
- News API integration (Finviz, CryptoPanic)
- Twitter sentiment via API
- Reddit WSB/cryptocurrency analysis
- BERT-based sentiment classifier

---

## 🤖 Model 4: PPO Reinforcement Learning Agent

**File:** `src/models/reinforcement_learning.py`

### Purpose
Learn optimal trading strategy by combining signals from all other models. Uses PPO (Proximal Policy Optimization) to maximize profit while managing risk.

### Architecture
```python
class CryptoTradingEnv(gymnasium.Env):
    """
    Custom Gym environment for crypto trading.
    
    State space:
        - TFT predictions (multi-horizon)
        - LSTM-CNN direction probability
        - Current position (long/short/neutral)
        - Portfolio value
        - Market features (volatility, volume, etc.)
    
    Action space:
        - Continuous: Position size [-1.0, 1.0]
          -1.0 = max short, 0 = neutral, 1.0 = max long
    
    Reward:
        - Profit/loss (scaled)
        - Sharpe ratio component
        - Penalty for excessive trading (transaction costs)
        - Drawdown penalty
    """
    
class PPOTradingAgent:
    """
    PPO agent that learns from model signals.
    
    Uses stable-baselines3 PPO implementation.
    """
    
    def __init__(self):
        self.env = CryptoTradingEnv()
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1
        )
```

### Training Configuration
- **Total timesteps:** 500,000 (can increase for better performance)
- **Environment steps per update:** 2048
- **Learning rate:** 3e-4
- **Discount factor (gamma):** 0.99

### Expected Training Time
- **Initial training:** 2-4 hours
- **Continuous learning:** Can update incrementally

### Integration
```python
def fit(self, historical_data: Dict, model_predictions: Dict) -> None:
    """
    Train RL agent in simulated environment.
    
    Args:
        historical_data: Historical OHLCV
        model_predictions: Cached predictions from TFT + LSTM-CNN
    """
    
def predict(self, current_state: Dict) -> Dict[str, Any]:
    """
    Determine optimal action given current market state.
    
    Returns:
        {
            'action': float,  # -1 to 1 (position size)
            'confidence': float,
            'expected_return': float
        }
    """
    
def generate_signals(self, action: Dict) -> List[Dict]:
    """
    Convert RL action to trading signal.
    
    Logic:
    - If action > 0.3: BUY signal (strength = action value)
    - If action < -0.3: SELL signal (strength = abs(action))
    - Otherwise: HOLD
    """
```

---

## 🎭 Model 5: Enhanced Ensemble

**File:** `src/models/enhanced_ensemble.py`

### Purpose
Intelligently combine signals from all models (TFT, LSTM-CNN, RL, plus existing mean reversion, momentum, XGBoost).

### Architecture
```python
class EnhancedEnsemble:
    """
    Meta-model that combines all signals with learned weights.
    
    Unlike simple voting, this uses:
    - Confidence-weighted averaging
    - Adaptive weights based on recent model performance
    - Correlation-aware signal combination
    """
    
    def __init__(self):
        self.models = {
            'tft': TemporalFusionTransformer(),
            'lstm_cnn': LSTMCNNHybrid(),
            'rl_agent': PPOTradingAgent(),
            'mean_reversion': MeanReversionModel(),  # Existing
            'momentum': MomentumModel(),              # Existing
            'ml_model': MLModel()                     # Existing XGBoost
        }
        
        # Learned weights (can be trained or rule-based)
        self.weights = {
            'tft': 0.25,
            'lstm_cnn': 0.20,
            'rl_agent': 0.30,  # Highest weight - it learns from others
            'mean_reversion': 0.10,
            'momentum': 0.10,
            'ml_model': 0.05
        }
```

### Signal Combination Logic
```python
def combine_signals(self, all_signals: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Intelligent signal fusion:
    
    1. Weight each model's signals by:
       - Base weight (self.weights)
       - Recent performance (Sharpe ratio over last 30 days)
       - Signal confidence
    
    2. Aggregate:
       - If weighted vote > 0.6: BUY
       - If weighted vote < -0.6: SELL
       - Otherwise: HOLD
    
    3. Position sizing:
       - Scale by aggregate confidence
       - Reduce size if models disagree (high variance)
    """
```

---

## 🏗️ Main Training Script

**File:** `scripts/train_deep_models.py`

### Purpose
Orchestrate training of all deep learning models with proper logging and checkpointing.

### Implementation
```python
"""
Main training pipeline for Option A deep learning models.

Usage:
    python scripts/train_deep_models.py --mode full --epochs 100
    python scripts/train_deep_models.py --mode quick --epochs 10  # Fast test
    python scripts/train_deep_models.py --mode resume --checkpoint latest
"""

import argparse
import logging
from pathlib import Path
import torch
from src.models.temporal_fusion_transformer import TemporalFusionTransformer
from src.models.lstm_cnn_hybrid import LSTMCNNHybrid
from src.models.reinforcement_learning import PPOTradingAgent
from src.data.fetcher import fetch_data

def setup_logging():
    """Configure training logs."""
    log_dir = Path("logs/training")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"),
            logging.StreamHandler()
        ]
    )

def train_tft(data, epochs=100):
    """Train Temporal Fusion Transformer."""
    logger = logging.getLogger("TFT")
    logger.info("Starting TFT training...")
    
    model = TemporalFusionTransformer()
    
    for epoch in range(epochs):
        loss = model.train_epoch(data)
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.save_checkpoint(f"checkpoints/tft/epoch_{epoch+1}.pt")
    
    logger.info("TFT training complete!")
    return model

def train_lstm_cnn(data, epochs=50):
    """Train LSTM-CNN hybrid."""
    logger = logging.getLogger("LSTM_CNN")
    logger.info("Starting LSTM-CNN training...")
    
    model = LSTMCNNHybrid()
    
    for epoch in range(epochs):
        metrics = model.train_epoch(data)
        logger.info(f"Epoch {epoch+1}/{epochs} | "
                   f"Loss: {metrics['loss']:.6f} | "
                   f"Accuracy: {metrics['accuracy']:.4f}")
        
        if (epoch + 1) % 10 == 0:
            model.save_checkpoint(f"checkpoints/lstm_cnn/epoch_{epoch+1}.pt")
    
    logger.info("LSTM-CNN training complete!")
    return model

def train_rl_agent(data, models, timesteps=500000):
    """Train RL agent using predictions from other models."""
    logger = logging.getLogger("RL_Agent")
    logger.info("Starting RL agent training...")
    
    # Pre-compute predictions from TFT and LSTM-CNN
    logger.info("Generating predictions for RL training environment...")
    tft_preds = models['tft'].predict_all(data)
    lstm_preds = models['lstm_cnn'].predict_all(data)
    
    agent = PPOTradingAgent()
    agent.fit(data, tft_preds, lstm_preds, total_timesteps=timesteps)
    
    logger.info("RL agent training complete!")
    return agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'quick', 'resume'], default='full')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger("Main")
    
    logger.info(f"Training mode: {args.mode}")
    logger.info(f"Device: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}")
    
    # Load data
    logger.info("Loading training data...")
    data = load_training_data()  # Load from data/historical/
    logger.info(f"Loaded {sum(len(df) for df in data.values())} total rows")
    
    # Adjust epochs for quick mode
    if args.mode == 'quick':
        tft_epochs = 10
        lstm_epochs = 5
        rl_timesteps = 50000
    else:
        tft_epochs = args.epochs
        lstm_epochs = args.epochs // 2
        rl_timesteps = 500000
    
    # Train models sequentially
    models = {}
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: Training Temporal Fusion Transformer")
    logger.info("="*60)
    models['tft'] = train_tft(data, epochs=tft_epochs)
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: Training LSTM-CNN Hybrid")
    logger.info("="*60)
    models['lstm_cnn'] = train_lstm_cnn(data, epochs=lstm_epochs)
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 3: Training RL Agent")
    logger.info("="*60)
    models['rl_agent'] = train_rl_agent(data, models, timesteps=rl_timesteps)
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete! Models saved to checkpoints/")
    logger.info("="*60)

if __name__ == "__main__":
    main()
```

### Logging Output Format
```
2026-02-13 18:00:00 | Main | INFO | Training mode: full
2026-02-13 18:00:00 | Main | INFO | Device: mps
2026-02-13 18:00:05 | Main | INFO | Loading training data...
2026-02-13 18:00:10 | Main | INFO | Loaded 6420 total rows
2026-02-13 18:00:10 | TFT | INFO | Starting TFT training...
2026-02-13 18:02:15 | TFT | INFO | Epoch 1/100 | Loss: 0.045231
2026-02-13 18:04:20 | TFT | INFO | Epoch 2/100 | Loss: 0.039847
...
```

---

## 🔧 Configuration Updates

**File:** `config.yaml`

Add new section:

```yaml
# Deep Learning Models (Option A)
deep_learning:
  enabled: true
  
  # Device config
  device: "mps"  # "mps" for M4 Mac, "cuda" for NVIDIA, "cpu" fallback
  
  # Temporal Fusion Transformer
  tft:
    enabled: true
    context_length: 168  # hours
    prediction_horizons: [1, 4, 24]  # hours
    hidden_size: 128
    attention_heads: 4
    dropout: 0.1
    learning_rate: 0.001
    batch_size: 64
    epochs: 100
    
  # LSTM-CNN Hybrid
  lstm_cnn:
    enabled: true
    sequence_length: 48  # hours
    lstm_hidden: 128
    lstm_layers: 2
    dropout: 0.2
    learning_rate: 0.0001
    batch_size: 128
    epochs: 50
    
  # Reinforcement Learning
  rl_agent:
    enabled: true
    algorithm: "PPO"
    total_timesteps: 500000
    learning_rate: 0.0003
    gamma: 0.99
    
  # Ensemble weights
  ensemble:
    weights:
      tft: 0.25
      lstm_cnn: 0.20
      rl_agent: 0.30
      mean_reversion: 0.10
      momentum: 0.10
      ml_model: 0.05
```

---

## 🎯 Success Criteria

**Phase 1 (Implementation):**
- ✅ All 5 model files created and runnable
- ✅ Training script executes without errors
- ✅ Models save/load checkpoints properly
- ✅ Logging outputs to console and file

**Phase 2 (Training):**
- ✅ TFT trains for 3-4 hours, loss decreases
- ✅ LSTM-CNN trains for 1-2 hours, accuracy improves
- ✅ RL agent trains for 2-4 hours, reward increases
- ✅ Total training time: 6-12 hours
- ✅ GPU utilization: >80% during training

**Phase 3 (Integration):**
- ✅ Models generate signals in dashboard format
- ✅ Ensemble combines signals intelligently
- ✅ Dashboard displays new model predictions
- ✅ Paper trading script uses new models

---

## 🚨 Important Notes for Codex

1. **M4 Mac GPU:** Use `torch.device("mps")` for Metal Performance Shaders
2. **Memory:** Models should fit in 16GB RAM. Use gradient checkpointing if needed.
3. **Data format:** Reuse existing data loading from `src/data/fetcher.py`
4. **Compatibility:** Must work with existing `test_dashboard.py` and signal format
5. **Error handling:** Graceful fallback if GPU not available (use CPU)
6. **Checkpointing:** Save every 10 epochs to allow resume
7. **Logging:** Verbose output for monitoring training progress
8. **Testing:** Include quick mode for fast validation (10 epochs instead of 100)

---

## 📝 Testing Instructions

After implementation:

```bash
# Quick test (10-15 minutes)
python scripts/train_deep_models.py --mode quick

# Full training (6-12 hours)
python scripts/train_deep_models.py --mode full

# Resume from checkpoint
python scripts/train_deep_models.py --mode resume --checkpoint checkpoints/tft/epoch_50.pt
```

---

## 🎓 References

- TFT paper: https://arxiv.org/abs/1912.09363
- PyTorch Forecasting: https://pytorch-forecasting.readthedocs.io/
- Stable Baselines3: https://stable-baselines3.readthedocs.io/
- Research findings: See `ML_RESEARCH_FINDINGS.md`

---

**End of Specification**

Codex: Please implement this architecture following PyTorch best practices. Focus on clean, modular code with comprehensive error handling and logging. The training should maximize M4 GPU utilization while remaining stable for overnight runs.
