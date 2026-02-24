#  FinSentLSTM — Multimodal Stock Direction Predictor

> **Dual-branch LSTM that fuses stock price sequences + news sentiment to predict next-day price direction.**
> Built with PyTorch · Streamlit · Yahoo Finance API
## Architecture
```
Price Data (30 days OHLCV)          News Headline (tokens)
         │                                    │
    ┌────▼────┐                         ┌─────▼────┐
    │  Price  │                         │Embedding │
    │  LSTM   │                         │  Layer   │
    │ 2 layers│                         │  (64-d)  │
    │ h=128   │                         └─────┬────┘
    └────┬────┘                               │
         │ 128-d                         ┌────▼─────┐
         │                               │ Sentiment│
         │                               │  LSTM    │
         │                               │  h=64    │
         │                               └────┬─────┘
         │ 128-d                              │ 64-d
         └──────────────┬─────────────────────┘
                        │ Concat (192-d)
                   ┌────▼─────┐
                   │ FC Head  │
                   │192→128→  │
                   │  64→2    │
                   └────┬─────┘
                        │
                   [DOWN] [UP]
```
##  Features

- **Multimodal fusion** — price time-series + NLP sentiment in one model
- **Explainable** — probability scores for UP and DOWN
- **Real-time data** — pulls live prices from Yahoo Finance
- **Early stopping + LR scheduling** — no overfitting
- **Class imbalance handling** — weighted CrossEntropy
- **Gradient clipping** — stable LSTM training
- **Beautiful Streamlit UI** — dark fintech dashboard with Plotly charts
- **Colab ready** — GPU training notebook included

## 📁 Project Structure
fintech-stock-predictor/
│
├── src/
│   ├── model.py          ← FinSentLSTM architecture (PriceLSTM + SentimentLSTM + Fusion)
│   ├── data_pipeline.py  ← Download, normalize, tokenize, DataLoader
│   ├── train.py          ← Full training loop with early stopping
│   └── inference.py      ← Live prediction + chart data
│
├── notebooks/
│   └── train_colab.ipynb ← Step-by-step Google Colab notebook
│
├── models/               ← Saved weights + vocab + history (auto-created)
│   ├── best_model.pt
│   ├── vocab.json
│   └── history.json
│
├── app.py                ← Streamlit dashboard
├── requirements.txt
└── README.md
```

##  Model Performance
| Metric    | Value  |
|-----------|--------|
| Val Accuracy  | ~57-63% |
| Val F1 Score  | ~0.55-0.62 |
| Architecture  | Dual LSTM Fusion |
| Parameters    | ~1.5M |
| Training Time | ~10 min (Colab T4) |
> Note: Stock direction prediction is inherently noisy. 55%+ sustained accuracy is considered strong.
## 🧪 Supported Tickers
Any ticker available on Yahoo Finance: `AAPL`, `TSLA`, `GOOGL`, `MSFT`, `AMZN`, `NVDA`, `META`, `RELIANCE.NS`, etc.
##  Key ML Concepts Demonstrated
- Multimodal deep learning (price + NLP fusion)
- Stacked LSTM with dropout regularization
- Gradient clipping for sequence model stability
- Class imbalance handling (weighted loss)
- Transfer of knowledge: NLP tokenization + price normalization
- Early stopping with model checkpointing
- ReduceLROnPlateau scheduling
##  Disclaimer
This project is for **educational purposes only**. It is not financial advice. Do not make real investment decisions based on model predictions.
