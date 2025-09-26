# ReinforcementLearningForAMM: Smart Liquidity Optimization in Decentralized Exchanges

This repository contains the complete implementation of **Reinforcement Learning (RL) frameworks** for optimizing **Automated Market Makers (AMMs)** in decentralized exchanges, with a focus on **Uniswap V3 and V4**. The project was developed as part of a bachelor thesis titled *"Smart Liquidity: Reinforcement Learning Approaches for AMM Optimization in Decentralized Exchanges"*.

## 📌 Overview

The goal of this project is to automate and optimize liquidity provisioning in AMMs using two advanced RL algorithms:

- **Dueling Double Deep Q-Network (DDQN)**
- **Proximal Policy Optimization (PPO)**

Both models are enhanced with an **LSTM-based price forecasting module** to enable proactive liquidity management. The frameworks are evaluated in a custom-built simulation environment that replicates Uniswap V3 and V4 mechanics—including **concentrated liquidity** and **programmable hooks**.

---

## 🧠 Key Features

- **Hybrid RL-LSTM Architecture**: Combines sequential price prediction with reinforcement learning for adaptive liquidity strategies.
- **Uniswap V3 & V4 Simulation**: Supports concentrated liquidity (V3) and dynamic hooks (V4) like `DynamicFee` and `RebalanceHook`.
- **Comprehensive Metrics**: Tracks slippage, divergence loss, capital utilization, and cumulative rewards.
- **Modular Design**: Separate implementations for DDQN and PPO with shared environment and data preprocessing.

---

## 📂 Repository Structure

```
ReinforcementLearningForAMM/
├── DDQN/                 # Dueling DDQN implementation for Uniswap V3 & V4
├── PPO/                  # PPO implementation for Uniswap V3 & V4
├── LSTM/                 # LSTM price prediction model
├── AMM_Simulator/        # Custom AMM environment with V3/V4 logic
├── data/                 # Preprocessed ETH/USDT 1-hour kline data
├── graphs/               # Training curves and evaluation metrics
├── utils/                # Helpers for data loading, normalization, etc.
└── README.md
```

---

## 📊 Dataset

The models are trained and evaluated on **historical ETH/USDT 1-hour kline data**, sourced from:

👉 [Ethereum Historical Dataset (Kaggle)](https://www.kaggle.com/datasets/prasoonkottarathil/ethereum-historical-dataset)

The dataset includes OHLC values, volume, and derived technical indicators (volatility, order flow, momentum, etc.).

---

## 🧪 Results Summary

| Model          | Liquidity Utilization (%) | Avg. Divergence Loss | Avg. Slippage |
|----------------|---------------------------|----------------------|---------------|
| Baseline AMM   | 56.00                     | 1.465                | 0.4779        |
| DDQN V3        | 62.85                     | 2.98                 | 0.0298        |
| PPO V3         | 68.65                     | 6.74                 | 0.0674        |
| **DDQN V4**    | **90.61**                 | **0.0017**           | 1.6292        |
| **PPO V4**     | **91.43**                 | **0.0008**           | 1.1722        |

- **DDQN V4** achieves the highest total reward but exhibits higher slippage.
- **PPO V4** offers greater stability and near-elimination of divergence loss.
- Both RL frameworks significantly outperform the static baseline.

---

## ⚙️ Installation & Requirements

### Dependencies
- Python 3.10+
- PyTorch / PyTorch Lightning
- NumPy, Pandas, Matplotlib, Scikit-learn
- Gymnasium (OpenAI Gym compatible)

### Setup
```bash
git clone https://github.com/KaraosmanAhmetCemil/ReinforcementLearningForAMM.git
cd ReinforcementLearningForAMM
pip install -r requirements.txt
```

---

## 🚀 Usage

### Training a Model
Example for training DDQN on Uniswap V3:
```bash
python DDQN/train_ddqn_v3.py
```

### Evaluating a Model
```bash
python DDQN/test_ddqn_v3.py
```

### Visualizing Results
See the `graphs/` directory for pre-generated plots, or use the provided plotting scripts to regenerate them.

---

## 📖 Thesis Reference

This code accompanies the bachelor thesis:  
**"Smart Liquidity: Reinforcement Learning Approaches for AMM Optimization in Decentralized Exchanges"**  
*Ahmet Cemil Karaosman, OST – Eastern Switzerland University of Applied Sciences, 2025*

For detailed methodology, theoretical background, and experimental analysis, refer to the full thesis document.

---

## 👨‍💻 Author

**Ahmet Cemil Karaosman**  
- Institute for Computational Engineering, OST  
- [GitHub Profile](https://github.com/KaraosmanAhmetCemil)

---

## 📄 License

This project is intended for academic and research purposes. Please cite the thesis if you use this code in your work.

---


*This repository is part of an academic research project. Use responsibly and acknowledge the source when referencing.*
