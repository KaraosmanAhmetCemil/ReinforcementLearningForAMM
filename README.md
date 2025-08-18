# ReinforcementLearningForAMM

Dieses Repository enthält die Implementierungen von **Reinforcement-Learning-Frameworks** zur Optimierung von **Automated Market Makers (AMMs)** auf Basis von **Uniswap V3 und V4**.  

Im Fokus stehen zwei Algorithmen:
- **Dueling Double Deep Q-Network (DDQN)**
- **Proximal Policy Optimization (PPO)**

Beide Ansätze wurden in einer simulierten Umgebung evaluiert, um **Slippage**, **Divergence Loss**, **Liquidity Utilization** und **Rewards** zu analysieren.  

---

## 📂 Struktur des Repositories
- `DDQN/` – Implementierung des Dueling DDQN-Frameworks für Uniswap V3 und V4  
- `PPO/` – Implementierung des PPO-Frameworks für Uniswap V3 und V4  
- `graphs/` – Visualisierungen der Trainingsergebnisse (Reward-Verläufe, Slippage, Divergence Loss etc.)  

---

## 📊 Datengrundlage
Für Training und Evaluation wurde der folgende Datensatz mit **ETH/USDT 1-Stunden-Klines** verwendet:  
👉 [Ethereum Historical Dataset (Kaggle)](https://www.kaggle.com/datasets/prasoonkottarathil/ethereum-historical-dataset)

Die Daten wurden preprocessiert und als Input für **LSTM-Preisprognosen** sowie für die **RL-Agenten** genutzt.  

---

## ⚙️ Anforderungen
- Python 3.10+  
- PyTorch / PyTorch Lightning  
- NumPy, Pandas, Matplotlib  
- Gymnasium (oder OpenAI Gym)  

---

## 🚀 Ausführung
1. Repository klonen:
   ```bash
   git clone https://github.com/KaraosmanAhmetCemil/ReinforcementLearningForAMM.git
   cd ReinforcementLearningForAMM
