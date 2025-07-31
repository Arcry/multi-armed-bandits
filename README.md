# Multi-Armed Bandit Dashboard

This repository contains a Streamlit application demonstrating three classic multi-armed bandit algorithms:

- **Epsilon-Greedy**  
- **UCB1 (Upper Confidence Bound 1)**  
- **Thompson Sampling**

---

## ▶️Running the App

You have two options:

1. Recommended: Run a specific page to avoid the default Home.py stub:
   ```bash
   streamlit run pages/EpsilonGreedy.py
   ```
2. Run the entire app, which will start with the Home page:
   ```bash
    streamlit run Home.py
    ```
## 📈 Features

- Interactive UI: Adjust batch size, auto-run steps, delays, and algorithm-specific parameters.

- Live charts: Scatter plot of rewards, smoothed RMSE, bar charts for estimates, JSON for counts.

- Logging: Each pull is logged to CSV with timestamp, enabling offline analysis.

- Warm-up: Thompson Sampling page automatically pulls each arm once to initialize the posterior.