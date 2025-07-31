import streamlit as st

st.set_page_config(page_title="Multi-Armed Bandit Dashboard")
st.title("🎯 Multi-Armed Bandit Dashboard")

# Description of algorithms with MathJax support via $...$ and $$...$$
markdown_text = r"""
This is a demo dashboard for three classic multi-armed bandit algorithms:

1. **Epsilon-Greedy**  
   At each step, with probability $\varepsilon$ we choose a random arm (explore), otherwise we select the arm with the highest estimated mean reward (exploit).

2. **UCB1 (Upper Confidence Bound 1)**  
   We pick arm $i$ with the maximum value:  
   $$
     \hat{\mu}_i + \sqrt{\frac{2 \ln t}{n_i}}
   $$  
   where $\hat{\mu}_i$ is the estimated mean reward, $t$ is the total number of pulls, and $n_i$ is the number of times arm $i$ has been pulled.

3. **Thompson Sampling**  
   For each arm, maintain a Beta posterior $\mathrm{Beta}(\alpha,\beta)$.  
   Sample $\theta \sim \mathrm{Beta}(\alpha,\beta)$ and select the arm with the highest $\theta$.

---

**How to use:**  
1. Navigate to the desired algorithm tab.  
2. Adjust the parameters in the sidebar (number of arms, batch size, auto-run steps, $\varepsilon$/prior, etc.).  
3. Click **Pull Once**, **Pull ×N**, or enable **Auto-run** to see updates in:
   - The pull log
   - The RMSE chart
   - Estimated probabilities and pull counts
"""

st.markdown(markdown_text)