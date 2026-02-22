# Safe Reinforcement Learning: Dynamic Pricing Engine

## 📌 Project Overview
This repository contains a production-ready **Safe Reinforcement Learning (RL)** architecture for Dynamic Pricing in E-Commerce. It uses a Markov Decision Process (MDP) to optimize pricing strategies while strictly adhering to business logic constraints via deterministic **Action Masking**. Built and evaluated on the **Olist Brazilian E-Commerce Dataset**.

## 💼 Business Value & CFO Directives
Deploying RL in live retail environments carries massive financial risk if the agent hallucinates. This engine guarantees mathematical safety constraints:
- **Net Margin Expansion:** Projected +620,000 €/yr.
- **Margin Floors:** Hard cut-off at $Cost \times 1.10$. No neural network action can breach this limit.
- **Legacy ERP Fallbacks:** Protects against negative inventory synchronization issues and competitor API outages by selectively masking discrete discount brackets ($-10\%, -5\% \rightarrow -\infty$ logits).

## 🛠️ Technical Architecture
1. **Markov Decision Process (MDP):**
   - **State Space ($S$):** Normalized features preventing data leakage ($t_{expiry}$, competitor index, inventory).
   - **Action Space ($A$):** Discrete brackets ($\{-10\%, -5\%, 0\%, +5\%, +10\%\}$) to enable robust logit-level action masking before the Softmax layer.
   - **Reward ($R$):** Balances margin optimization and conversion probability, penalizing unsold near-expiry inventory.
2. **Off-Policy Evaluation (OPE):**
   - Policy superiority is proven offline using the **Doubly Robust Estimator ($\hat{V}_{DR}$)**.
3. **Inference Latency:** Feature-Fetching via Redis (<5ms), PyTorch Model as ONNX-Graph for C++ execution (<50ms total roundtrip).

## 🚀 Execution & Audit
The `src/rl_guards.py` intercepts the RL agent's `step()` function. In simulations, it successfully blocked thousands of fatal ERP synchronization errors, demonstrating the critical necessity of Safe RL wrappers in financial applications.
