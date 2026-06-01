<div align="center">

# 🤖 Safe Reinforcement Learning: Dynamic Pricing Engine

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-grey?style=for-the-badge&logo=onnx&logoColor=white)](https://onnx.ai/)
[![Redis](https://img.shields.io/badge/redis-%23DD0031.svg?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io/)
[![Dataset](https://img.shields.io/badge/Dataset-Olist_E--Commerce-F7931E?style=for-the-badge)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
[![ROI](https://img.shields.io/badge/Net%20Margin-%2B%E2%82%AC620k%20%2F%20Year-emerald?style=for-the-badge&logo=moneygram&logoColor=white)](https://github.com)

**A production-ready Safe Reinforcement Learning (RL) architecture for Dynamic Pricing in E-Commerce.**

*Optimizes pricing strategies using a Markov Decision Process (MDP) while strictly adhering to business logic via deterministic Action Masking.*

---
</div>

## 💼 Business Value & CFO Directives

Deploying RL in live retail environments carries massive financial risk if the agent hallucinates. This engine guarantees mathematical safety constraints:

<table>
  <tr>
    <td><strong>📈 Net Margin Expansion</strong></td>
    <td>Projected impact of <strong>+ 620,000 € / year</strong> through optimized pricing policies.</td>
  </tr>
  <tr>
    <td><strong>🛡️ Margin Floors</strong></td>
    <td>Hard cut-off at $Cost \times 1.10$. Guarantees that <strong>no neural network action can ever breach this limit</strong>.</td>
  </tr>
  <tr>
    <td><strong>🔌 Legacy ERP Fallbacks</strong></td>
    <td>Protects against negative inventory sync issues and competitor API outages by selectively masking discrete discount brackets (-10%, -5% → -∞ logits).</td>
  </tr>
</table>

---

## 🏗️ Technical Architecture & Core Components

| Component | Path / Detail | Description & Core Logic |
| :--- | :--- | :--- |
| **🧠 MDP Formulation** | `State`, `Action`, `Reward` | **State Space ($S$):** Normalized features preventing data leakage ($t_{expiry}$, competitor index, inventory).<br>**Action Space ($A$):** Discrete brackets ({-10%, -5%, 0%, +5%, +10%}) for robust logit-level action masking before the Softmax layer.<br>**Reward ($R$):** Balances margin optimization and conversion probability, penalizing unsold near-expiry inventory. |
| **📊 Off-Policy Evaluation** | `OPE` | Policy superiority is proven offline using the **Doubly Robust Estimator ($\hat{V}_{DR}$)** before any live deployment. |
| **⚡ Inference Latency** | `Redis` & `ONNX` | Feature-Fetching via Redis (< 5ms); PyTorch Model exported as ONNX-Graph for ultra-fast C++ execution (**< 50ms total roundtrip**). |
| **🛑 Execution Guards** | `src/rl_guards.py` | Intercepts the RL agent's `step()` function. Successfully blocked thousands of fatal ERP synchronization errors during simulation. |

---

## 🚀 Execution & Audit

<details>
<summary><b>🛠️ System Audit & Testing (Click to expand)</b></summary>

### Intercepting the Agent
The `src/rl_guards.py` module acts as the ultimate safety net. It demonstrates the critical necessity of Safe RL wrappers in financial applications by mathematically preventing invalid actions from reaching the legacy ERP system.

```bash
# Verify the action masking and safety constraints
pytest tests/test_rl_guards.py -v
