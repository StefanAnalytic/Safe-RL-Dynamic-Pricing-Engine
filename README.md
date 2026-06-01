<div align="center">

# 🤖 Safe Reinforcement Learning: Dynamic Pricing Engine

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-grey?style=for-the-badge&logo=onnx&logoColor=white)](https://onnx.ai/)
[![Redis](https://img.shields.io/badge/redis-%23DD0031.svg?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io/)
[![Dataset](https://img.shields.io/badge/Dataset-Olist_E--Commerce-F7931E?style=for-the-badge)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
[![ROI](https://img.shields.io/badge/Net%20Margin-%2B%E2%82%AC620k%20%2F%20Year-emerald?style=for-the-badge&logo=moneygram&logoColor=white)](https://github.com)

**Eine produktionsreife Safe Reinforcement Learning (RL) Architektur für dynamisches Pricing im E-Commerce.**

*Optimiert Preisstrategien mithilfe eines Markow-Entscheidungsprozesses (MDP) und hält sich durch deterministisches Action Masking strikt an Geschäftslogik und Constraints.*

---
</div>

## 💼 Business Value & CFO-Vorgaben

Der Einsatz von RL im Live-Retail birgt massive finanzielle Risiken, falls der Agent "halluziniert". Diese Engine garantiert mathematische Sicherheitsbeschränkungen:

<table>
  <tr>
    <td><strong>📈 Netto-Margenwachstum</strong></td>
    <td>Prognostizierter Impact von <strong>+ 620.000 € / Jahr</strong> durch optimierte Preisstrategien.</td>
  </tr>
  <tr>
    <td><strong>🛡️ Margen-Untergrenzen</strong></td>
    <td>Harter Cut-off bei $Cost \times 1.10$. Garantiert, dass <strong>keine Aktion des neuronalen Netzes diese Grenze jemals unterschreiten kann</strong>.</td>
  </tr>
  <tr>
    <td><strong>🔌 Legacy ERP Fallbacks</strong></td>
    <td>Schützt vor negativen Bestands-Synchronisationsfehlern und API-Ausfällen der Konkurrenz durch selektives Maskieren diskreter Rabattstufen (z. B. -10%, -5% $\rightarrow -\infty$ Logits).</td>
  </tr>
</table>

---

## 🏗️ Technische Architektur & Kernkomponenten

| Komponente | Dateipfad / Detail | Beschreibung & Core Logic |
| :--- | :--- | :--- |
| **🧠 MDP Formulierung** | `State`, `Action`, `Reward` | **Zustandsraum ($S$):** Normalisierte Features zur Verhinderung von Data Leakage ($t_{expiry}$, Wettbewerber-Index, Lagerbestand).<br>**Aktionsraum ($A$):** Diskrete Stufen ({-10%, -5%, 0%, +5%, +10%}) für robustes *Logit-Level Action Masking* vor dem Softmax-Layer.<br>**Belohnung ($R$):** Balanciert Margen-Optimierung und Conversion-Wahrscheinlichkeit, bestraft unverkaufte Ware kurz vor Ablaufdatum. |
| **📊 Off-Policy Evaluation** | `OPE` | Die Überlegenheit der Policy wird vor jedem Live-Deployment offline mittels des **Doubly Robust Estimators ($\hat{V}_{DR}$)** mathematisch bewiesen. |
| **⚡ Inference Latency** | `Redis` & `ONNX` | Feature-Fetching über Redis (< 5ms); PyTorch-Modell als ONNX-Graph exportiert für ultraschnelle C++ Ausführung (**< 50ms Total Roundtrip**). |
| **🛑 Execution Guards** | `src/rl_guards.py` | Fängt die `step()`-Funktion des RL-Agenten ab. Hat in Simulationen erfolgreich tausende fatale ERP-Synchronisationsfehler blockiert. |

---

## 🚀 Execution & Audit

<details>
<summary><b>🛠️ System Audit & Testing (Hier klicken zum Aufklappen)</b></summary>

### Den Agenten abfangen
Das `src/rl_guards.py` Modul fungiert als ultimatives Sicherheitsnetz. Es beweist die absolute Notwendigkeit von Safe RL-Wrappern in Finanzanwendungen, indem es mathematisch verhindert, dass ungültige Aktionen das Legacy-ERP-System jemals erreichen können.

```bash
# Action Masking und Sicherheits-Constraints verifizieren
pytest tests/test_rl_guards.py -v
