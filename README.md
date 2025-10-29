# CoDeR-RL: Reward-Shaped Reinforcement Learning for Efficient Reasoning in LLMs
 
Lenovo — Summer 2025  
**Base Model:** Qwen2.5-3B-Instruct  
---

## 🧩 Overview
**CoDeR-RL** (Cosine + Delta + Regularization Reinforcement Learning) is a reward-shaping framework that improves reasoning quality and stability in small-to-mid-sized large language models.  
It builds on NVIDIA’s Tool-N1 baseline but replaces the binary reward with a structured, multi-component design that teaches *how* to reason better—not just *whether* the model is right.

---

## ⚙️ Reward Design

### **1️⃣ Cosine Reward**
Smoothly scales reward by **answer correctness** and **reasoning length**.  
- Higher reward for short, correct answers.  
- Soft penalty for long or incorrect outputs.  
- Prevents runaway reasoning chains.

### **2️⃣ Delta Reward**
Rewards only **improvement between steps**, not cumulative reward.  
\[
r_k = \alpha (r_{k} - r_{k+1})
\]
- Promotes genuine reasoning progress.  
- Eliminates “reward farming” from repetition.

### **3️⃣ Clip Regularization**
Caps per-step rewards at threshold η:  
\[
r(q,p_k) = \min(r_{\text{process}}(q,p_k) - \eta, 0)
\]
- Stops reward inflation from verbose steps.  
- Stabilizes gradients and limits over-optimization.

🧠 **Combined Effect:**  
Cosine governs *scaling*, Delta enforces *progress-based learning*, and Clip maintains *stability*.  
Together they yield cleaner reasoning, shorter CoTs, and smoother reward curves.

---

### 🧭 Reward Interaction Diagram
                  ┌──────────────────────────────────────────────┐
                  │               Reward Signal                   │
                  └──────────────────────────────────────────────┘
                                     │
                                     ▼
                     ┌─────────────────────────────────┐
                     │   Step-level Reward Components   │
                     └─────────────────────────────────┘
         ┌───────────────────┬────────────────────┬───────────────────┐
         │                   │                    │                   │
         ▼                   ▼                    ▼                   ▼
  [ Cosine Reward ]   [ Delta Reward ]       [ Clip Regularizer ]   (others…)
 smooth scaling by   reward = diff(prev,    cap per-step reward
 length + accuracy     next) → promotes     η to prevent loops
  → concise CoT         genuine progress       and verbosity
         │                   │                    │
         └──────────┬────────┴────────────┬────────┘
                    ▼                     ▼
           combine signals → stabilize → scale → bound
                    │
                    ▼
          ┌────────────────────────────┐
          │   Total Shaped Reward r̂    │
          │   (used by GRPO/PPO loop)   │
          └────────────────────────────┘
                    │
                    ▼
         improved reasoning efficiency
            + shorter CoTs
            + higher accuracy
            + smoother training

---

## 🧪 Results

| Setup | BFCL Accuracy | Notes |
|:------|:--------------:|:------|
| SFT only | 45.53 % | baseline |
| RL (binary reward) | 44.52 % | degraded |
| RL + cosine shaping | ↑ | smoother reasoning |
| RL + delta + clip | ↑ | stable updates |
| **CoDeR-RL (cosine + delta + clip)** | **86.63 %** | best 3B-scale result |

- **+3 points** accuracy gain over all other 3B models.  
- **40 % shorter** reasoning traces.  
- Matched Qwen2.5-7B performance with **half the compute**.

---

## 💡 Insight
Binary rewards only say *if* the model is correct.  
CoDeR-RL teaches *how* to reason: concise, stable, and efficient.  
Reward shaping turned RL from brute-force search into guided learning.

---

## 🔮 Next Steps
- Add non-tool-calling data (e.g., [When2Call](https://github.com/NVIDIA/When2Call)) for better irrelevance detection.  
- Extend to **risk-aware** and **uncertainty-penalized** objectives.  
- Evaluate on **multi-turn reasoning** and **function-calling benchmarks**.

---

> *CoDeR-RL: Small models that reason smartly, not just loudly.*
