# CoDeR-RL: Reward-Shaped Reinforcement Learning for Efficient Reasoning in LLMs
Lenovo — Summer 2025  
**Base Model:** Qwen2.5-3B-Instruct  
---

## 🧩 Overview
**CoDeR-RL** (Cosine + Delta + Regularization Reinforcement Learning) is a reward-shaping framework that improves reasoning quality and stability in small-to-mid-sized large language models.  
It builds on NVIDIA’s Tool-N1 baseline but replaces the brittle **binary reward** with a structured, multi-component design that teaches *how* to reason better—not just *whether* the model is right.

---

## ⚙️ Reward Design

### 1️⃣ Cosine Reward
Smoothly scales reward by **answer correctness** and **reasoning length**.  
- Higher reward for short, correct answers.  
- Soft penalty for long or incorrect outputs.  
- Prevents runaway reasoning chains.

### 2️⃣ Delta Reward
Rewards only **improvement between steps**, not cumulative reward.  
Formula: `r_k = α × (r_k − r_{k+1})`  
- Promotes genuine reasoning progress.  
- Eliminates “reward farming” from repetition.

### 3️⃣ Clip Regularization
Caps per-step rewards at a threshold η.  
Formula: `r(q,p_k) = min(r_process(q,p_k) − η, 0)`  
- Stops reward inflation from verbose steps.  
- Stabilizes gradients and limits over-optimization.

🧠 **Combined Effect**  
Cosine governs *scaling*, Delta enforces *progress-based learning*, and Clip maintains *stability*.  
Together they yield cleaner reasoning, shorter chains of thought, and smoother reward curves.

---

### 🧭 Reward Interaction Diagram
```text
                 ┌──────────────────────────────────────────────┐
                 │                 Reward Signal                │
                 └──────────────────────────────────────────────┘
                                     │
                                     ▼
                     ┌─────────────────────────────────┐
                     │   Step-Level Reward Components   │
                     └─────────────────────────────────┘
          ┌──────────────────┬──────────────────┬──────────────────┐
          │                  │                  │                  │
          ▼                  ▼                  ▼                  ▼
   [ Cosine Reward ]   [ Delta Reward ]   [ Clip Regularizer ]   (others…)
  scales by length &   rewards only step   caps per-step reward η
  correctness → concise   improvements →     → prevents loops &
  reasoning                genuine progress    verbosity
          │                  │                  │
          └──────────┬───────┴──────────┬───────┘
                     ▼                  ▼
            combine signals → stabilize → scale → bound
                     │
                     ▼
           ┌────────────────────────────┐
           │   Total Shaped Reward r̂    │
           │   (used by GRPO / PPO loop)│
           └────────────────────────────┘
                     │
                     ▼
       improved reasoning efficiency:
         • shorter CoTs
         • higher accuracy
         • smoother training
