# 🧬 Disease Risk Predictor

A machine learning model that predicts disease risk based on lifestyle habits and personal data (e.g. age, smoking, exercise). Built for learning, experimentation, and real-world use cases.

---

## 🚀 Features

- ✅ Clean tabular dataset (CSV)
- ✅ Binary encoding for habits (smoking, drinking, etc.)
- ✅ CLI-based prediction system
- ✅ Scikit-learn + Pandas + PyTorch

## ⚙️ Model Balancing Update: Habit-Aware Risk Classification

We're currently improving the health risk prediction model by rebalancing how it interprets various lifestyle habits.

### 🧠 Goal
To ensure the model is more **realistic and fair** by:
- Penalizing harmful habits (e.g., heavy smoking, heavy drinking) **more harshly**
- Giving **moderate credit** to positive habits (e.g., exercise, walking, yoga)
- Preventing dangerous behaviors from being **canceled out** by a few good habits

### 🧪 Why?
Previously, the model sometimes predicted **Low Risk** for users with severely unhealthy habits just because they also exercised occasionally. That didn’t reflect reality. We're now adjusting the influence of each habit so that:
- **Negative habits dominate** if extreme
- **Positive habits help**, but don't create unrealistic optimism

### 🔧 What's Changing
- Updated `habit_risk_scores` dictionary to rebalance weights
- Improved preprocessing logic in `process_habits()`
- Refined dataset with more extreme and realistic combinations
- Adjusted labels in training data to better reflect true health risks

### 🚧 In Progress
- Further dataset cleanup and label validation
- Training with more robust, balanced samples
- Monitoring validation accuracy and overfitting during tuning

---

Stay tuned for a healthier, more honest AI.
