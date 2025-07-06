# 🧬 Disease Risk Predictor

A machine learning model that predicts disease risk based on lifestyle habits and personal data (e.g., age, smoking, exercise). Built for learning, experimentation, and real-world use cases.

---

## 🚀 Features

- ✅ Clean tabular dataset (`.csv`)
- ✅ Weighted binary encoding for habits (smoking, drinking, exercise, etc.)
- ✅ CLI-based prediction system with confidence scores
- ✅ Scikit-learn + Pandas + PyTorch
- ✅ Honest, habit-aware classification logic
- ✅ 5 Risk Levels: Very Low, Low, Medium, High, Very High

---

## ⚖️ Habit-Aware Risk Classification

The model now thinks more like a doctor. It considers not just how many good or bad habits you have, but how **intense** they are. Bad habits now weigh more heavily than before.

### 🎯 Why This Update?

Previously, the model sometimes predicted **Low Risk** for users with extremely unhealthy lifestyles just because they exercised occasionally. That wasn’t realistic.

### 🧠 What Changed?

- **Negative habits dominate** if they are extreme (e.g., heavy smoking, drinking)
- **Positive habits help**, but no longer cancel out severe risks
- Prediction system is now **honest**, not blindly optimistic
- Confidence score is now displayed with each prediction

---

## 🔨 Technical Improvements

- Rebalanced `habit_risk_scores` dictionary
- Improved logic in `process_habits()` function
- Expanded dataset with more realistic combinations
- Labels manually refined for increased training accuracy
- Model retrained for improved performance
- Added prediction confidence output for user transparency
- Added 5 classification levels for more precise risk assessment

---

## 📚 Motivation

This project was built to understand how to make ML models reflect **real-world logic**, not just achieve accuracy. By thinking deeply about what matters in health prediction, I learned how to:

- Design fair scoring systems
- Handle messy lifestyle data
- Make transparent, honest models for real users

---

## ✅ Future Goals

- Add more health indicators (BMI, sleep, heart rate, etc.)
- Explore deep learning-based classification
- Build a web-based version for public access
- Polishing and ready for deployment

---

Stay tuned for more improvements — this is just the beginning of smarter, more ethical AI in health.
