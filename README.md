# 🧠 HealthRisk AI

A simple health risk prediction web app powered by a PyTorch linear model. Users can input personal and lifestyle data to get a prediction and confidence score — plus lifestyle recommendations based on common health guidelines.

## 🚀 Features

- Predicts health risk level (Low, Moderate, High)
- Model confidence score
- Personalized lifestyle recommendations
- Interactive “What-If” Simulator for smoking & drinking
- Clean Bootstrap UI
- About and How It Works pages

## 🛠 How It Works

- User inputs are scaled and encoded
- Habits are converted into a point-based score
- A PyTorch model computes risk based on features
- Habits are weighted ~1.1x more than other factors to reflect their health impact

## 🧪 Tech Stack

- **Flask** – web framework
- **WTForms** – form validation
- **PyTorch** – ML model
- **Pandas & NumPy** – data handling
- **Bootstrap 4** – UI

## 💡 Sample Inputs

- Age
- Gender
- Systolic & Diastolic BP
- Cholesterol level
- Lifestyle habits (e.g., smoking, stress, exercise)

## 📷 Screenshots

*Add screenshots of the home, predict form, and result page here if you want*

## 🧵 Try It Locally

```bash
git clone https://github.com/yourusername/healthrisk-ai.git
cd healthrisk-ai
pip install -r requirements.txt
python run.py
