# 🏡 HousePrice.AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://soumyasatapathy1819-houseprice-ai-main-lls5vj.streamlit.app/)
An interactive, end-to-end Machine Learning web dashboard designed to estimate residential property valuations based on structural layouts, structural condition, and location metrics. 

This full-stack application brings a data science training pipeline out of a **Google Colab** notebook and turns it into a live, production-ready system.

---

## 🚀 Features

- **Machine Learning Core**: Uses a high-performing `RandomForestRegressor` ensemble model to predict prices based on historical real estate layouts.
- **Robust REST API Backend**: Built with **Flask** to securely ingest JSON data from user forms, handle automatic fallback imputation for data safety, align operational features, and return real-time low-latency predictions.
- **Interactive UI Dashboard**: Developed using **Streamlit**, boasting custom dark-mode modern design components, input sliders, split toggles, data preview tables, and real-time visualization comparison charts.
- **Resilient Pipeline**: Preprocessor pipelines (`StandardScaler`, `OneHotEncoder`, and imputers) are serialized alongside the model weights to guarantee that validation and deployment preprocessing exactly match training preprocessing.

---

## 🛠️ Architecture Flow

```text
  [ User Interface ]  ---> Sends user specs (JSON) --->   [ Flask API Backend ]
  (Streamlit Frontend) <--- Returns Estimated Price <---   (Loads .pkl + Evaluates)
