# HousePrice.AI

HousePrice.AI is a full-stack machine learning web app that predicts residential property prices from user-provided house attributes. The app combines a Streamlit frontend with a Flask backend and uses a trained regression model packaged as a joblib file.

## Features

- Interactive house price prediction interface
- Real-time predictions through a Flask API
- Visual summaries of predicted values and feature inputs
- Clean, modern Streamlit dashboard experience

## Project Structure

- `main.py` – Streamlit frontend for user input and visualization
- `app.py` – Flask API that loads the trained model and returns predictions
- `house_model_package.pkl` – Serialized machine learning model package

## Tech Stack

- Python
- Streamlit
- Flask
- Flask-CORS
- pandas
- scikit-learn
- joblib
- requests

## Setup Instructions

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the backend API:
   ```bash
   python app.py
   ```

4. In a second terminal, start the frontend:
   ```bash
   streamlit run main.py
   ```

5. Open the local Streamlit URL shown in the terminal and use the dashboard to predict house prices.

## Usage

- Enter house attributes such as area, bedrooms, bathrooms, year built, location, condition, and garage availability.
- Click the prediction button to send the data to the backend.
- Review the predicted value and the summary of your input features.

## Notes

The current version is intended for local development and demo use. The backend must be running on port 5000 for the frontend to communicate with it successfully.
