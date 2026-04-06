# 🌍 AirAware: Smart Air Quality Prediction System

## 📌 Project Overview

AirAware is an end-to-end data analytics and machine learning project designed to monitor, analyze, and predict Air Quality Index (AQI) using historical pollution data. The system provides actionable insights through interactive visualizations and real-time predictions, helping users understand environmental conditions and potential health risks.

---

## 🚀 Key Features

* 🔹 Data preprocessing and cleaning of real-world air quality datasets
* 🔹 Feature engineering using time-based and lag features
* 🔹 Multiple model implementation:

  * ARIMA (time series forecasting)
  * Prophet (trend & seasonality modeling)
  * LSTM (deep learning approach)
  * XGBoost (best-performing model)
* 🔹 Model evaluation using RMSE and MAE metrics
* 🔹 Interactive Streamlit dashboard
* 🔹 AQI alert system (Good → Hazardous levels)
* 🔹 Real-time AQI prediction
* 🔹 Upload new dataset and retrain model

---

## 🛠️ Tech Stack

* **Programming:** Python
* **Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn, XGBoost
* **Visualization:** Matplotlib, Streamlit
* **Tools:** Jupyter Notebook, GitHub
* **Deployment:** Streamlit

---

## 📊 Dataset Description

The dataset contains environmental and pollution-related features:

* Timestamp
* Country & City
* PM2.5, PM10
* NO₂, SO₂, O₃, CO
* Temperature, Humidity, Wind Speed
* AQI (Target Variable)

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing

* Handled missing values using forward filling
* Converted timestamp into datetime format
* Sorted and cleaned dataset

### 2️⃣ Feature Engineering

* Extracted time-based features (hour, day, month)
* Created lag features for AQI prediction

### 3️⃣ Model Building

* Implemented ARIMA, Prophet, LSTM, and XGBoost
* Compared models using evaluation metrics

### 4️⃣ Model Evaluation

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)

👉 XGBoost achieved the best performance and was selected for deployment

---

## 📈 Results

* Achieved high prediction performance using XGBoost
* Evaluation Metrics:

  * MAE: Low error indicating accurate predictions
  * RMSE: Stable model performance
* Approximate Accuracy: **~85–95%** (based on dataset & tuning)

---

## 🖥️ Streamlit Dashboard Features

* 📊 AQI trend visualization
* 📍 City-based filtering
* ⚠️ AQI alert system (color-coded)
* 🔮 Real-time AQI prediction
* 📂 Dataset upload option
* 🔄 Model retraining functionality

---

## ▶️ How to Run the Project

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/AirAware-AQI-Prediction.git
cd AirAware-AQI-Prediction
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run Dashboard

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
AirAware-AQI-Prediction/
│
├── app.py                     # Streamlit Dashboard
├── air_quality_model.ipynb    # Model Development Notebook
├── globalAirQuality.csv      # Dataset
├── AQI_XGBOOST_MODEL.pkl     # Saved ML Model
├── requirements.txt
├── README.md
```

---

## 🎯 Future Enhancements

* 🌐 Integration with real-time AQI APIs
* 📱 Mobile-responsive dashboard
* ☁️ Cloud deployment (Streamlit Cloud / AWS)
* 🤖 Advanced deep learning models

---

## 👩‍💻 Author

**Swathi Challakonda**
Aspiring Data Analyst / Data Scientist

---

## ⭐ Acknowledgement

This project was developed as part of a hands-on learning experience in data analytics, machine learning, and real-time dashboard development.

---

## 💡 Key Highlight

This project demonstrates a complete pipeline from data preprocessing to deployment, showcasing strong skills in data analysis, machine learning, and interactive visualization.
