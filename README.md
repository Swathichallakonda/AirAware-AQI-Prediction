# 🌍 AirAware: Smart Air Quality Prediction System

## 📌 Project Overview
AirAware is an end-to-end data analytics and machine learning project designed to monitor, analyze, and predict Air Quality Index (AQI) using historical pollution data. The system provides actionable insights through interactive visualizations and real-time predictions.

---

## 🚀 Key Features
- Data preprocessing and cleaning of air quality datasets  
- Feature engineering using time-based and lag features  
- Multiple models:
  - ARIMA  
  - Prophet  
  - LSTM  
  - XGBoost (best-performing)  
- Model evaluation using MAE and RMSE  
- Interactive Streamlit dashboard  
- AQI alert system (Good → Hazardous levels)  
- Real-time AQI prediction  
- Upload new dataset and retrain model  

---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Matplotlib  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Jupyter Notebook  

---

## 📊 Dataset Description
The dataset includes:
- Timestamp  
- Country, City  
- PM2.5, PM10  
- NO₂, SO₂, O₃, CO  
- Temperature, Humidity, Wind Speed  
- AQI (target variable)  

---

## ⚙️ Project Workflow

### 1. Data Preprocessing
- Converted timestamp to datetime  
- Handled missing values  
- Cleaned and sorted dataset  

### 2. Feature Engineering
- Extracted time features (hour, day, month)  
- Created lag features for AQI  

### 3. Model Building
- Trained ARIMA, Prophet, LSTM, XGBoost  
- Compared performance  

### 4. Model Evaluation
- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)  

👉 XGBoost performed best and was selected  

---

## 📈 Results
- High prediction performance  
- Approx Accuracy: **~85–95%**  
- Stable RMSE and low MAE  

---

## 🖥️ Dashboard Features
- AQI trend visualization  
- City selection  
- AQI alerts (color-based)  
- Real-time prediction  
- Dataset upload  
- Model retraining  

---

## ▶️ How to Run

```bash
git clone https://github.com/your-username/AirAware-AQI-Prediction.git
cd AirAware-AQI-Prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Project Structure
```
AirAware-AQI-Prediction/
│
├── app.py
├── air_quality_model.ipynb
├── globalAirQuality.csv
├── AQI_XGBOOST_MODEL.pkl
├── requirements.txt
├── README.md
```

---

## 🎯 Future Improvements
- Real-time AQI API integration  
- Cloud deployment  
- Mobile-friendly UI  
- Advanced deep learning models  

---

## 👩‍💻 Author
**Swathi Challakonda**  
Aspiring Data Analyst / Data Scientist  

---

## ⭐ Key Highlight
This project demonstrates a complete pipeline from data preprocessing to deployment using machine learning and interactive dashboards.
