# 📊 Internship Task 2 – Streamlit Clustering App

## 📌 Project Overview
This project is a Streamlit-based web application developed as part of an internship task.  
The application uses a dataset whose file path is directly specified inside the Python code.  
It applies the **K-Means clustering algorithm** to segment customers and visually display  
different customer groups based on their purchasing behavior.

---

## 🛠️ Technologies Used
- Python  
- Streamlit  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  

---

## 📂 Project Structure
task_2/
│
├── app.py
├── README.md
└── dataset.csv

---

## 📥 Dataset Information
- The dataset path is **hardcoded inside the `app.py` file**
- No dataset upload option is provided in the UI
- Dataset loads automatically when the application starts
- Dataset contains numerical customer-related data
- Used for customer segmentation using K-Means clustering

Example dataset path used in code:
C:/Users/kumar/OneDrive/Desktop/streamlit/internship/task_2/dataset.csv

---

## 🚀 Features
- Automatic dataset loading from local file path  
- K-Means clustering implementation  
- User-defined number of clusters (K)  
- Cluster labels added to the dataset  
- Visual representation of clusters  
- Simple and interactive Streamlit interface  

---

## ⚙️ Installation Steps

### Step 1: Install Python
Download Python from:  
https://www.python.org/

---

### Step 2: Install Required Libraries
pip install streamlit pandas numpy scikit-learn matplotlib

---

### Step 3: Run the Application
Navigate to the project folder and run:
streamlit run app.py

---

## ❌ Common Error and Solution

Error:  
ValueError: Input contains NaN values  

Solution:  
Ensure missing values are handled before applying K-Means clustering.

---

## 🧾 Conclusion
This project demonstrates a basic **unsupervised machine learning** system using Streamlit,  
where customer data is grouped into meaningful clusters using the **K-Means algorithm**.  
It is suitable for internship and academic project submissions.

---

## 👤 Author
Name: DURAIMURUGAN   
Project Type: Internship Task – Machine Learning  
Framework: Streamlit  
