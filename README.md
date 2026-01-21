# Intelligent-ATM-Decision-Support-System
End-to-end Business Intelligence and predictive analytics project for ATM management, combining a constellation-schema data warehouse, KNIME-based ETL, Power BI dashboards, machine/deep learning forecasting models, and a web-based decision-support platform.# PFE – Intelligent ATM Decision Support System

##  Project Overview
This project is an end-to-end **Business Intelligence and predictive analytics solution** developed as a **Final Year Project (PFE)**.  
It aims to support decision-making related to **ATM operations**, including availability monitoring, performance analysis, and cash demand forecasting.

The solution integrates **data warehousing, ETL processes, interactive dashboards, machine learning and deep learning models**, as well as a **web-based decision platform**.

---

##  Objectives
- Centralize ATM operational data in a structured data warehouse
- Ensure data quality and consistency through automated ETL processes
- Provide interactive dashboards for decision-makers
- Predict ATM cash demand using advanced ML and DL models
- Deploy dashboards and predictions through a web-based platform

---

##  Global Architecture
The project follows a complete BI workflow:

1. **ETL Process**  
   Data extraction, transformation, and loading using KNIME

2. **Data Warehouse**  
   Constellation schema integrating multiple analysis dimensions related to ATM activities

3. **Analytics & Prediction**  
   Machine Learning and Deep Learning models for cash demand forecasting

4. **Visualization & Decision Support**  
   Power BI dashboards and a web-based decision platform

---

## Data Warehouse Design
- Modeling approach: **Constellation Schema**
- Multiple fact tables related to ATM operations
- Shared dimensions 
- Optimized for analytical and decision-support queries

---

##  ETL Process
- Tool used: **KNIME**
- Main steps:
  - Data extraction from heterogeneous sources
  - Data cleaning and transformation
  - Loading into the data warehouse
- Focus on data quality, consistency, and automation

---

##   Power BI Dashboard
- Interactive decision-support dashboard
- Organized into **targeted analytical themes**
- Key features:
  - ATM availability monitoring
  - Performance indicators
  - Historical analysis
  - Support for operational and strategic decisions

---

##   Machine & Deep Learning Models
ATM cash demand forecasting was implemented using the following models:

- **Linear Regression**
- **Random Forest**
- **SARIMA**
- **LSTM (Long Short-Term Memory)**

All models were developed using **Python and Jupyter Notebooks**, with comparative performance analysis.

---

##  Web-Based Decision Platform
A web application was developed to:
- Deploy interactive dashboards
- Display real-time indicators
- Integrate predictive results from ML/DL models
- Provide a centralized decision-support interface

---
##  Technologies Used
- **ETL**: KNIME  
- **Data Warehouse**: Postgresql
- **Visualization**: Power BI  
- **Programming & Analytics**: Python (Pandas, Scikit-learn, TensorFlow)  
- **Machine & Deep Learning**: ML models + LSTM  
- **Web Development**: Frontend & Backend technologies (Vue.js + Flask)
- **Environment**: Jupyter Notebook

---

## Project Structure
