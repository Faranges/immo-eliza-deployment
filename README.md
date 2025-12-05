# Immo Eliza - Deployment
## 📋 Project Overview
Immo Eliza is a real estate company that has developed a machine learning model to predict property prices. This project focuses on deploying the trained model through a REST API and creating a user-friendly web interface for different stakeholders to get a property price prediction by filling in the form.

## 🎯 Learning Objectives

1. Create a small web application using Streamlit that will allow non-technical people to use the API.

    The Streamlit application will send requests to the API and display the results in a visual interface.

2. Deploy your application on Streamlit Community Cloud
 
## 🏗️ Architecture
`````
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │      │                 │
│  Streamlit Web  │─────▶│   FastAPI API   │─────▶│   ML Model &    │
│  Application    │      │   (Backend)     │      │   Artifacts     │
│  (Frontend)     │◀─────│                 │      │                 │
│                 │      │                 │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────┘
       ▲                         ▲                         ▲
       │                         │                         │
│  Streamlit     │      │      Render       │      │   GitHub       │
│  Community     │      │     (Deployment)  │      │  Repository    │
│     Cloud      │      │                   │      │                │
└────────────────┘      └───────────────────┘      └────────────────┘
`````

## 📁 Project Structure
````
immo-eliza-deployment/
│
├── api/                          # Backend API
│   ├── app.py                   # FastAPI application
│   ├── predict.py               # Prediction logic and model loading
│   ├── Dockerfile               # Docker configuration for API
│   
├── streamlit/                   # Frontend web application
│   ├── app.py                   # Streamlit application
│      
│
├── models/             # Trained model and preprocessing artifacts
│   ├── xgb_pipeline.pkl               # Serialized ML model
│   ├── preprocessor.pkl        # Preprocessing pipeline
│   

├── .gitignore
├── README.md                    
└── requirements.txt
````

## 🚀 Backend API (FastAPI)
(In progress)
## 🌐 Frontend Web Application (Streamlit)
### Features

* User-friendly interface for non-technical users
* Interactive predicting form with 14 property features
* Clear display of predicted pric and price range estimation
* Input validation and error handling

### Running locally
1. Navigate to the Streamlit directory:
   ````` 
   cd streamlit 
2. Install dependencies:
   ````
   pip install -r requirements.txt
3. Run the application:
    ````
    streamlit run app.py
    ````

## 🐳 Docker Configuration

### API Dockerfile
(In progress)
## 📊 Data Schema
### Input Features

The model accepts 14 property features including:

* Location: province, zip code (to be added)
* Property details: type, living area size, state of the building
* Amenities: bedrooms, equiped kitchen, terrace, garden, etc.

### Output

* Predicted price in EUR
* Estimated price range
* Confidence score (to be added)

## 📄 Personal context note

This project was created for educational purposes as part of the BeCode Data Science & AI course (class of 2026).


