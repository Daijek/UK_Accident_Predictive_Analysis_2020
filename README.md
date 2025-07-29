# 🚦 Predictive Analysis of UK Road Traffic Accidents

## 🔍 Comprehensive Data Pipeline for Accident Severity Prediction
This project implements a complete data science pipeline for analyzing UK road accident data, extracting insights, and building predictive models for accident severity. The system processes raw SQLite data through cleaning, feature engineering, association mining, clustering, and machine learning to predict fatal injuries with **over 80% accuracy**.

![Data Pipeline](https://via.placeholder.com/800x400?text=Data+Extraction+-%3E+Cleaning+-%3E+Analysis+-%3E+Modeling)

## 🧰 Project Components

### 📂 Data Extraction & Transformation
| Component              | Description                                | Key Features                              |
|------------------------|--------------------------------------------|-------------------------------------------|
| **Database Extraction** | Extract_DB_info class for SQLite interaction | Table/column extraction, DataFrame conversion |
| **Data Cleaning**       | Custom imputation pipelines                | KDE for continuous variables, probabilistic categorical imputation |
| **Feature Engineering** | Temporal feature extraction                | Time binning, day-of-week encoding        |

### 📊 Analysis Modules
| Module               | Technique            | Insights Generated                          |
|----------------------|----------------------|---------------------------------------------|
| Temporal Analysis    | Time distribution plots | Peak accident hours (8-9am, 3-6pm)          |
| Association Mining   | Apriori algorithm    | Speed limit + urban roads → higher severity |
| Regional Clustering  | KMeans/KMedoids      | Accident hotspots in Humberside             |

### 🤖 Machine Learning
| Model             | Accuracy | Key Features                              |
|-------------------|----------|-------------------------------------------|
| Stacked Ensemble  | 81%      | RF + XGBoost + KNN meta-learner           |
| Random Forest     | 80%      | Optimized hyperparameters                 |
| XGBoost           | 80%      | Gradient boosting with early stopping     |

## ⚙️ Technical Implementation

### 🧩 Core Technologies & Key Features
- **Custom Imputation Classes**:
  - KDE-based continuous variable handling
  - Probability-based categorical imputation
  - Location-aware missing value filling
- **Association Rule Mining**:
  - Apriori algorithm implementation
- **Geospatial Clustering**:
  - KMeans/KMedoids with elbow method optimization
  - Interactive Folium maps for cluster visualization
- **Stacked Modeling**:
  - Logistic regression meta-learner combining:
    - Random Forest
    - XGBoost
    - K-Nearest Neighbors

## 📊 Insights Discovered

### ⏰ Temporal Patterns
![Temporal Patterns](https://via.placeholder.com/400x300?text=Peak+Hours+8-9am+and+3-6pm)
- Motorbike accidents peak on weekends (2-5pm)
- Pedestrian incidents cluster in evening hours (6-9pm)
- Friday has highest accident volume overall

### 🗺️ Regional Hotspots
![Regional Hotspots](https://via.placeholder.com/400x300?text=Regional+Hotspot+Visualization)

### 📈 Key Predictors of Severity
1. Speed limit (30mph zones)
2. Urban/rural classification
3. Junction control type
4. Number of casualties
5. Road surface conditions

## 🛠️ Setup & Execution
### Prerequisites
*Details to be added*

### Running Analysis
*Details to be added*

## 🚧 Challenges Overcome
| Challenge                 | Solution                                  |
|---------------------------|-------------------------------------------|
| Missing Location Data     | Custom imputation using road/junction attributes |
| Class Imbalance           | Random undersampling (1:1 fatal/non-fatal ratio) |
| High Cardinality Features | Probability-based imputation              |
| Geospatial Clustering     | Elbow method for optimal cluster selection |
| Model Stacking            | Logistic regression meta-learner          |

## 📈 Future Enhancements
- **Real-time Prediction API**:
  - FastAPI implementation
  - Cloud deployment (AWS/GCP)
- **Weather Data Integration**:
  - API-based historical weather merging
- **Interactive Dashboard**:
  - Streamlit/Power BI visualization
  - Cluster exploration maps
- **Deep Learning Models**:
  - Temporal CNN for accident forecasting
  - GNN for regional risk propagation

## ⚠️ Ethical Considerations
All data comes from the UK Government's publicly available Road Safety Open Dataset. The project adheres to OGL (Open Government Licence) requirements:
- Analysis for research purposes only
- No commercial application of raw data
- All models include ethical bias testing
- Proper attribution to data sources

> *"This work uses data from the UK Department for Transport but the analysis and interpretation are my own."*
