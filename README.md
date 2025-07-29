![image alt](https://github.com/Daijek/UK_Accident_Predictive_Analysis_2020/blob/main/images/downloaded-image.png?raw=true)

# 🚦 Predictive Analysis of UK Road Traffic Accidents

## 🔍 Comprehensive Data Pipeline for Accident Severity Prediction
This project implements a complete data science pipeline for analyzing UK road accident data, extracting insights, and building predictive models for accident severity. The system processes raw SQLite data through cleaning, feature engineering, association mining, clustering, and machine learning to predict fatal injuries with **over 80% accuracy**.

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

### 🧩Key Features
- **Core Technologies**;
```
graph LR
A[SQLite] --> B[Pandas]
B --> C[Scikit-learn]
C --> D[MLxtend]
D --> E[Folium]
E --> F[Matplotlib]
```
- **Custom Imputation Classes**:
  - KDE-based continuous variable handling
  - Probability-based categorical imputation
  - Location-aware missing value filling
- **Association Rule Mining**:
  - Apriori algorithm implementation example
    ```
    rules = association_rules(freq_item_sets, metric="lift", min_threshold=0.5)
    ```
    
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
![Temporal Patterns](https://github.com/Daijek/UK_Accident_Predictive_Analysis_2020/blob/main/images/peak_hours.png?raw=true)
- Motorbike accidents peak on weekends (2-5pm)
- Pedestrian incidents cluster in evening hours (6-9pm)
- Friday has highest accident volume overall

### 🗺️ Regional Hotspots
Example 
```
humberside.plot_clusters_on_map(5, humberside_kmeans_cluster[0])
```
![Regional Hotspots](https://github.com/Daijek/UK_Accident_Predictive_Analysis_2020/blob/main/images/humberside%20accident%20clusters.png?raw=true)

### 📈 Key Predictors of Severity
1. Speed limit (30mph zones)
2. Urban/rural classification
3. Junction control type
4. Number of casualties
5. Road surface conditions

## 🛠️ Setup & Execution
### Prerequisites
```
pip install pandas numpy scikit-learn mlxtend folium matplotlib seaborn
```
### Execution workflow
```
sequenceDiagram
    participant S as SQLite DB
    participant P as Python
    participant M as Models
    
    S->>P: Extract via Extract_DB_info
    P->>P: Clean/transform data
    P->>P: Temporal analysis
    P->>P: Association mining
    P->>P: Regional clustering
    P->>P: Outlier detection
    P->>M: Train classifiers
    M-->>P: Evaluation metrics
```

### Running Analysis
```
# Instantiate accident analysis pipeline
accident_model = model_classification(x_balanced, y_balanced)

# Generate classification reports
accident_model.get_classification_report()

# Visualize model performance
accident_model.visualize_model_results()
```

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
  e.g.
  ```
  app.post("/predict-severity", input_schema=AccidentFeatures)
  ```
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
