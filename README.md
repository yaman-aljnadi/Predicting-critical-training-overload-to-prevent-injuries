# Predicting Running-Related Injuries with Machine Learning

This project aims to predict running-related injuries using machine learning techniques based on data from a large-scale athletic study. The primary dataset is sourced from the following publication:

**Dataset Source**:  
[Running Injury Dataset - DataverseNL](https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/UWU9PV)

## 📊 Dataset Overview

The dataset contains longitudinal data collected from runners and is available in two formats:
- **Daily-level data** (primary focus)
- **Weekly-level data**

The daily dataset provides more granular insights into training loads and injury risk, and thus serves as the primary focus of this study.

---

## 🧠 Models Developed

We evaluated four models on the daily dataset to predict whether a runner would sustain an injury. Below are the models and their performance metrics.

### 1. Random Forest

**Classification Report:**
         precision    recall  f1-score   support

       0       0.98      0.87      0.93     11142
       1       0.01      0.10      0.02       182

accuracy                           0.86     11324
