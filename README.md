# Clustering & Prediction for Customer Segmentation in Online Sales

## 📖 **Introduction**  
This repository contains a project aimed at customer segmentation for an online sales platform using both unsupervised (clustering) and supervised (prediction) learning techniques. The project utilizes Python and various machine learning models to analyze customer behavior and predict future sales trends.

## 📊 **Project Overview**

### 🏷️ **Clustering**  
- Applied K-Means clustering to segment customers based on their purchasing behavior.  
- Used the Elbow Method and Silhouette Score to determine the optimal number of clusters.  
- **Dataset**: `customer_data.csv`

### 🔍 **Prediction**  
- Built classification models (Logistic Regression, Random Forest) to predict customer categories or purchasing behavior.  
- Evaluated models using accuracy, precision, recall, and F1-score.  
- **Dataset**: `sales_data.csv`

## 🛠️ **Tech Stack**  
- **Programming Language**: Python 🐍  
- **Libraries Used**: Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, etc
- **Development Environment**: Jupyter Notebooks (Google Colab)

## 🚀 **How to Run the Project**
1. **Clone this repository**:
    ```bash
    git clone https://github.com/yourusername/clustering-prediction-onlinesales.git
    ```

2. **Open Jupyter Notebook** and run the `.ipynb` files:
    - `Clustering_Model.ipynb` for clustering analysis
    - `Prediction_Model.ipynb` for prediction analysis

## 📜 **Results & Insights**  
Here's a more concise version of the **Results & Insights** section:

---

📜 **Results & Insights**  
**Clustering**:  
- Used **HDBScan** for clustering, enhanced with manual Grid Search to optimize parameters. The clustering effectively segmented customers into distinct groups based on their behaviors.

**Prediction**:  
- **Logistic Regression** and **Random Forest** performed similarly, achieving **93.19%** accuracy.  
   - **Logistic Regression** is fast and interpretable, while **Random Forest** handles complex, non-linear patterns better.  
   - Other models like **KNN** (93.12%) and **Gradient Boosting** (93.05%) showed similar results but with slower performance or slightly lower accuracy.  
   - **Decision Tree** had the lowest accuracy (**86.08%) due to overfitting.  

**Tuning Results**:  
- After tuning, both models reached **93.43%** accuracy.  
   - **Logistic Regression**: Best parameters: `{'C': 0.01, 'solver': 'lbfgs'}`  
   - **Random Forest**: Best parameters: `{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}`  

## 🎯 **Conclusion**  
This project enhances the understanding of customer segmentation and prediction, making it possible to target marketing strategies more effectively. Future improvements could involve tuning hyperparameters further or exploring more advanced models like Gradient Boosting or Neural Networks.

## 🌐 **Connect with Me**  
If you find this project useful, let's connect! 😊  

- **LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/felixrafael/) 
- **GitHub**: [GitHub Profile](https://github.com/FR21)

📌 Feel free to fork, star, and contribute! 🚀
