# Water-potability-prediction

This project uses different machine learning algorithms to perform the water potability prediction. 
Factors like pH value, conductivity, hardness were used to achieve the above and determine whether it is safe to drink or not. 

Dataset : https://www.kaggle.com/datasets/adityakadiwal/water-potability

## Deployment

StreamLit - Open source framework that helps in creating web apps for data science and machine learning

Model - The ipynb file used for the build, training, and testing of the model

## Documentation

IMPORTING THE DATASET 

Dataset, along with the parameters was taken from Kaggle and imported into the python Jupiter notebook.

## Data preprocessing

1. Handling Missing Data

Considering the missing data values in the dataset, the median imputation has been applied to handle them. 

2.Outlier Analysis

Outliers were identified and were capped to their respective maximum and minimum bounds with the help of IQR implementation.

3.Exploratory Data Analysis

The data has been thoroughly analysed and multiple graphs have been visualized in association with the parameters considered. 

## Model building and prediction

The machine learning model built in this project uses a Decision Tree classifier, Random Forest Classifier and XGB Classifier.

To find the enhanced performance resulting model parameters, hyper-parametric tuning was performed in the approach.



