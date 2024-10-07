# Iowa Corn Yield Prediction

## Goal:
Predict corn yield in bushels per acre for each county in the state of Iowa.

## Data Sources:
- Corn yield: [USDA QuickStats](https://quickstats.nass.usda.gov/)
- Meterological and climate data: [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/request/coop/fe.phtml)

## Workflow and Code Files Explanation:
For each step of the pipeline, I write the classes and functions in a .py file, and then created a short .ipynb file to create instances of classes or call functions to perform the step. The general workflow is as follows:
1. Load data from multiple sources using API and perform some initial manipulations such as matching weather station to county and finding nearest station for counties without a weather station: \
    [Data Loading Classes and Functions](https://github.com/selinawaang/Iowa-Corn-Yield-Prediction/blob/main/code/Load_Datasets.py)  
    [Data Loading Notebook](https://github.com/selinawaang/Iowa-Corn-Yield-Prediction/blob/main/code/Load_Datasets.py)  
2. Clean datasets and merge multiple datasets together: \
[Data Cleaning and Merging Classes and Functions](https://github.com/selinawaang/Iowa-Corn-Yield-Prediction/blob/main/code/Clean_and_Merge.py)  
     [Data Cleaning and Merging Notebook](https://github.com/selinawaang/Iowa-Corn-Yield-Prediction/blob/main/code/Clean_and_Merge.ipynb)
3. Data preprocessing: scaling data, feature engineering and feature selection \
[Data Cleaning Classes and Functions](https://github.com/selinawaang/Iowa-Corn-Yield-Prediction/blob/main/code/Preprocess_Data.py)
4. Model training and evaluation: pipeline includes data preprocessing, model training and hyperparameter tuning, results evaluation on test set \
[Model Training and Evaluation Pipeline Classes and Functions](https://github.com/selinawaang/Iowa-Corn-Yield-Prediction/blob/main/code/Train_Models.py)  
[Model Training and Evaluation Notebook](https://github.com/selinawaang/Iowa-Corn-Yield-Prediction/blob/main/code/Data%20Preprocessing%20and%20Training.ipynb) (In Progress)


## Project Stages:

### Stage 1: Complete
- Build an end to end modular pipeline to:
    - retrieve data using API
    - clean and preprocess data
    - engineer features features
    - train and evaluate models: RandomForest, Linear Regression, Lasso Regression, XGBoost
    - predict on test data
- Implement Bayesian optimization to tune model parameters
- Create visualizations using Tableau and GeoPandas


### Stage 2: In Progress
- Improve model performance
- Expand dataset to include more soil data
- Implement Emsemble model (Complete)

### Stage 3: In Progress
- Use sattelite data to build a deep learning network and compare performance with traditional ML approach

[Check here for my detailed project notes!](https://longing-acrylic-b82.notion.site/Corn-Yield-Prediction-Project-Notes-a69afaaa5af74f448f4201e5804c6cfe)

### 