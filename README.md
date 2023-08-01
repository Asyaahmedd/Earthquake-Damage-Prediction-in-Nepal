# Earthquake Damage Prediction in Nepal

In 2015, Nepal experienced a devastating earthquake with a magnitude of 7.8, resulting in significant loss of life, thousands of buildings being destroyed, and an estimated damage cost of $10 billion USD. The objective of this project is to develop a predictive model to assess the building damage caused by the earthquake that struck the Gorkha district of Nepal in 2015.

# Data Source
The primary data source for this project is Open Data Nepal, which provides a comprehensive dataset on the 2015 Nepal earthquake and its impact on buildings in the Gorkha district. The dataset includes valuable information on various factors, such as building characteristics, geographical location, structural integrity, and damage severity.

# Notebooks
1. Data Wrangling Using SQLite3: This notebook contains the data wrangling process, where the dataset from Open Data Nepal has been loaded and prepared for analysis. SQLite3 is used as the database management system to facilitate data manipulation.

2. Earthquake_Damage_Prediction_in_Nepal: This notebook presents the machine learning and predictive modelling aspect of the project. It includes data exploration, feature engineering, model training, and evaluation steps to create a reliable earthquake damage prediction model.
3. Dataset Description: a summary of the features stored in the `nepal.sqlite` database.

# Dataset Description
The dataset is stored in an SQLite database and consists of three tables: `building_structure`, `building_damage`, and `id_map`.
These tables provide valuable information related to building characteristics, damage assessment, and geographic mapping for buildings affected by the 2015 earthquake in Nepal's Gorkha district. The dataset is a comprehensive resource for developing a predictive model to assess the severity of building damage caused by earthquakes.
# Project Scope
The primary objective of this project is to build a machine-learning model capable of predicting the level of damage to buildings in the aftermath of an earthquake. The model will take into account various factors such as building age, foundation type, roof type, land surface condition, and other relevant features available in the dataset.

The project's scope includes the following key tasks:

1. Data Exploration and Preprocessing: Thoroughly explore the dataset to gain insights into the distribution of features, identify missing values, and handle any data anomalies. Perform data preprocessing to ensure the data is suitable for model training.

2. Feature Engineering: Analyze the existing features and create new relevant features, if necessary, to enhance the model's predictive power.

3. Model Selection and Training: Evaluate different machine learning algorithms suitable for regression tasks and select the most appropriate one based on performance metrics. Train the chosen model using the training data.

4. Model Evaluation: Validate the model's performance using suitable evaluation metrics, such as accuracy.

5. Prediction and Interpretation: Apply the trained model to predict the building damage severity on a separate test dataset. Interpret the model's results to understand the key factors influencing the building damage.
