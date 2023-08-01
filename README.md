# Earthquake Damage Prediction in Nepal

In 2015, Nepal experienced a devastating earthquake with a magnitude of 7.8, resulting in significant loss of life, thousands of buildings being destroyed, and an estimated damage cost of $10 billion USD. The objective of this project is to develop a predictive model to assess the building damage caused by the earthquake that struck the Gorkha district of Nepal in 2015.

# Data Source
The primary data source for this project is Open Data Nepal, which provides a comprehensive dataset on the 2015 Nepal earthquake and its impact on buildings in the Gorkha district. The dataset includes valuable information on various factors, such as building characteristics, geographical location, structural integrity, and damage severity.

# Notebooks
1. Data Wrangling Using SQLite3: This notebook contains the data wrangling process, where the dataset from Open Data Nepal has been loaded and prepared for analysis. SQLite3 is used as the database management system to facilitate data manipulation.

2. Earthquake_Damage_Prediction_in_Nepal: This notebook presents the machine learning and predictive modelling aspect of the project. It includes data exploration, feature engineering, model training, and evaluation steps to create a reliable earthquake damage prediction model.

# Dataset Description
The dataset is stored in an SQLite database and consists of three tables: `building_structure`, `building_damage`, and `id_map`.

## Table `building_structure`
|Variable                              |Description                                                                                        |Type       |
|:-------------------------------------|:--------------------------------------------------------------------------------------------------|:----------|
|`age_building`                          |Age of the building (in years)                                                                     |Number     |
|`building_id`                           |A unique ID that identifies a unique building from the survey                                      |Text       |
|`condition_post_eq`                     |Actual condition of the building after the earthquake                                              |Categorical|
|`count_floors_post_eq`                  |Number of floors that the building had after the earthquake                                        |Number     |
|`count_floors_pre_eq`                   |Number of floors that the building had before the earthquake                                       |Number     |
|`foundation_type`                       |Type of foundation used in the building                                                            |Categorical|
|`ground_floor_type`                     |Ground floor type                                                                                  |Categorical|
|`height_ft_post_eq`                     |Height of the building after the earthquake (in feet)                                              |Number     |
|`height_ft_pre_eq`                      |Height of the building before the earthquake (in feet)                                             |Number     |
|`land_surface_condition`                |Surface condition of the land in which the building is built                                       |categorical|
|`other_floor_type`                      |Type of construction used in other floors (except ground floor and roof)                           |Categorical|
|`plan_configuration`                    |Building plan configuration                                                                        |Categorical|
|`plinth_area_sq_ft`                     |Plinth area of the building (in square feet)                                                       |Number     |
|`position`                              |Position of the building                                                                           |Categorical|
|`roof_type`                             |Type of roof used in the building. Categories are (1) light bamboo/timber, (2) heavy bamboo timber, and (3) reinforced cement concrete/reinforced brick/reinforced brick concrete                                                                |Categorical|
|`superstructure`                        |Superstructure of the building|Categorical

## Table `building_damage`

|Variable                                              |Description                                                                                                                                                                                      |Type       |
|:-----------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------|
|`area_assesed`                                        |Indicates the nature of the damage assessment in terms of the areas of the building that were assessed                                                                                           |Categorical|
|`building_id`                                         |A unique ID that identifies every individual building in the survey                                                                                                                              |Text       |
|`damage_beam_failure_insignificant`                   |Categorical variable that captures insignificant beam failure related damage to the building in terms of the proportion of overall area that is insignificantly damaged                          |Categorical|
|`damage_beam_failure_moderate`                        |Categorical variable that captures moderate beam failure related damage to the building in terms of the proportion of overall area that is moderately damaged                                    |Categorical|
|`damage_beam_failure_severe`                          |Categorical variable that captures severe beam failure related damage to the building in terms of the proportion of overall area that is severely damaged                                        |Categorical|
|`damage_cladding_glazing_insignificant`               |Categorical variable that captures insignificant cladding/glazing related damage to the building in terms of the proportion of overall area that is insignificantly damaged                      |Categorical|
|`damage_cladding_glazing_moderate`                    |Categorical variable that captures moderate cladding/glazing related damage to the building in terms of the proportion of overall area that is moderately damaged                                |Categorical|
|`damage_cladding_glazing_severe`                      |Categorical variable that captures severe cladding/glazing related damage to the building in terms of the proportion of overall area that is severely damaged                                    |Categorical|
|`damage_column_failure_insignificant`                 |Categorical variable that captures insignificant column failure related damage to the building in terms of the proportion of overall area that is insignificantly damaged                        |Categorical|
|`damage_column_failure_moderate`                      |Categorical variable that captures moderate column failure related damage to the building in terms of the proportion of overall area that is moderately damaged                                  |Categorical|
|`damage_column_failure_severe`                        |Categorical variable that captures severe column failure related damage to the building in terms of the proportion of overall area that is severely damaged                                      |Categorical|
|`damage_corner_separation_insignificant`              |Categorical variable that captures insignificant corner separation damage to the building in terms of the proportion of overall area that is insignificantly damaged                             |Categorical|
|`damage_corner_separation_moderate`                   |Categorical variable that captures moderate corner separation damage to the building in terms of the proportion of overall area that is moderately damaged                                       |Categorical|
|`damage_corner_separation_severe`                     |Categorical variable that captures severe corner separation damage to the building in terms of the proportion of overall area that is severely damaged                                           |Categorical|
|`damage_delamination_failure_insignificant`           |Categorical variable that captures insignificant delamination failure related damage to the building in terms of the proportion of overall area that is insignificantly damaged                  |Categorical|
|`damage_delamination_failure_moderate`                |Categorical variable that captures moderate delamination failure related damage to the building in terms of the proportion of overall area that is moderately damaged                            |Categorical|
|`damage_delamination_failure_severe`                  |Categorical variable that captures severe delamination failure related damage to the building in terms of the proportion of overall area that is severely damaged                                |Categorical|
|`damage_diagonal_cracking_insignificant`              |Categorical variable that captures insignificant diagonal cracking damage to the building in terms of the proportion of overall area that is insignificantly damaged                             |Categorical|
|`damage_diagonal_cracking_moderate`                   |Categorical variable that captures moderate diagonal cracking damage to the building in terms of the proportion of overall area that is moderately damaged                                       |Categorical|
|`damage_diagonal_cracking_severe`                     |Categorical variable that captures severe diagonal cracking damage to the building in terms of the proportion of overall area that is severely damaged                                           |Categorical|
|`damage_foundation_insignificant`                     |Categorical variable that captures insignificant foundational damage to the building in terms of the proportion of overall area that is insignificantly damaged                                  |Categorical|
|`damage_foundation_moderate`                          |Categorical variable that captures moderate foundational damage to the building in terms of the proportion of overall area that is moderately damaged                                            |Categorical|
|`damage_foundation_severe`                            |Categorical variable that captures severe foundational damage to the building in terms of the proportion of overall area that is severely damaged                                                |Categorical|
|`damage_gable_failure_insignificant`                  |Categorical variable that captures insignificant gable failure related damage to the building in terms of the proportion of overall area that is insignificantly damaged                         |Categorical|
|`damage_gable_failure_moderate`                       |Categorical variable that captures moderate gable failure related damage to the building in terms of the proportion of overall area that is moderately damaged                                   |Categorical|
|`damage_gable_failure_severe`                         |Categorical variable that captures severe gable failure related damage to the building in terms of the proportion of overall area that is severely damaged                                       |Categorical|
|`damage_grade`                                        |Damage grade assigned to the building by the surveyor after assessment                                                                                                                           |Categorical|
|`damage_in_plane_failure_insignificant`               |Categorical variable that captures insignificant in plane failure related damage to the building in terms of the proportion of overall area that is insignificantly damaged                      |Categorical|
|`damage_in_plane_failure_moderate`                    |Categorical variable that captures moderate in plane failure related damage to the building in terms of the proportion of overall area that is moderately damaged                                |Categorical|
|`damage_in_plane_failure_severe`                      |Categorical variable that captures severe in plane failure related damage to the building in terms of the proportion of overall area that is severely damaged                                    |Categorical|
|`damage_infill_partition_failure_insignificant`       |Categorical variable that captures insignificant infill/partition failure related damage to the building in terms of the proportion of overall area that is insignificantly damaged              |Categorical|
|`damage_infill_partition_failure_moderate`            |Categorical variable that captures moderate infill/partition failure related damage to the building in terms of the proportion of overall area that is moderately damaged                        |Categorical|
|`damage_infill_partition_failure_severe`              |Categorical variable that captures severe infill/partition failure related damage to the building in terms of the proportion of overall area that is severely damaged                            |Categorical|
|`damage_out_of_plane_failure_insignificant`           |Categorical variable that captures insignificant out of plane failure related damage to the building in terms of the proportion of overall area that is insignificantly damaged                  |Categorical|
|`damage_out_of_plane_failure_moderate`                |Categorical variable that captures moderate out of plane failure related damage to the building in terms of the proportion of overall area that is moderately damaged                            |Categorical|
|`damage_out_of_plane_failure_severe`                  |Categorical variable that captures severe out of plane failure related damage to the building in terms of the proportion of overall area that is severely damaged                                |Categorical|
|`damage_out_of_plane_failure_walls_ncfr_insignificant`|Categorical variable that captures insignificant out of plane failure of walls not carrying floor/roof in the building in terms of the proportion of overall area that is insignificantly damaged|Categorical|
|`damage_out_of_plane_failure_walls_ncfr_moderate`     |Categorical variable that captures moderate out of plane failure of walls not carrying floor/roof in the building in terms of the proportion of overall area that is moderately damaged          |Categorical|
|`damage_out_of_plane_failure_walls_ncfr_severe`       |Categorical variable that captures severe out of plane failure of walls not carrying floor/roof in the building in terms of the proportion of overall area that is severely damaged              |Categorical|
|`damage_overall_adjacent_building_risk`               |Adjacent building risk                                                                                                                                                                           |Categorical|
|`damage_overall_collapse`                             |Overall damage assessment for the building - collapse                                                                                                                                            |Categorical|
|`damage_overall_leaning`                              |Overall damage assessment for the building - leaning                                                                                                                                             |Categorical|
|`damage_parapet_insignificant`                        |Categorical variable that captures insignificant parapet-related damage to the building in terms of the proportion of the overall area that is insignificantly damaged                               |Categorical|
|`damage_parapet_moderate`                             |Categorical variable that captures moderate parapet related damage to the building in terms of the proportion of overall area that is moderately damaged                                         |Categorical|
|`damage_parapet_severe`                               |Categorical variable that captures severe parapet-related damage to the building in terms of the proportion of the overall area that is severely damaged                                             |Categorical|
|`damage_roof_insignificant`                           |Categorical variable that captures insignificant roof damage to the building in terms of the proportion of the overall area that is insignificantly damaged                                          |Categorical|
|`damage_roof_moderate`                                |Categorical variable that captures moderate roof damage to the building in terms of the proportion of the overall area that is moderately damaged                                                    |Categorical|
|`damage_roof_severe`                                  |Categorical variable that captures severe roof damage to the building in terms of the proportion of the overall area that is severely damaged                                                        |Categorical|
|`damage_staircase_insignificant`                      |Categorical variable that captures insignificant staircase related damage to the building in terms of the proportion of overall area that is insignificantly damaged                             |Categorical|
|`damage_staircase_moderate`                           |Categorical variable that captures moderate staircase-related damage to the building in terms of the proportion of the overall area that is moderately damaged                                       |Categorical|
|`damage_staircase_severe`                             |Categorical variable that captures severe staircase-related damage to the building in terms of the proportion of the overall area that is severely damaged                                           |Categorical|
|`district_id`                                         |District where the building is located                                                                                                                                                           |Text       |
|`has_damage_beam_failure`                             |Flag variable that indicates if the building has beam failure                                                                                                                                    |Boolean    |
|`has_damage_cladding_glazing`                         |Flag variable that indicates if the building has damaged cladding/glazing                                                                                                                        |Boolean    |
|`has_damage_column_failure`                           |Flag variable that indicates if the building has column failure                                                                                                                                  |Boolean    |
|`has_damage_corner_separation`                        |Flag variable that indicates if the building has corner separation related damage                                                                                                                |Boolean    |
|`has_damage_delamination_failure`                     |Flag variable that indicates if the building has delamination failure                                                                                                                            |Boolean    |
|`has_damage_diagonal_cracking`                        |Flag variable that indicates if the building has diagonal cracking related damage                                                                                                               |Boolean    |
|`has_damage_foundation`                               |Flag variable that indicates if the building has foundational damage                                                                                                                             |Boolean    |
|`has_damage_gable_failure`                            |Flag variable that indicates if the building has gable failure                                                                                                                                   |Boolean    |
|`has_damage_in_plane_failure`                         |Flag variable that indicates if the building has in-plane-failure                                                                                                                                |Boolean    |
|`has_damage_infill_partition_failure`                 |Flag variable that indicates if the building has infill/partition failure                                                                                                                        |Boolean    |
|`has_damage_out_of_plane_failure`                     |Flag variable that indicates if the building has out-plane-failure                                                                                                                               |Boolean    |
|`has_damage_out_of_plane_walls_ncfr_failure`          |Flag variable that indicates if the building has out-of-plane-failure of walls not carrying floor or roof                                                                                        |Boolean    |
|`has_damage_parapet`                                  |Flag variable that indicates if the building has damaged parapet                                                                                                                                 |Boolean    |
|`has_damage_roof`                                     |Flag variable that indicates if the building has roof damage                                                                                                                                     |Boolean    |
|`has_damage_staircase`                                |Flag variable that indicates if the building has damaged staircase                                                                                                                               |Boolean    |
|`has_geotechnical_risk_fault_crack`                   |Flag variable that indicates if the building has geotechnical risks related to fault cracking                                                                                                    |Boolean    |
|`has_geotechnical_risk_flood`                         |Flag variable that indicates if the building has geotechnical risks related to flood                                                                                                             |Boolean    |
|`has_geotechnical_risk_land_settlement`               |Flag variable that indicates if the building has geotechnical risks related to land settlement                                                                                                   |Boolean    |
|`has_geotechnical_risk_landslide`                     |Flag variable that indicates if the building has risk geotechnical risks related to landslide                                                                                                    |Boolean    |
|`has_geotechnical_risk_liquefaction`                  |Flag variable that indicates if the building has geotechnical risks related to liquefaction                                                                                                      |Boolean    |
|`has_geotechnical_risk_other`                         |Flag variable that indicates if the building has any other geotechnical risk                                                                                                                     |Boolean    |
|`has_geotechnical_risk_rock_fall`                     |Flag variable that indicates if the building has geotechnical risk related to rockfall                                                                                                           |Boolean    |
|`has_geotechnical_risk`                               |Flag variable that indicates if the building has geotechnical risk                                                                                                                               |Boolean    |
|`has_repair_started`                                  |Flag variable that indicates if the repair work had started during the time of the survey                                                                                                        |Boolean    |
|`id`                                                  |A unique ID that identifies unique information from all tables                                                                                                                                  |Number     |
|`technical_solution_proposed`                         |Technical solution proposed by the surveyor after assessment                                                                                                                                     |Categorical|
## Table `id_map`

| Variable       | Description                                                   | Type |  
| :------------- | :------------------------------------------------------------ | :--- |  
| `district_id`  | District of residence of the household                        | Text |  
| `ward_id` | specific to a particular municipality, city, or region        | Text |  
| `vdcmun_id`    | Municipality of residence of the household                    | Text |  
| `district_name`    |  the name of the district or administrative division in which a building is located.                    | Text | 
| `vdcmun_name`    |  the name of the Village Development Committee/Municipality where a building is located.  


# Project Scope
The primary objective of this project is to build a machine-learning model capable of predicting the level of damage to buildings in the aftermath of an earthquake. The model will take into account various factors such as building age, foundation type, roof type, land surface condition, and other relevant features available in the dataset.

The project's scope includes the following key tasks:

1. Data Exploration and Preprocessing: Thoroughly explore the dataset to gain insights into the distribution of features, identify missing values, and handle any data anomalies. Perform data preprocessing to ensure the data is suitable for model training.

2. Feature Engineering: Analyze the existing features and create new relevant features, if necessary, to enhance the model's predictive power.

3. Model Selection and Training: Evaluate different machine learning algorithms suitable for regression tasks and select the most appropriate one based on performance metrics. Train the chosen model using the training data.

4. Model Evaluation: Validate the model's performance using suitable evaluation metrics, such as accuracy.

5. Prediction and Interpretation: Apply the trained model to predict the building damage severity on a separate test dataset. Interpret the model's results to understand the key factors influencing the building damage.

