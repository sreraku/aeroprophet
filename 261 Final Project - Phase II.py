# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # AeroProphet

# COMMAND ----------

# Import package for showing images
import matplotlib.pyplot as plt

# COMMAND ----------

# DBTITLE 1,Abstract
# MAGIC %md
# MAGIC The overall objective of this project is to predict departure delays greater than 15 minutes, 2 hours before takeoff. The business case is described in more detail in the next section, but the results of this analysis have significant benefits for both airlines and passengers.
# MAGIC
# MAGIC The specific goals of Phase II were to perform a data join, do a full EDA informed by Phase I, clean up the data, run a logistic regression baseline model, and use LASSO regularization to reduce our feature set. 
# MAGIC
# MAGIC We started with 54 features at the beginning of this phase (including derived features) and ended up with 24. Later iterations of the model had a 22.85% improvement in F1 score over our baseline model.
# MAGIC
# MAGIC In the next phase of this project, we will revisit our engineering pipeline, experiment with additional models, and fine-tune those that perform best. 

# COMMAND ----------

# DBTITLE 1,Business Case
# MAGIC %md
# MAGIC
# MAGIC The ability to anticipate delays will usher in a new era of empowered decision-making for passengers and seamless operations for airlines.
# MAGIC
# MAGIC Armed with this foresight, passengers will be able to adapt their plans in real-time and make more informed choices from the outset.
# MAGIC
# MAGIC Meanwhile, airlines will be able to more proactively address issues and come up with timely resolutions if they're able to foresee delays. Beyond these immediate benefits, this predictive analysis will also shed light on the major causes of delays, which airlines can use to improve their operational efficiency and elevate the customer experience. 
# MAGIC
# MAGIC In a world that puts an increasing emphasis on efficiency, the power of real-time insights will transform the travel landscape.

# COMMAND ----------

# DBTITLE 1,Assignment/Credit Plan
# MAGIC %md
# MAGIC Phase Leader: Sreeram
# MAGIC   
# MAGIC This table will be updated to include a phase's assigments at the beginning of that phase.
# MAGIC
# MAGIC | Phase | Task | Person | Expected Time (in Days) | Start/End Dates |
# MAGIC |----------|----------|----------|----------|----------|
# MAGIC | 2 | Notebook & Tasks Setup | Hailee | 1 | 11/7/23 |
# MAGIC | 2 | Abstract | Hailee | 1 | 11/27/23 |
# MAGIC | 2 | Business Case | Hailee | 1 | 11/8/23 |
# MAGIC | 2 | EDA | Nick, Sreeram | 10 | 11/9/23 - 11/20/23 |
# MAGIC | 2 | Data Cleaning | Landon | 7 | 11/19/23 - 11/25/23 |
# MAGIC | 2 | Feature Engineering | Nick | 7 | 11/19/23 - 11/26/23 |
# MAGIC | 2 | Baseline model | Nick, Landon, Sreeram, Hailee | 4 | 11/21/23 - 11/24/23 |
# MAGIC | 2 | Datasets Join | Landon | 10 | 11/9/23 - 11/20/23 |
# MAGIC | 2 | Feature Selection | Landon, Hailee, Sreeram | 4 | 11/21/23 - 11/24/23 |
# MAGIC | 2 | Evaluation | Hailee | 2  | 11/27/23 - 11/29/23 |
# MAGIC | 2 | Putting together presentation | Hailee, Nick, Landon, Sreeram | 4 | 11/21/23 - 11/29/23 |

# COMMAND ----------

# DBTITLE 1,Joining Data - Extra Credit
# MAGIC %md
# MAGIC We created a custom dataset using publicly available data from the Bureau of Labor Statistics on US flights from 2015 to 2019, which we joined with weather station data from that same time period using station and airport location datasets. 
# MAGIC
# MAGIC Please use the following link to see our Data Join notebook (https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/661347987654537/command/1537154891796692)
# MAGIC
# MAGIC The 4 datasets are summarized below:
# MAGIC | Dataframe | Source | Description
# MAGIC |----------|----------|----------|
# MAGIC | df_flights | Bureau of Labor Statistics | US flight data from 2015 to 2019, containing information on airlines, date, origin and destination airports, flight delay, time of departure, arrival. This is a historical data that can be used to figure out relevant features required to model to predict or classify if and when a flight is delayed.|
# MAGIC | df_weather | unknown | This dataset contains weather information for all the airports (source, destination) with columns capturing wind speed, dewpoint, visibility, elevation, humidity, precipitation which have a direct impact on the departure of a flight.|
# MAGIC |df_airports | DataHub | This dataset contains location, size, and identification information about US airports.|
# MAGIC |df_stations | unknown | This dataset contains location and identification information about weather stations and their distance to nearby airports. |
# MAGIC
# MAGIC The simplified database diagram below illustrates the relationships we were able to leverage in order to create our final dataset ('df_FSW') for the rest of the analysis.

# COMMAND ----------

import matplotlib.pyplot as plt 

plt.figure(figsize=(10, 10))
image = plt.imread('Images/DatasetDiagram.png')
plt.imshow(image)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC To join the most recent weather of the closest station to the departing flight's origin airport, a time 'leveling' feature was derived for both the flights and weather datasets. This was done by collapsing every weather and flight to one of 8 possible time levels, each comprising a 3 hour time window. Since multiple weather reports for a station may have been reported within this window, the most recent weather reporting information to the time level was used to join with the flights dataset. This facilitated a compression of the weather dataset creating a more scalable join process. The table below summarizes the final joined dataset's dataframe size, memory storage, and join time requirements. 
# MAGIC
# MAGIC
# MAGIC | Table | Rows | Columns | Memory (GB) | Time to Join (HH:MM:SS) |
# MAGIC |----------|----------|----------|----------|----------|
# MAGIC | df_flights | 74,177,433| 109 | 2.93 | None|
# MAGIC | df_weather | 898,983,399| 124 | 35.05 | None|
# MAGIC | df_stations | 5,004,169| 12 | 1.3 | None|
# MAGIC | df_airports | 57,421| 12 | 0.01 | None|
# MAGIC | flights | 72,515,921| 292 | 43.5 | 02:13:07|
# MAGIC
# MAGIC Cluster Details: DBR 13.3 LTS ML, Spark 3.4.1, Scala 2.12, Standaard_DS3_v2, 14GB, 4 Cores, Standard_DS3_v2, 84GB, 24 Cores, 1-6 workers

# COMMAND ----------

# DBTITLE 1,EDA
# MAGIC %md
# MAGIC
# MAGIC In this section we'll describe our exploration of the available features. We began by identifying duplicates, exploring missing data, converting datatypes, reviewing the distribution of each variable, and finally calculating the Pearson correlation for each variable with our outcome variable. To accomplish this, we first identified a variable of interest. We quickly selected "DEP_DEL15" which the flights [data dictionary](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ) describes as "Departure Delay Indicator, 15 Minutes or More (1=Yes)".
# MAGIC
# MAGIC Please use the following link to see our full data cleaning process: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/2798664673116916
# MAGIC
# MAGIC The exploratory section of our EDA can be reviewed here: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/2798664673116918

# COMMAND ----------

# MAGIC %md
# MAGIC *Duplicates values*
# MAGIC
# MAGIC Upon loading the dataset, we found 31,133,308 duplicate values. We found the orignal source flights dataset had a duplicate for every record which likely drove this impact on our joined data. Since we aim to predict delays for each flight once, these duplicates were dropped from the dataset.

# COMMAND ----------

import matplotlib.pyplot as plt 

plt.figure(figsize=(20, 20))
image = plt.imread('Images/duplicates_count.png')
plt.imshow(image)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC *Missing Values & Data Types*
# MAGIC
# MAGIC We next created a table showing the "Null" percentage of each column and their datatypes. The image below shows the top 10 values. As you can see, many features contain no information at all. Particularly features relating to [monthly averages and totals](https://www.ncei.noaa.gov/pub/data/cdo/documentation/LCD_documentation.pdf). It seems the National Oceanic and Atmospheric Administration has not maintained these fields. Given our purpose is to predict at a much more granular level (2 hrs before departure), this should not pose an issue. We decided to drop these features and any others with greater than 99% missing values given then contain little to no information. 
# MAGIC
# MAGIC Thankfully, the dataset our team created did not have any missing values for our outcome variable, DEP_DEL15. However, initial analysis with other datasets (OTPW) did show some missing information in this field so our pipeline includes steps to drop such rows from our dataset so we do not infer the wrong label. Afterall, we do not want to train on delays that never occured. The rest of the features contained little missing data so we opted to impute them with the mean.
# MAGIC
# MAGIC Lastly, you'll notice that many features are stored as strings despite clearly representing numerical data. We converted all such fields to the proper data types (using the ast libary to convert them to python literals). This ensure we can generate summary statistics and enables us to train. The string values representing categorical data were left as strings but later one hot encoded to enable training.

# COMMAND ----------

import matplotlib.pyplot as plt 

plt.figure(figsize=(20, 20))
image = plt.imread('Images/missing_values.png')
plt.imshow(image)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC In addition to dropping features with high amounts of missing values, we decided to drop any feature that is only known after take off. This avoids data leakage since our goal is to predict flight delays no more than 2 hours in advance of depature.
# MAGIC
# MAGIC Several other features were dropped because they were categorial and contained too many values to one hot-encode, datetime objects (which are not by themselves understandable to a machine learning model), or were redundant to other features. 
# MAGIC
# MAGIC Note the following are features we engineered which we'll discuss in more detail in a coming section:   <br>  <br> 
# MAGIC   
# MAGIC - previous_flight_delay
# MAGIC - days_to_nearest_holiday 
# MAGIC - Plane_Delays_last_24H
# MAGIC - airport_delays_in_previous_24_hours
# MAGIC - region

# COMMAND ----------

# DBTITLE 0,Summary statistics
# MAGIC %md
# MAGIC   *Summary statistics*   <br>  <br> 
# MAGIC   
# MAGIC   - We captured some statistics on the flight delay for the entire data set and found the following
# MAGIC     * Total Flight Count: 64457088
# MAGIC     * Number of Flights Delayed by More Than 15 Minutes: 11175240
# MAGIC     * Delayed Flight Percentage: 17.33748815956439
# MAGIC     * On Time Flight Percentage: 82.66251184043561
# MAGIC
# MAGIC During the time period between 2015-2019, 17.33% of total flights were delayed which is considerable amount. Predicting these delays in advance will provide a significant advantage to our customers and operations by reducing distributions.

# COMMAND ----------

plt.figure(figsize=(20, 20))
image = plt.imread('Images/Mean,median,stddevofvars.png')
plt.imshow(image)
plt.show()



# COMMAND ----------

# MAGIC %md
# MAGIC   
# MAGIC The bar plot above shows mean, median, and standard deviation on 3 months of data. This helps us understand the scale and distribution of each feature. Clearly, some features have a much larger range than others. To account for this, standardization was applied as part of EDA by dividing by the standard deviation of each variable.

# COMMAND ----------

# DBTITLE 0,Pearson's Correlation Analysis
# MAGIC %md
# MAGIC
# MAGIC *Pearson's Correlation Analysis*   <br>
# MAGIC
# MAGIC After dropping several non-relevant columns from the joined dataset, we examined pearson's correlation between each numerical feature and our outcome variable DEP_DEL15. Note that we'd prefer to analyze the correlation between our features and a continous delay variable (instead of binary) but this was previously dropped from our dataset to avoid data leakage. We plan to recalculate correlation in the future.
# MAGIC
# MAGIC The resulting bar chart below shows a strong correlation between depature delays and each of our engineered features (previous_flight_delay, days_to_nearest_holiday, Plane_Delays_last_24H, airport_delays_in_previous_24_hours, region). Other features show no relationship and were dropped from our dataset as a result ('FLIGHTS', 'dest_station_dis', 'origin_station_dis', 'ELEVATION', 'HourlyPressureChange', 'HourlyPressureTendency', 'HourlyStationPressure').

# COMMAND ----------

plt.figure(figsize=(20, 20))
image = plt.imread('Images/corr_to_dep_del15.png')
plt.imshow(image)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Next we created a correlation heatmap to examine relationships between features. We found that DISTANCE_GROUP and HourlyWetBulbTemperature were strongly correlated to other features and could be dropped.

# COMMAND ----------

import matplotlib.pyplot as plt 

plt.figure(figsize=(20, 20))
image = plt.imread('Images/Correlation heatmap.png')
plt.imshow(image)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC *Exploratory plots*
# MAGIC
# MAGIC After selecting a set of preliminary features, we explored where delays were occuring and who was causing them. Each plot below gives a different perspecitive on this question.

# COMMAND ----------

plt.figure(figsize=(20, 20))
image = plt.imread('Images/Airports geoplot.png')
plt.imshow(image)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC   - A geo-plot captured for top 10 airport delays shows Atlanta, Chicago O'Hare, Dallas fortworth airports picking the top 3 spots of most delayed flights from an airport.

# COMMAND ----------

plt.figure(figsize=(20, 20))
image = plt.imread('Images/Delay distribution.png')
plt.imshow(image)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC   - Looking at the delay distribution, it is very clear that the data is more skewed towards ontime departures. This tells that this is a highly imbalanced data and choosing accuracy would be heavily biased.
# MAGIC

# COMMAND ----------

plt.figure(figsize=(20, 20))
image = plt.imread('Images/airlines delay.png')
plt.imshow(image)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC   - A simple bar plot was done to capture the top 10 airlines which are late most of the times over the 60M period and Southwest, American Airlines and Delta airlines take the top 3 spots. 
# MAGIC

# COMMAND ----------

# DBTITLE 1,Feature Engineering
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC With 292 features, dimensionality reduction was essential for a viable, scalable model. To do this, we evaluated redundancy, Pearson correlation with target variable, and LASSO regularization. We ultimately selected 24 variables with information we believed to be most useful in predicting if a flight will be delayed 2 hours prior to scheduled departure.
# MAGIC
# MAGIC Feature transformations included standardization to bring each feature value to similar value scales, data type conversions from string to the appropriate integer/float/datetime, and one hot encoding. We imputed the mean for missing values. We also applied weights to account for class imbalance as the number of "not delayed" flights heavily outweighed "delayed" flights.
# MAGIC
# MAGIC The table below details our final raw and derived features.
# MAGIC
# MAGIC | Feature | Dataset Origination | Defintion | 
# MAGIC |----------|----------|----------|
# MAGIC | OP_UNIQUE_CARRIER_dummy_DL | flights | dummy variable identifying a flight as being flown by Delta Airlines | 
# MAGIC | OP_UNIQUE_CARRIER_dummy_WN | flights | dummy variable identifying a flight as being flown by SouthWest Airlines | 
# MAGIC | OP_UNIQUE_CARRIER_dummy_OO | flights | dummy variable identifying a flight as being flown by  SkyWest Airlines | 
# MAGIC | OP_UNIQUE_CARRIER_dummy_B6 | flights | dummy variable identifying a flight as being flown by  JetBlue Airlines | 
# MAGIC | OP_UNIQUE_CARRIER_dummy_AS | flights | dummy variable identifying a flight as being flown by  Alaska Airlines | 
# MAGIC | OP_UNIQUE_CARRIER_dummy_HA | flights | dummy variable identifying a flight as being flown by  Hawaiian Airlines | 
# MAGIC | OP_UNIQUE_CARRIER_dummy_F9 | flights | dummy variable identifying a flight as being flown by  Frontier Airlines | 
# MAGIC | DAY_OF_WEEK_dummy_2 | flights | dummy variable identifying if the flight departed on a Monday.  | 
# MAGIC | DAY_OF_WEEK_dummy_3 | flights | dummy variable identifying if the flight departed on a Tuesday. | 
# MAGIC | DISTANCE | flights | Integer distance between a flight's origin airport and destination airport | 
# MAGIC | CRS_DEP_HOUR | flights | local Datetime value of a flights scheduled departure hour | 
# MAGIC | CRS_ARR_HOUR | flights | local Datetime value of a flights scheduled arrival hour |
# MAGIC | HourlyAltimeterSetting | Weather | Float value measuring average hourly altimeter reading from closest weather station to departing airport | 
# MAGIC | HourlyDryBulbTempature | Weather | Float value measuring average hourly dry bulb tempature reading from closest weather station to departing airport | 
# MAGIC | HourlyPrecipitation | Weather | Float value measuring average hourly precipitation reading from closest weather station to departing airport | 
# MAGIC | HourlyRelativeHumidity | Weather | Float value measuring average hourly relative humidity reading from closest weather station to departing airport | 
# MAGIC | HourlyVisibility | Weather | Float value measuring average vsibility reading from closest weather station to departing airport | 
# MAGIC | HourlyWindSpeed | Weather | Float value measuring average hourly wind speed from closest weather station to departing airport | 
# MAGIC | previous_flight_delay | derived from flights | Boolean value that is '1' if a departing flight's scheduled aircraft tailnumber was delayed on its previous flight| 
# MAGIC | Plane_Delays_last_24H | derived from flights | Integer value of the total number of times a departing flight's scheduled aircraft tailnumber was delayed in the previous 24 hours|  
# MAGIC | airport_delays_in_previous_24_hours | derived from flights | Integer value of the total number of flights delayed at the departing airport in the previous 24 hours|  
# MAGIC | days_to_nearest_holiday | derived from flights | Integer value of the total number of days to the closest US holiday before or after the scheduled flight |  
# MAGIC | region_dummy_Southeast | derived from flights | dummy variable identifying if the flight departed from an airport in the "Southeast" region of the United States. | 
# MAGIC | region_dummy_West | derived from flights | dummy variable identifying if the flight departed from an airport in the "West" region of the United States. | 
# MAGIC

# COMMAND ----------

# DBTITLE 1,Logistic & LASSO Regressions
# MAGIC %md
# MAGIC
# MAGIC We used logistic regression for all of our initial modeling. Please use the following link to see our Phase II - Modeling code (https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/2798664673116921).

# COMMAND ----------

# MAGIC %md 
# MAGIC *K Cross Folds*
# MAGIC
# MAGIC To set up for modeling, we first created 5 train/validate folds of the data according to common practice for time series data (see visual below). After splitting the full data into train (2015-2019) and test (2019) sets, we used the train data for our folds and for modeling. 

# COMMAND ----------

plt.figure(figsize=(20, 20))
image = plt.imread('Images/K Cross Folds.png')
plt.imshow(image)
plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC *Baseline Models*
# MAGIC
# MAGIC We used logistic regression for our initial models as it's a simple way to model classification problems that's easy to interpret. Our first logistic regression contained the 50 post-data-cleaning features without our derived features. This was followed by a logistic regression containing both the 50 original features and the 4 derived features. 

# COMMAND ----------

# MAGIC %md *LASSO Modeling*
# MAGIC
# MAGIC To identify the most useful features for model prediction, we ran a  LASSO regression model, excluding any derived features, on each of the 5 training folds. We then analyzed the resulting model coefficients, storing features which were not zeroed out by the model as a set. This resulted in 5 unique combinations of non-zeroed features from which we derived a sixth combination, which included every feature that had appeared in at least one of the other 5 feature sets.  
# MAGIC | Feature Set | Features Included |
# MAGIC |----------|----------|
# MAGIC | Set 1 | Non-zero features from fold 1 |
# MAGIC | Set 2 | Non-zero features from fold 2 |
# MAGIC | Set 3 | Non-zero features from fold 3 | 
# MAGIC | Set 4 | Non-zero features from fold 4 |
# MAGIC | Set 5 | Non-zero features from fold 5 |
# MAGIC | Set 6 | All non-zero features combined  |
# MAGIC
# MAGIC Using this list of 6 possible feature combinations, we trained logistic regression models on each feature set. We used their average F1 and AUC scores across all training folds for evaluation. From this, we identified 'Set 3' as the feature set with greatest predictive power for our model, as shown in the visual below. 
# MAGIC

# COMMAND ----------

plt.figure(figsize=(8, 8))
image = plt.imread('Images/LASSOFeatureSelect.png')
plt.imshow(image)
plt.show()

# COMMAND ----------

# DBTITLE 1,Grid Search
# MAGIC %md
# MAGIC
# MAGIC Using Grid Search, we were able to slightly improve our baseline model performance by increasing the L2 regularlization and limiting the max iterations of gradient descent. The selected parameters were 0.02 for regularization at 20 iterations.

# COMMAND ----------

# DBTITLE 1,Evaluation
# MAGIC %md
# MAGIC
# MAGIC F1 score was our primary evaluation metric because it uses a balanced approach between disrupting flight decisions and providing a reliable metric. If there were any tiebreakers or close calls, we decided we’d use AUC. Both metrics are common for evaluating classification models, like logistic regression. AUC also summarizes performance across different classification thresholds.
# MAGIC
# MAGIC The best LASSO feature set was the one selected by Fold 3, which contained 20 features. We added the 4 derived features back into the 20-LASSO feature set and the 33-LASSO feature set for final evaluation.
# MAGIC
# MAGIC As you can see in the first visual below, there’s a steady improvement in the validation metrics from our initial baseline logistic regression model through the iterations after LASSO feature selection. We saw a 22.85% increase in F1 score in the averaged validation folds.
# MAGIC
# MAGIC The training metrics can be seen in the second visual below. The train and validation F1 scores are not very far off, so we do not believe overfitting occurred. It is worth noting that the initial model with derived features performed similarly to the best LASSO model in our training folds. Therefore, it may be worth revisiting the full feature set when we explore different models in the next phase.

# COMMAND ----------

plt.figure(figsize=(8, 8))
image = plt.imread('Images/Phase II Val Metrics.png')
plt.imshow(image)
plt.show()

# COMMAND ----------

plt.figure(figsize=(8, 8))
image = plt.imread('Images/Phase II Train Metrics.png')
plt.imshow(image)
plt.show()

# COMMAND ----------

# MAGIC %md Runtime Table
# MAGIC
# MAGIC | Job | Runtime |
# MAGIC |----------|----------|
# MAGIC | Post-Cleaning Checkpoint | 216 minutes |
# MAGIC | Create Folds | 18 minutes |
# MAGIC | Baseline Logistic Model w/o Derived Features | 53 minutes |
# MAGIC | Baseline Logistic Model w Derived Features  | 51 minutes |
# MAGIC | LASSO Regularization  | 49 minutes |
# MAGIC | Grid Search  | 69 minutes |
# MAGIC | Logistic on All LASSO Feature Sets in Each Fold | 420 minutes |
# MAGIC | Logistic on Full LASSO Feature Set| 60 minutes |
# MAGIC | Logistic on Best LASSO Feature Set | 55 minutes |

# COMMAND ----------

# DBTITLE 1,Conclusion
# MAGIC %md
# MAGIC
# MAGIC Our preliminary exploration and modeling indicates a strong proof of concept that a model can be used to effectively make predictions on flight delays 2 hours in advance. Our hypothesis is that using thoughtful data engineering and feature selection we will be able to accurately forecast flight delays greater than 15 minutes, 2 hours ahead of time. Our results will help both airlines and customers make informed decisions and enhance operational efficiency.
# MAGIC
# MAGIC We started Phase II by joining the data. This was followed by the cyclical EDA and data cleaning process. During feature engineering, we added 4 new derived features to our dataset. As a next step, we created data cuts and cross folds for training, validation, and holdout. We then conducted a series of logistic regressions, LASSO regularization, and Grid Search. In our evaluation of these results, we identified a set of 24 features that produced a notable increase in F1 score over our initial baseline model.
# MAGIC
# MAGIC In the next phase, we plan to start by refining our existing pipleine based on feedback. This may include re-doing our pearson correlation analysis with continuous variables, recreating our folds with overlap, fold weighting in evaluation to account for staleness, exploring interaction terms, extracting more from our weather dataset, and potentially creating more derived features. After that, we’re going to continue exploring models, features, and hyperparameters to see if we improve our F1 score. 