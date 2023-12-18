# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # AeroProphet
# MAGIC
# MAGIC Group 2.2

# COMMAND ----------

# Import package for showing images
import matplotlib.pyplot as plt

# COMMAND ----------

# DBTITLE 1,Team
# Import package for showing images
import matplotlib.pyplot as plt

# Create a figure with a 2x2 grid
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 3))

# Image 1
image1 = plt.imread('Images/Hailee Schuele.png')
axes[0].imshow(image1)
axes[0].set_title('Hailee Schuele')
axes[0].set_xlabel('hschuele@berkeley.edu')
axes[0].set_xticks([])
axes[0].set_yticks([])

# Image 2
image2 = plt.imread('Images/Landon Yurica.png')
axes[1].imshow(image2)
axes[1].set_title('Landon Yurica')
axes[1].set_xlabel('lyurica@berkeley.edu')
axes[1].set_xticks([])
axes[1].set_yticks([])

# Image 3
image3 = plt.imread('Images/Sreeram Ravinoothala.png')
axes[2].imshow(image3)
axes[2].set_title('Sreeram Ravinoothala')
axes[2].set_xlabel('sreeram@berkeley.edu')
axes[2].set_xticks([])
axes[2].set_yticks([])

# Image 4
image4 = plt.imread('Images/Nick Johnson.png')
axes[3].imshow(image4)
axes[3].set_title('Nick Johnson')
axes[3].set_xlabel('nickjohnson@berkeley.edu')
axes[3].set_xticks([])
axes[3].set_yticks([])

# Adjust spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC *Phase Leaders*
# MAGIC - Phase I: Hailee
# MAGIC - Phase II: Nick
# MAGIC - Phase III: Sreeram
# MAGIC - Phase IV: Landon

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC *Final Phase Assignment Table*
# MAGIC
# MAGIC |Date | Sreeram | Nick | Hailee | Landon | 
# MAGIC |------|------|------|------|------|
# MAGIC |12/5| DL experiments| update cv fold data, clean up blob| cv staleness metric scaling, RF & GBDT experiments | F.E. max_weather |
# MAGIC |12/6| DL experiments| Restructure Cross Fold Data | RF & GBDT experiments| F.E. airport size|
# MAGIC |12/7| DL experiments| train MLP on 2015-2018 data | RF & GBDT experiments | F.E. |
# MAGIC |12/8| DL experiments| grid_search best model on 2015-2018 data| RF & GBDT experiments| F.E. airport centrality|
# MAGIC |12/9| DL experiments| re-run correlation analysis| RF & GBDT experiments| Re-checkpoint data|
# MAGIC |12/10| slides| grid_search best model| slides| E.C. data cleaning |
# MAGIC |12/11| slides|slides | slides| slides|
# MAGIC |12/12| slides|slides |slides |slides |
# MAGIC |12/13| presentation| presentation |presentation |presentation |
# MAGIC |12/15| final report | final report |final report |final report |
# MAGIC

# COMMAND ----------

# DBTITLE 1,Abstract
# MAGIC %md
# MAGIC
# MAGIC The overall objective of this project was to predict departure delays greater than 15 minutes, 2 hours before takeoff. The business case is described in more detail in the next section, but the results of this analysis have significant benefits for both airlines and passengers.
# MAGIC
# MAGIC In Phase I of this project, we sorted out project management and our computing environment. We also made plans for our data pipeline and did our initial EDA to identify variables of interest. 
# MAGIC
# MAGIC In Phase II of this project, we first performed a join to combine our datasets. This was followed by the iterative data cleaning, EDA, and feature engineering process. We ran a logistic regression model as our baseline, after which we conducted LASSO regularizaiton. The resulting 24 features were used in another logistic regression, which showed substantial improvements over our baseline.
# MAGIC
# MAGIC In the final Phase III, we started by performing additional feature engineering and EDA to ensure we were getting the most utility out of the data. We then experimented with a variety of models (logistic regression, random forest, gradient boosted decision trees, multilayer perceptron, and long short-term memory), hyperparameters, and LASSO regularization to see how they performed on the validation data. The best model was selected, trained on the full data, and tested on an unseen data cut to give us our final performance.
# MAGIC
# MAGIC Our final model showed a 35% increase in F1 score from our baseline model. Further investigation showed that precision and recall were very similar, indicating we have a balanced model.

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

# DBTITLE 1,Timeline
plt.figure(figsize=(20, 20))
image = plt.imread('Images/project_plan_gantt_chart.png')
plt.imshow(image)
plt.show()

# COMMAND ----------

# DBTITLE 1,Outcome Definition
# MAGIC %md
# MAGIC
# MAGIC For this project, we are dealing with a binary classification problem since our goal is to categorize each flight as "delayed" or "not delayed". With classification tasks, accuracy is a poor metric since always predicting the majority class can lead to high accuracy (especially in unbalanced datasets). As a result, we plan to focus on other evaluation metrics: *F1 Score* and *AUC*.
# MAGIC
# MAGIC **F1 Score**: To get an idea of performance at a specific threshold value, we can leverage the F1 score. This is the harmonic mean between recall & precision.
# MAGIC $$2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
# MAGIC
# MAGIC **AUC**: There is a tradeoff between recall and precision as controlled by a threshold value. Typically, if the probabliy of the positive class is greater than 0.5 we predict a positive and negative otherwise. As the threshold is increased, we become more conservative and precision will increase at the expense of recall. As the threshold is lowered, the inverse occurs. We can visualize model performance at all threshold values by plotting an ROC curve (receiver operating characteristic curve). Finding the area under this curve, called the AUC, summarizes the total performance of the model for all threshold values.
# MAGIC
# MAGIC F1 score was our primary evaluation metric because it uses a balanced approach between disrupting flight decisions and providing a reliable metric. If there were any tiebreakers or close calls, we decided we’d use AUC. Both metrics are common for evaluating classification models, like logistic regression. AUC also summarizes performance across different classification thresholds.

# COMMAND ----------

# DBTITLE 1,Modeling Pipeline
plt.figure(figsize=(20, 20))
image = plt.imread('Images/Pipeline.png')
plt.imshow(image)
plt.show()

# COMMAND ----------

# DBTITLE 1,Data
# MAGIC %md 
# MAGIC Our model is built on publicly available data from the Bureau of Labor Statistics on US flights from 2015 to 2019, and weather station data from the National Center for Environmental Information during the same time period.
# MAGIC
# MAGIC The 4 datasets are summarized in both tables below:
# MAGIC | Dataset | Source | Description
# MAGIC |----------|----------|----------|
# MAGIC | Flights | Bureau of Labor Statistics | US flight data from 2015 to 2019, containing information on airlines, date, origin and destination airports, flight delay, time of departure, arrival. This is a historical data that can be used to figure out relevant features required to model to predict or classify if and when a flight is delayed.|
# MAGIC | Weather | National Center for Environmental Information | This dataset contains weather information including  wind speed, dewpoint, visibility, elevation, humidity, precipitation, in hourly, weekly, and monthly intervals from weather stations located around the world.|
# MAGIC |Airport | DataHub | This dataset contains location, size, and identification information about US airports.|
# MAGIC |Weather Stations | DataHub | This dataset contains location and identification information about weather stations and their distance to nearby airports. |
# MAGIC
# MAGIC
# MAGIC Dataset dimensions and memory requirements
# MAGIC |Table | Rows | Columns | Memory (GB) |
# MAGIC | -------- | ------- | -------- | ------- | 
# MAGIC | df_flights | 74,177,433| 109| 2.93| 
# MAGIC | df_weather | 898,983,399 | 124 |35.05 |
# MAGIC | df_stations | 5,004,169 | 12|1.3 |
# MAGIC | df_airports | 57,421| 12 |0.01 | 
# MAGIC

# COMMAND ----------

# MAGIC %md *Data Join*
# MAGIC
# MAGIC We were able to combine these four datasets through a series of data transformations and joins, leveraging the relationships illustrated in the simplified schematic below. 
# MAGIC

# COMMAND ----------

import matplotlib.pyplot as plt 

plt.figure(figsize=(10, 10))
image = plt.imread('Images/DatasetDiagram.png')
plt.imshow(image)
plt.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC The table below briefly summarizes the steps involved in our data join:
# MAGIC
# MAGIC | Step | Transformation | Notes |
# MAGIC |----------|----------|----------|
# MAGIC | 1| Calculate min distance of weather stations to airports |  | 
# MAGIC |2| Join Airport and Weather Station | Airports['ident'] ==Station[`neighbor_call`] |
# MAGIC |3| Join Flights| Flights[`ORIGIN`] == Airports_Stations[`iata_code`], Flights[`DEST`] == Airports_Stations[`iata_code`]|
# MAGIC |4| Convert Flight and Weather times to UTC| |
# MAGIC |5| Create time key in Flights and Weather| Flight and weather time rounded to nearest hour |
# MAGIC |6| Calculate Min, Max, Avg. and Most Recent Weather reports within 2 hour interval for every Station| |
# MAGIC |6| Weather from 2 hours prior joined with Flights | Flights_Airports_Weather['origin_station_id'] == Weather['STATION`] & FAW[`Time_key]== Weather[`time_key_2hr_utc`]|  
# MAGIC
# MAGIC
# MAGIC
# MAGIC Joined data details and join time
# MAGIC |Table | Rows | Columns | Memory (GB) | Run Time (HH:MM:SS) |
# MAGIC | -------- | ------- | -------- | ------- | ------- |
# MAGIC | df_flights | 74,177,433| 109| 2.93| None |
# MAGIC | df_weather | 898,983,399 | 124 |35.05 | None|
# MAGIC | df_stations | 5,004,169 | 12|1.3 | None|
# MAGIC | df_airports | 57,421| 12 |0.01 | None|
# MAGIC | df_FSW | 72,515,921| 292 |43.5  | 02:13:00|
# MAGIC | df_FSW_Clean | 64,457,088 | 87 | 64.78 |06:18:00|
# MAGIC
# MAGIC Cluster Details:
# MAGIC DBR 13.3 LTS ML, Spark 3.4.1, Scala 2.12,
# MAGIC Standaard_DS3_v2, 14GB, 4 Cores, Standard_DS3_v2, 84GB, 24 Cores, 1-6 workers
# MAGIC
# MAGIC Please use the following link to see our full Phase III - Data Join code (https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/661347987654537/command/15371548917966927)

# COMMAND ----------

# MAGIC %md *Train/Test Split and Cross Folds*
# MAGIC
# MAGIC Since our data is temporal in nature, we used standard time series methods for the train/validate/test splits and cross validation. This prevents data leakage by ensuring data in the future is not used to make predictions about the past. 
# MAGIC
# MAGIC To set up for modeling, we first split our train data (2015-2018) into train/validate cross folds. Each sequentual fold incorporates the prior validation dataset as illustrated in the figure below. We set aside 100 days of 2018 data for our larger validation set to be used during intermediary modeling and hyperparameter tuning. The test set was comprised of 2019 data and remained unused until our final model evaluation.

# COMMAND ----------

plt.figure(figsize=(20, 20))
image = plt.imread('Images/Cross Folds New.png')
plt.imshow(image)
plt.show()

# COMMAND ----------

# DBTITLE 1,EDA & Data Cleaning
# MAGIC %md
# MAGIC
# MAGIC Please use the following link to see our full Phase III - EDA code (https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1346418319517318)
# MAGIC
# MAGIC Our preliminary data cleaning included dropping columns with substantial missing data, redundant information, or that risked data leakage. 
# MAGIC
# MAGIC After dropping duplicate rows, we used mean imputation where applicable to fill missing data. We opted for using the mean over median because median resulted in increased shuffle operations resulting significantly longer compute requirements. 

# COMMAND ----------

# MAGIC %md *Pearson Correlation Analysis*
# MAGIC
# MAGIC To evaluate our features, we ran a pearson correlation analysis. First, we checked the correlation between our features and found that the maximum and minimum features often correlated to the average of the same measurement. In cases where max or min correlated over 95% to the average, we decided to mantain only the average column.
# MAGIC
# MAGIC Next we reviewed the correlation of each feature with a continous version of our outcome variable (DEP_DELAY). This was helpful in reviewing feature importance. Any non-dervived feature with 0.01 correlation or lower was dropped from our dataset.
# MAGIC
# MAGIC Features Dropped Due to Low Correlation:
# MAGIC
# MAGIC - FLIGHTS
# MAGIC - dest_station_dis
# MAGIC - origin_station_dis
# MAGIC - ELEVATION
# MAGIC - HourlyPressureTendency
# MAGIC - HourlyStationPressure
# MAGIC - HourlyWetBulbTemperature
# MAGIC - Avg_HourlyWindDirection
# MAGIC - Min_HourlyWindSpeed

# COMMAND ----------

plt.figure(figsize=(20, 20))
image = plt.imread('Images/continous_correlation.png')
plt.imshow(image)
plt.show()

# COMMAND ----------

# DBTITLE 1,Feature Engineering
# MAGIC %md

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC With 292 features, dimensionality reduction was essential for a viable, scalable model. To do this, we evaluated redundancy, Pearson correlation with target variable, and LASSO regularization. We ultimately selected 40 variables with information we believed to be most useful in predicting if a flight will be delayed 2 hours prior to scheduled departure.
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
# MAGIC | OP_UNIQUE_CARRIER_dummy_UA | flights | dummy variable identifying a flight as being flown by  United Airlines | 
# MAGIC | OP_UNIQUE_CARRIER_dummy_B6 | flights | dummy variable identifying a flight as being flown by  JetBlue Airlines | 
# MAGIC | OP_UNIQUE_CARRIER_dummy_AS | flights | dummy variable identifying a flight as being flown by  Alaska Airlines | 
# MAGIC | OP_UNIQUE_CARRIER_dummy_HA | flights | dummy variable identifying a flight as being flown by  Hawaiian Airlines | 
# MAGIC | OP_UNIQUE_CARRIER_dummy_F9 | flights | dummy variable identifying a flight as being flown by  Frontier Airlines | 
# MAGIC | OP_UNIQUE_CARRIER_dummy_VX | flights | dummy variable identifying a flight as being flown by  Virgin America | 
# MAGIC | OP_UNIQUE_CARRIER_dummy_NK | flights | dummy variable identifying a flight as being flown by  Spirit Airlines | 
# MAGIC | OP_UNIQUE_CARRIER_dummy_OH | flights | dummy variable identifying a flight as being flown by  PSA America | 
# MAGIC | OP_UNIQUE_CARRIER_dummy_US | flights | dummy variable identifying a flight as being flown by  Noble Air Charter | 
# MAGIC | DAY_OF_WEEK_dummy_1 | flights | dummy variable identifying if the flight departed on a Sunday.  | 
# MAGIC | DAY_OF_WEEK_dummy_2 | flights | dummy variable identifying if the flight departed on a Monday. |  
# MAGIC | DAY_OF_WEEK_dummy_3 | flights | dummy variable identifying if the flight departed on a Tuesday. |
# MAGIC | DAY_OF_WEEK_dummy_4 | flights | dummy variable identifying if the flight departed on a Wednesday.  | 
# MAGIC | DAY_OF_WEEK_dummy_5 | flights | dummy variable identifying if the flight departed on a Thursday. |  
# MAGIC | region_dummy_Midwest | derived from flights | dummy variable identifying if the flight departed from an airport in the "Midwest" region of the United States. | 
# MAGIC | region_dummy_West | derived from flights | dummy variable identifying if the flight departed from an airport in the "West" region of the United States. | 
# MAGIC | origin_type_dummy_large_airport| Airports | dummy variable identifying a origin airport as being large |
# MAGIC | MONTH| flights | Integer value representing the Month of a departing flight |
# MAGIC | DAY_OF_MONTH | flights | Integer representing the numerical day of the month of a departing flight |
# MAGIC | DISTANCE | flights | Integer distance between a flight's origin airport and destination airport | 
# MAGIC | pagerank| derived from flights | float32 Pagerank score of a origin airport measuring its centrality |
# MAGIC | dest_airport_lat | airports | float32 latitude of destination airport |
# MAGIC | dest_airport_lon | airports | float32 longitude of destination airport |
# MAGIC | origin_airport_lat | airports | float32 latitude of origin airport |
# MAGIC | origin_airport_lon | airports | float32 longitude of origin airport |
# MAGIC | CRS_DEP_HOUR | flights | local Datetime value of a flights scheduled departure hour | 
# MAGIC | CRS_ARR_HOUR | flights | local Datetime value of a flights scheduled arrival hour |
# MAGIC | HourlyAltimeterSetting | Weather | float32 value measuring most recent hourly altimeter reading from closest weather station to departing airport 2 hours prior | 
# MAGIC | HourlyDryBulbTempature | Weather | float32 value measuring most recent hourly Dry Bulb reading from closest weather station to departing airport 2 hours prior |
# MAGIC | HourlyDewPointTemperature | Weather | float32 value measuring most recent hourly Dew Point Temperature reading from closest weather station to departing airport 2 hours prior | 
# MAGIC | HourlyRelativeHumidity | Weather | float32 value measuring most recent relative humdity reading from closest weather station to departing airport 2 hours prior |
# MAGIC | HourlyPrecipitation | Weather | float32 value measuring average hourly precipitation reading from closest weather station to departing airport | 
# MAGIC | HourlyRelativeHumidity | Weather | float32 value measuring average hourly relative humidity reading from closest weather station to departing airport | 
# MAGIC | HourlyWindSpeed | Weather | float32 value measuring most recent hourly wind speed from closest weather station to departing airport | 
# MAGIC | Avg_HourlyPrecipitation | Weather | float32 value measuring averag hourly precipitation from closest weather station to departing airport |
# MAGIC | Avg_HourlySeaLeavelPressure | Weather | float32 value measuring averag hourly Sea Level Pressure from closest weather station to departing airport | 
# MAGIC | Avg_HourlyVisibility | Weather | float32 value measuring averag hourly visibility from closest weather station to departing airport | 
# MAGIC | Avg_HourlyWindSpeed | Weather | float32 value measuring averag hourly Wind speed from closest weather station to departing airport | 
# MAGIC | previous_flight_delay | derived from flights | Boolean value that is '1' if a departing flight's scheduled aircraft tailnumber was delayed on its previous flight| 
# MAGIC | Plane_Delays_last_24H | derived from flights | Integer value of the total number of times a departing flight's scheduled aircraft tailnumber was delayed in the previous 24 hours|  
# MAGIC | days_to_nearest_holiday | derived from flights | Integer value of the total number of days to the closest US holiday before or after the scheduled flight |  
# MAGIC
# MAGIC
# MAGIC
# MAGIC *Derived Features*
# MAGIC
# MAGIC Below are the final feature we engineered for use in our models
# MAGIC
# MAGIC | Feature | Description | 
# MAGIC |----------|----------|
# MAGIC |previous flight delay|length of previous flight delay for same aircraft (if takeoff was > 2 hrs before next flight)|
# MAGIC |depature region| west, south, midwest, northeast, southeast|
# MAGIC |plane delays last 24h|sum of delays for the same plane over the last 24h|
# MAGIC |airport delays in last 24h|sum of delays for the departure airport over the last 24h|
# MAGIC |page rank|measure of graph centrality for the depature airport|
# MAGIC |schuled departure & arrival hour|truncation of local depature & arrival times|
# MAGIC |days to nearest holiday|integer showing number of days to the closest national holiday, past or future (e.g. on July 5th days_to_nearest_holiday is 1) |
# MAGIC |class & recency weighting|class label proportion X log transformed date index normalized between 0.3 & 0.1 (to avoid dropping any rows by assigning 0 weight) |
# MAGIC
# MAGIC These features we're excluded from our lasso feature selection discussed in future sections with the exception of region & hour since these are simply reductions in granularity from the existing dataset. The class & recency weighting is also unique in that is was not part of the training data. Instead it was used as input to the weightCol parameter of our MLlib models.

# COMMAND ----------

# DBTITLE 1,Feature Selection
# MAGIC %md *LASSO Regularization*
# MAGIC
# MAGIC After experimenting with different feature selection methods in the previous phases, in this final phase we decided to perform LASSO regularization and retain any features that were not zeroed out by at least one fold. We started with 70 features and ended up with 43 (including derived features) after LASSO. 

# COMMAND ----------

# DBTITLE 1,Modeling
# MAGIC %md
# MAGIC
# MAGIC Please use the following link to see our full Phase III - Modeling code (https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1962809137470270)

# COMMAND ----------

# MAGIC %md *Logistic Regression*
# MAGIC
# MAGIC We used logistic regression for our initial models as it's a simple way to model classification problems and is easy to interpret. 
# MAGIC
# MAGIC Our very first logistic regression was run in Phase II and contained our original set of 50 cleaned features without any derived features. That was followed in Phase III by another logistic containing a newer set of 70 cleaned and derived features. Our final logistic contained the 38 features retained by LASSO and 5 derived features.

# COMMAND ----------

# MAGIC %md *Random Forest*
# MAGIC
# MAGIC The Random Forest was only run on the LASSO + derived features outlined above. Before running on all folds, we tested a series of hyperparameters on a subset containing only the first train/val fold. These hyperparameters included the feature subset strategy, maximum depth of the tree, and the number of trees. The default hyperparameters typically had better or about the same performance on the validation fold as others that were tested. When performance was similar, we chose the option that would be less computationally expensive. Runtimes were pretty consistent across experiments. In some instances, we averaged results across multiple seeds. The hyperparameters chosen by experimentation and used in the final RF model are listed below.
# MAGIC <br><br>
# MAGIC - numTree = 10
# MAGIC - featureSubsetStrategy = auto
# MAGIC - maxDepth = 5
# MAGIC

# COMMAND ----------

# MAGIC %md *Gradient Boosted Decision Trees*
# MAGIC
# MAGIC A very similar approach to the Random Forest hyperparameter selection was used for GBDT. Again, only the LASSO + derived features were used on a subset of the folds. Tested hyperparameters included the feature subset strategy, step size (a.k.a learning rate), and maximum number of iterations. Again, the default settings appeared to prevail and had similar runtimes to other options. The hyperparameters chosen by experimentation and used in the final GBDT model are listed below.
# MAGIC <br><br>
# MAGIC - maxIter = 10
# MAGIC - stepSize = 0.1
# MAGIC - featureSubsetStrategy = all (which in experiments appeared to be the same as auto)
# MAGIC - maxDepth = 5

# COMMAND ----------

# MAGIC %md *Multilayer Perceptron ( MLP-43 - Sigmoid - 2 Softmax )*
# MAGIC
# MAGIC MLP is a deep learning model which was applied on the lasso and derived features on all the folds. Multiple experiments were conducted on the vector representation of the features that were selected by changing the number of hidden layers, maxIter, blockSize. F1, Auc, were captured for each of the runs while the number of nodes in the last layer were chosen as 2 for binary classification. All the runs came with similar outpurts for these parameters.

# COMMAND ----------

# MAGIC %md *Long Short-Term Memory (LSTM-43-Sigmoid-1tanh)*
# MAGIC
# MAGIC LSTM is another deep learning model that was applied on the lasso folds as well as the final validation data. Similar to MLP, multiple experiments were conducted by changing the number of layers, epochs, loss functions, optimizer, learning rate. The output layer chosen here is 1 as this is a binary classification. Finally the epochs were restricted to 10 as the loss converged to a minimum at 10th epoch. F1 and AUC scores were collected where the AUC score was close but less than the ones from MLP while F1 was half of the MLP one. As the model was taking too long to run on all the 5 folds, only 10% of that data was chosen to be run with LSTM. Given good resources, we would have been able to run the LSTM on entirety of dataset. If given more time, we would go in this path of experimentation using LSTM or other DL models.
# MAGIC
# MAGIC Time taking to run 10 epochs with BCELoss and Adam optimizer with just 10% of data, learning rate of 0.001 was around 2.25hrs. With changing minimal parameters, with 10% of data we were getting the scores less than other models. Hence we stopped pursuing this route.

# COMMAND ----------

# MAGIC %md Runtime Table
# MAGIC
# MAGIC | Job | Runtime |
# MAGIC |----------|----------|
# MAGIC | Post-Cleaning Checkpoint | 216 minutes |
# MAGIC | Create Folds | 18 minutes |
# MAGIC | Initial Logistic Model w/o Derived Features | 53 minutes |
# MAGIC | Updated Logistic Model w Derived Features  | 27 minutes |
# MAGIC | LASSO Regularization  | 31 minutes |
# MAGIC | Logistic Model w LASSO and Derived Features  | 24 minutes |
# MAGIC | Grid Search | 69 minutes |
# MAGIC | Random Forest | 29 minutes |
# MAGIC | GBDT | 30 minutes |
# MAGIC | MLP 1 Hidden Layer | 45 minutes |
# MAGIC | MLP 2 Hidden Layers | 40 minutes |
# MAGIC | LSTM | 136 minutes |
# MAGIC | Final Model (MLP1) | 49 minutes |

# COMMAND ----------

# DBTITLE 1,Evaluation
# MAGIC %md
# MAGIC
# MAGIC The results of the model experimentation can be seen in the table and graph below. We evaluated their performance on the large validation set (blue bars). We averaged results across folds to get the training scores (orange bars). There is a **35%** increase in validation F1 score from our best MLP1 model compared to our Initial Logistic.
# MAGIC
# MAGIC It's worth nothing that the validate score is noticeably less than the train score for the second logistic regression model. However, after LASSO was performed, the validate scores jumped right back up to meet the train scores. After LASSO, the train/validate scores weren’t too far off from one another so we don’t think overfitting occurred.
# MAGIC
# MAGIC The MLP and GBDT models performed similarly on the validation cut. That said, MLP with 1 hidden layer performed best (last row in the table), so we used it as our final model.
# MAGIC
# MAGIC | Model | Train F1 | Val F1 |
# MAGIC |----------|----------|----------|
# MAGIC | Initial Logistic | 0.648 | 0.604 |
# MAGIC | Secondary Logistic | 0.741 | 0.422 |
# MAGIC | LASSO Logistic | 0.740 | 0.743 |
# MAGIC | MLP 2 Hidden Layers | 0.792 | 0.801 |
# MAGIC | Random Forest | 0.755 | 0.756 |
# MAGIC | Gradient Boosted Decision Trees | 0.783 | 0.791 |
# MAGIC | ***MLP 1 Hidden Layer*** | ***0.808*** | ***0.814*** |

# COMMAND ----------

plt.figure(figsize=(20, 20))
image = plt.imread('Images/Experiment Models.png')
plt.imshow(image)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The final MLP model was fit on a larger training cut without folds (2015-2018) and evaluated using the test cut (2019), which had remained unseen up to this point.
# MAGIC
# MAGIC The F1 score for the training data was 0.805 and **0.803** for the test data.

# COMMAND ----------

# DBTITLE 1,Gap Analysis
# MAGIC %md
# MAGIC
# MAGIC Our data join captured flight and weather data from 2015-2021. Out of personal curiosity we took our final model and used 2019-2021 data as our holdout test set (instead of 2019 only).  Our model had an F1 Score of 0.656, a notable reduction in performance from our 2019 only F1-score. This is not entirely surprising given the severe impact of Covid-19 on airline traffic patterns, which occured primarily during that time frame. It gave the team a healthy appreciation for data shift that occurs naturally over time. It clearly illustrated the importance of regularly evaluating and retraining models to maintain desired performance.    
# MAGIC
# MAGIC A few other gaps our team indentified relate to pre-processing and feature engineering. We used standard scaler to rescale our data which may have skewed our distributions for non-normal features. In future work, we'd like to leverage min max scaling instead for non-normal features. Next we identified a small area of data leakage relating to page rank. We calculated page rank on the entire dataset instead of using past data. We don't believe this had a significant affect on performance, but when put into production, our model will need to leverage existing page ranks and could lead to slight performance changes.
# MAGIC
# MAGIC One area our model performed quite well is striking a balance between recall and precision. As discussed, we achieved a high F1 score. However, this could be due to high precision and low recall, or vice versa. In reality, our model achieved a 0.806 precision and 0.831 recall. Our goal at the outset was to achieve balance so we can warn customers about as many delays as possible, but without disrupting schedules unnecessarily. 

# COMMAND ----------

# DBTITLE 1,Conclusion
# MAGIC %md
# MAGIC
# MAGIC Our original hypothesis was that thoughtful data engineering and feature selection would enable us to accurately forecast flight delays greater than 15 minutes, 2 hours ahead of time. After many rounds of exploration and experimentation, our final results indicate that we have a useful model for prediction. This product will help both airlines and customers make informed decisions and enhance operational efficiency.
# MAGIC
# MAGIC Throughout the phases of this project, we performed iterative EDA, data cleaning, feature engineering, feature selection, and model experimentation. We used common practice time series analysis to identify 43 variables that were useful for prediction. Our efforts resulted in a final model that gave us a large improvement over our baseline model and is balanced in its predictions.
# MAGIC
# MAGIC If we were to continue on with this project, we'd conduct additional hyperparameter tuning, look into creating more derived features, experiment with ensemble modeling methods, and try additional deep learning techniques.