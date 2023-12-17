# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # AeroProphet

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Team

# COMMAND ----------

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
# MAGIC ## Abstract
# MAGIC
# MAGIC Phase 1 of "AeroProphet" involved project management planning, setting up databricks and blob storage, defining our outcome, identifying appropriate algorithms, and beginning to think through our data split and checkpointing process. We also did some initial EDA. Each individual on the team picked up the joined df_OTPW_3M dataset to become better informed for later analysis. We checked how the data is balanced and used visuals to explore missing data and outliers. We then identified the initial evaluation metrics and target variables for prediction. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase Leader Plan
# MAGIC
# MAGIC | Phase | Leader | Overarching Tasks |
# MAGIC |----------|----------|----------|
# MAGIC | 1 | Hailee | Project Plan, describe datasets, joins, tasks, and metrics |
# MAGIC | 2 | Sreeram | EDA, baseline pipeline on all available data, Scalability, Efficiency, Distributed/parallel Training, Scoring Pipeline, Feature engineering and hyperparameter tuning |
# MAGIC | 3 | Nick | Select the optimal algorithm, fine-tune and submit a final report  |
# MAGIC | 4 | Landon | Final presentation |

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Assignment/Credit Plan
# MAGIC   
# MAGIC This table will be updated to include a phase's assigments at the beginning of that phase.
# MAGIC
# MAGIC | Phase | Task | Person | Expected Time |
# MAGIC |----------|----------|----------|----------|
# MAGIC | 1 | Project Title | Landon | 10 min |
# MAGIC | 1 | Team | Hailee | 10 min |
# MAGIC | 1 | Abstract | Sreeram | 1 hr |
# MAGIC | 1 | Phase Leader Plan | Hailee | 5 min |
# MAGIC | 1 | Assignment Plan | Hailee | 10 min |
# MAGIC | 1 | Gantt Diagram | Nick | 45 min |
# MAGIC | 1 | Outcome Definition | Nick | 45 min |
# MAGIC | 1 | Data Description | Sreeram | 2 hrs |
# MAGIC | 1 | Data Description - Visuals | Hailee | 3 hrs |
# MAGIC | 1 | ML Pipelines | Hailee | 30 min |
# MAGIC | 1 | ML Algorithms & Metrics | Landon | 2 hrs |
# MAGIC | 1 | Plan for Data Split | Landon | 45 min |
# MAGIC | 1 | Conclusion | Hailee | 30 min |
# MAGIC | 1 | Final Review, Cleanup, and Submission | Hailee | 1 hour |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Machine Learning Pipelines & Gantt Diagram
# MAGIC
# MAGIC For this project, we plan to use parquet files for consistent data consumption and efficiency. Parquet files are much faster to read, write, and query than CSVs. They're also amenable to parallelization, which will be essential to process the large amount of data for this project. We plan to partition by date or week. 
# MAGIC
# MAGIC We will checkpoint the data at crucial points during the project, such as after the data engineering/cleaning process and data splits in Phase II. Having the most up-to-date dataframe will ensure smoother performance and that all team members are on the same page.
# MAGIC

# COMMAND ----------

plt.figure(figsize=(20, 20))
image = plt.imread('Images/project_plan_gantt_chart.png')
plt.imshow(image)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Outcome Definition
# MAGIC
# MAGIC For this project, we are dealing with a binary classification problem since our goal is to categorize each flight as "delayed" or "not delayed". Specifically, the goal is to predict arrival delays greater than 15 minutes, 2 hours before takeoff. With classification tasks, accuracy is a poor metric since always predicting the majority class can lead to high accuracy (especially in unbalanced datasets). As a result, we plan to focus on other evaluation metrics: *recall*, *precision*, *AUC*, and *F1 Score*. 
# MAGIC
# MAGIC **Recall**: the ratio of true positives out of all positives (how many of the positive cases are we able to identify?).
# MAGIC $$\frac{\text{True Positive}}{\text{True Positive} + \text{False Negative}}$$
# MAGIC **Precision**: the ratio of true positives out of all positive predictions (when we predict a positive, are we confident it is correct?).
# MAGIC $$\frac{\text{True Positive}}{\text{True Positive} + \text{False Positive}}$$
# MAGIC
# MAGIC **AUC**: There is a tradeoff between recall and precision as controlled by a threshold value. Typically, if the probabliy of the positive class is greater than 0.5 we predict a positive and negative otherwise. As the threshold is increased, we become more conservative and precision will increase at the expense of recall. As the threshold is lowered, the inverse occurs. We can visualize model performance at all threshold values by plotting an ROC curve (receiver operating characteristic curve). Finding the area under this curve, called the AUC, summarizes the total performance of the model for all threshold values.
# MAGIC
# MAGIC **F1 Score**: To get an idea of performance at a specific threshold value, we can leverage the F1 score. This is the harmonic mean between recall & precision.
# MAGIC $$2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Description
# MAGIC
# MAGIC The visual EDA was conducted using the df_OTPW_3M dataframe, which has joined flight and weather data from Q1 of 2015. There are 14 carriers in this dataframe. The bullets below outline our initial observations.
# MAGIC
# MAGIC - We observe close to 4 million records and 216 columns in this subset. 
# MAGIC - We are initially interested in the date, carrier, flight number, tail number, origin, destination, departure time, arrival time, departure delay (new), arrival delay (new), carrier delay, weather delay, cancelled, wind speed, visibility, humidity, pressure, visibility, longitude, and latitude variables.
# MAGIC - Many columns that have null values for entire data set can be dropped. 
# MAGIC - A cancelled flight will have null values for the time and all the delay columns. 
# MAGIC - If there is no delay then all the delay columns will be null whereas if there is a delay then the appropriate reasoning columns will be populated with how much delay had occured.
# MAGIC - EDA on data needs to be done by dropping unwanted features and normalizing to bring them together on a scale.  
# MAGIC - Total number of departure outliers: 32016 with max departure outlier value: 1988 and min departure outlier value: 112
# MAGIC - Total number of arrival outliers: 31131 with max arrival outlier value: 1971 and min arrival outlier value: 116
# MAGIC - We observed negative values indicating the respective airlies completed the journey without any delays.
# MAGIC - The DIV variables are all null and they were dropped from the OTPW dataset as they bring nothing to the table.
# MAGIC
# MAGIC To see all the Phase I EDA code, please go to "261 FP - Phase I EDA Code & Visuals" at https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1084138578716137

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## EDA Visuals

# COMMAND ----------

# MAGIC %md Discussion
# MAGIC
# MAGIC The visual EDA was conducted using the df_OTPW_3M dataframe. The bullets below outline our initial observations and speculations from these visuals.
# MAGIC
# MAGIC - Arrival delays (our outcome of interest) are most frequently 0 minutes across all carriers. Almost all delays are under 150 minutes, but many outliers extend to 1500 minutes (25 hours) and beyond.
# MAGIC - Tail number contains granular information about the airplane itself and is globally unique. With date and time it identifies the trip info, carrier, etc. The boxplot against arrival delays shows more varied distributions, indicating it may be a more useful variable than carrier.
# MAGIC - Distance traveled is very right-skewed. It may be worth exploring whether delays are associated with flights closer to the median traveled distance or those at the extreme.
# MAGIC - Latitude and Longitude may be able to capture biases related to geography. For example, if there's a fixed effect related to flights departing from the Midwest.
# MAGIC - Excluding the outliers, delay categories have similar distributions. As a result, this variable may not be able to provide distinguising information.
# MAGIC - Cancellations vary by carrier. It's possible that carriers who cancel more frequently also have longer delays, which may be worth exploring.
# MAGIC - There's clear weekly seasonality in the "Flights per Date" time series. This indicates a relationship between day of week and number of flights.
# MAGIC - Departures by hour peaks between 6am and 7pm. As a next step, it may be worth exploring whether delays spike during this timeframe as well, or whether they accumulate throughout the day.
# MAGIC - There's definitely a relationship between delays and weather conditions. A next step will be to determine how to tease out those conditions (i.e. separate individual conditions), or whether combinations of conditions are more indicative of delays. That said, the DailyWeather variable is virtually empty in our subset dataframe. We'll likely have to explore other options.
# MAGIC - Environmental conditions like wind speed, visibility, and pressure are all skewed. Extremes could be correlated with delay times. Humidity is more uniform.
# MAGIC - The final visual here shows the proportion of  missing values for some of our initial variables of interest. Overall the proportions are pretty high, generally over 60%. We'll have to have a high threshold for missing values in our analysis. More exploration will be necessary to see which of the non-null values are viable, as initial EDA showed many nonsensical string values.
# MAGIC
# MAGIC To see all the Phase I EDA visuals and code, please go to "261 FP - Phase I EDA Code & Visuals" at https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/1084138578716137

# COMMAND ----------

# MAGIC %md Visuals
# MAGIC
# MAGIC We've chosen to only display a few important visuals below.

# COMMAND ----------

# Display important visuals

plt.figure(figsize=(10, 10))
vis1 = plt.imread('Images/Arrival Delays by Tail Num.png')
plt.imshow(vis1)

plt.figure(figsize=(10, 10))
vis2 = plt.imread('Images/Cancellations by Carrier.png')
plt.imshow(vis2)

plt.figure(figsize=(10, 10))
vis3 = plt.imread('Images/Departures by Hour.png')
plt.imshow(vis3)

plt.figure(figsize=(10, 10))
vis4 = plt.imread('Images/Delay Time by Weather Condition(s).png')
plt.imshow(vis4)

plt.figure(figsize=(10, 10))
vis5 = plt.imread('Images/Missing Values.png')
plt.imshow(vis5)

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Machine Learning Algorithms & Metrics
# MAGIC We plan on exploring both classification and regression machine learning algorithms to effectively classify if a plane will be delayed by over 15 minutes 2 hours in advance. We choose to include regression models since we are able to transform the predicted output of minutes delayed into a classification of "on time" or "delayed". We plan to use a logistic regression model and linear regression model to establish baseline metrics. We will then explore the effectiveness of the following models, each of which is either ideally suited for interpretability or effectively capturing complex relationships, and has been used successfully by others (see references) to effectively predict flight delays:
# MAGIC
# MAGIC - Classification
# MAGIC     - Logistic Regression
# MAGIC     - Neural Network
# MAGIC     - Random Forest
# MAGIC - Regression
# MAGIC     - Linear Regression
# MAGIC     - Neural Network
# MAGIC     - Random Forest
# MAGIC     - Markov Jump Linear System
# MAGIC
# MAGIC Once we have identified which models appear best suited to the data, we will explore ensembling models to improve classification metrics.
# MAGIC
# MAGIC
# MAGIC References:
# MAGIC
# MAGIC - https://cs229.stanford.edu/proj2017/final-reports/5243248.pdf # This seems like it has a data leakage issue
# MAGIC - https://medium.com/analytics-vidhya/using-machine-learning-to-predict-flight-delays-e8a50b0bb64c
# MAGIC - https://www.mit.edu/~hamsa/pubs/GopalakrishnanBalakrishnanATM2017.pdf # this seems the most legit out of the three

# COMMAND ----------

# MAGIC %md
# MAGIC ## Plan for Data Split
# MAGIC Since our data is highly temporal in nature, we will use traditional time series methods for train/test split and cross validation. This will involve preventing data leakage by never using data in the future to make predictions about the past. 
# MAGIC
# MAGIC Test data:
# MAGIC -  We will holdout all data from 2019 and later to be used as our final model evaluation. 
# MAGIC - We will explore the impact of augmenting our data by over-sampling and under-sampling since it is highly likely our data will be imbalanaced due to most flights not being delayed.  
# MAGIC
# MAGIC Training data/ Validation data:
# MAGIC - Training data will be all data up to but not including 2019.
# MAGIC - From this data, we will create 5 folds for validation and training, with each sequentual fold incorporating the prior validation dataset as illustrated in the figure below:
# MAGIC
# MAGIC
# MAGIC https://miro.medium.com/v2/resize:fit:1400/1*IRvpd84up9PU99697Qlpmw.png

# COMMAND ----------

# Create a figure 
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

# Image 
image = plt.imread('Images/timeseriesCV.png')
axes.imshow(image)
axes.set_xticks([])
axes.set_yticks([])

# Adjust spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In Phase I, we set the stage for the rest of this project. We took care of project management and assignments, identified and explored our initial variables of interest, and started to think through our evaluation metrics, data pipeline, and engineering.
# MAGIC
# MAGIC In Phase II, we will continue to identify possible features by further exploring the relationships outlined above and assessing variable viability. This feature selection will depend on the number of non-null/viable values for our variables of interest. We've already seen that some of our initial variables may not be viable due to null values, such as daily weather. As such, we may want to expand our feature list. We'll then plot these variables directly against arrival delays to inspect correlations. 
# MAGIC
# MAGIC Other featuring engineering tasks will include converting many of the string variables to their appropriate float, integer, and datetime types. We'll then balance the data and normalize features. Ultimately, we'd like to end up testing 10-15 valid features in our model.
# MAGIC
# MAGIC Throughout the data engineering process, we plan to regularly checkpoint our data. We'll checkpoint again once we split our data into train, validation, and test sets (to avoid data leakage). Once we have a finalized dataframes, we'll run our baseline models.
# MAGIC
# MAGIC Some issues we're currently running into have to do with runtimes in the databricks shared resource and we'll need to explore more efficent ways to ingest the data. It also took some finagling to get our blob storage set up, but it should be resolved now.