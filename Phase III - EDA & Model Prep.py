# Databricks notebook source
# MAGIC %md
# MAGIC ### Imports

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.functions import col, isnan, when, count, split, trim, lit, avg, sum, expr
from pyspark.sql.types import IntegerType, FloatType, StringType, TimestampType
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.feature import Imputer
from pyspark.ml.stat import Correlation
from pyspark.sql.window import Window
import ast
import holidays
from dateutil.parser import parse
from datetime import datetime
from collections import namedtuple
from datetime import date, timedelta
import re

# COMMAND ----------

# MAGIC %md
# MAGIC ### Connect to Storage

# COMMAND ----------

blob_container  = "261container"        # The name of your container created in https://portal.azure.com
storage_account = "261bucket"  # The name of your Storage account created in https://portal.azure.com
secret_scope    = "261scope"           # The name of the scope created in your local computer using the Databricks CLI
secret_key      = "scopekey"             # The name of the secret key created in your local computer using the Databricks CLI
team_blob_url   = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"  #points to the root of your team storage bucket
mids261_mount_path      = "/mnt/mids-w261"

# SAS Token: Grant the team limited access to Azure Storage resources
spark.conf.set(
    f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
    dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# Print what's saved in blog storage
display(dbutils.fs.ls(f"{team_blob_url}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read in Data

# COMMAND ----------

df_60M = spark.read.parquet(f"{team_blob_url}/feature_selection_2023_12_10").cache()

# COMMAND ----------

# Cast string columns to float
df_60M = df_60M.withColumn('HourlyPrecipitation', col('HourlyPrecipitation').cast('float'))
df_60M = df_60M.withColumn('Max_HourlyPrecipitation', col('Max_HourlyPrecipitation').cast('float'))

# COMMAND ----------

df_60M = df_60M.drop('HourlyWindDirection','Max_HourlyWindDirection','Min_HourlyWindDirection')

# COMMAND ----------

df_60M = df_60M.dropDuplicates()

# COMMAND ----------

# MAGIC %md ### Correct YEAR

# COMMAND ----------

df_60M = df_60M.drop('YEAR')
df_60M = df_60M.withColumn('YEAR', F.year(col('FL_DATE')))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Handle Missing Data

# COMMAND ----------

# df_60M = df_60M.na.fill(value=0) # just filling with 0 for now because it's much faster
numerical_cols = [c for c, d_type in df_60M.dtypes if d_type in ['int','float', 'double']]

df_training = df_60M.filter(col('YEAR') < 2019)

imputer = Imputer(inputCols= numerical_cols, outputCols=numerical_cols)
imputer.setStrategy("mean")

imputer_model = imputer.fit(df_training)
df_60M = imputer_model.transform(df_60M)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Correlation analysis

# COMMAND ----------

numerical_cols = [c for c, d_type in df_60M.dtypes if d_type in ['int','float', 'double']]
cols = [c for c in numerical_cols if c not in ['DEP_DELAY','DEP_DEL15','OP_CARRIER_FL_NUM','YEAR', 'MONTH', 'DAY_OF_WEEK']]
cols = ['DEP_DELAY',*cols]

assembler = VectorAssembler(inputCols= cols, outputCol='corr_features')
df_vector = assembler.transform(df_60M).select('corr_features')

# get correlation matrix
matrix = Correlation.corr(df_vector, 'corr_features').collect()[0][0] 
corr_matrix = matrix.toArray().tolist() 
corr_matrix_df = pd.DataFrame(data=corr_matrix, columns = cols, index=cols) 
corr_matrix_df.style.background_gradient(cmap='coolwarm').set_precision(2)

plt.figure(figsize=(65,65))  
sns.heatmap(corr_matrix_df, 
            xticklabels=corr_matrix_df.columns.values,
            yticklabels=corr_matrix_df.columns.values,  cmap="Greens", annot=True)

plt.show()

# COMMAND ----------

corr_matrix_df = corr_matrix_df.fillna(0)

# COMMAND ----------

filtered_rows_df = corr_matrix_df[corr_matrix_df.apply(lambda row: any((0.95 < x < 1) for x in row), axis=1)]
df = filtered_rows_df.loc[:, filtered_rows_df.apply(lambda col: any((0.95 < x < 1) for x in col), axis=0)]

drop_lst = []
for column in df.index:
    if column[:3] == 'Max' or column[:3] == 'Min':
        drop_lst.append(column)

# df_sample = df_sample.drop(*drop_lst)
df_60M = df_60M.drop(*drop_lst)
corr_matrix_df = corr_matrix_df.drop(drop_lst)
corr_matrix_df = corr_matrix_df.drop(drop_lst, axis = 1)

# COMMAND ----------

drop_lst

# COMMAND ----------

filtered_rows_df = corr_matrix_df[corr_matrix_df.apply(lambda row: any((0.95 < x < 1) for x in row), axis=1)]
filtered_rows_df.loc[:, filtered_rows_df.apply(lambda col: any((0.95 < x < 1) for x in col), axis=0)]

# COMMAND ----------

df_60M = df_60M.drop('DISTANCE_GROUP','sch_depart_hour','Avg_HourlyDewPointTemperature','Avg_HourlyDryBulbTemperature','Avg_HourlyRelativeHumidity','Avg_HourlyAltimeterSetting','Avg_HourlyStationPressure','Avg_HourlyWetBulbTemperature')
corr_matrix_df = corr_matrix_df.drop(['DISTANCE_GROUP','sch_depart_hour','Avg_HourlyDewPointTemperature','Avg_HourlyDryBulbTemperature','Avg_HourlyRelativeHumidity','Avg_HourlyAltimeterSetting','Avg_HourlyStationPressure','Avg_HourlyWetBulbTemperature'])
corr_matrix_df = corr_matrix_df.drop(['DISTANCE_GROUP','sch_depart_hour','Avg_HourlyDewPointTemperature','Avg_HourlyDryBulbTemperature','Avg_HourlyRelativeHumidity','Avg_HourlyAltimeterSetting','Avg_HourlyStationPressure','Avg_HourlyWetBulbTemperature'], axis = 1)

# COMMAND ----------


corr_df = pd.DataFrame(list(corr_matrix_df.DEP_DELAY.items()),columns=['Column', 'Correlation'])
corr_df = corr_df[corr_df.Column != 'DEP_DELAY']

plt.figure(figsize=(20, 6))
sns.barplot(x='Column', y='Correlation', data=corr_df)
plt.title('Correlation with DEP_DELAY')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

low_corr_features = corr_df[abs(corr_df.Correlation) < 0.01].Column.tolist()

# List of features to remove
features_to_remove = ['DAY_OF_MONTH', 'dest_airport_lat', 'origin_airport_lat', 'days_to_nearest_holiday']

# Removing the specified features
low_corr_features = [feature for feature in low_corr_features if feature not in features_to_remove]

low_corr_features

# COMMAND ----------

# MAGIC %md
# MAGIC Drop any features with less than 0.01 correlation

# COMMAND ----------

df_60M = df_60M.drop(*low_corr_features)

# COMMAND ----------

# MAGIC %md
# MAGIC Drop any unwanted variables

# COMMAND ----------

df_60M = df_60M.drop('OP_CARRIER_FL_NUM','ORIGIN','DEST')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Prep

# COMMAND ----------

# MAGIC %md
# MAGIC One hot encode categorical data, split into training and test, standardize

# COMMAND ----------

df_60M = df_60M.drop('DEP_DELAY')

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

dummy_source_columns = ['OP_UNIQUE_CARRIER', 'DAY_OF_WEEK', 'region', 'origin_type']

# Creating a list to store stages of the Pipeline for one-hot encoding and vector assembly
stages = []

# Process each integer column for one-hot encoding
for c in dummy_source_columns:
    # First convert integer IDs to indexed categories
    indexer = StringIndexer(inputCol=c, outputCol=c + "_indexed")
    encoder = OneHotEncoder(inputCol=c + "_indexed", outputCol=c + "_dummy") #<- this expands category index above into one hot encoding e.g. southwest becomes vector 0,1,0,0 representing midwest = 0, southweest = 1 etc
    stages += [indexer, encoder]

# Create a list of all feature columns, excluding original ID columns and the target variable
feature_columns = [c + "_dummy" for c in dummy_source_columns] + [c for c in df_60M.columns if c not in [*dummy_source_columns,*[c + '_indexed' for c in dummy_source_columns], 'DEP_DEL15', 'FL_DATE']]

# Add the VectorAssembler to the stages
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features") # collects each of the features into a single column
stages += [assembler]

# Create the pipeline
pipeline = Pipeline(stages=stages)

# Fit the pipeline to the data and transform
df_60M_transformed = pipeline.fit(df_60M).transform(df_60M)

# Split the data into training and test sets on flights prior to 2019, and 2019 and later
train_df = df_60M_transformed.filter(df_60M_transformed["YEAR"] < 2019)
test_df = df_60M_transformed.filter(df_60M_transformed["YEAR"] >= 2019)

# Create a pipeline for standardization
std_pipeline = Pipeline(stages=[
    StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
])

# Fit the standardization pipeline to the training data
std_pipeline_model = std_pipeline.fit(train_df)

# Transform both training and test data using the standardization pipeline model
train_df_transformed = std_pipeline_model.transform(train_df)
test_df_transformed = std_pipeline_model.transform(test_df)


# COMMAND ----------

# Save the transformed test data to blob storage
today_str = datetime.today().strftime('%Y_%m_%d')

test_df_transformed.write.partitionBy('FL_DATE').mode("overwrite").parquet(f"{team_blob_url}/test_transformed_{today_str}")

# COMMAND ----------

# Calculate the maximum date
max_date = train_df_transformed.agg(F.max('FL_DATE').alias('max_date')).collect()[0]['max_date']

# COMMAND ----------

# Calculate the cutoff date (100 days before the max date)
cutoff_date = max_date - F.expr('INTERVAL 100 DAYS')
val_df_transformed = train_df_transformed.filter(col('FL_DATE') > cutoff_date)
train_df_transformed = train_df_transformed.filter(col('FL_DATE') <= cutoff_date)

# COMMAND ----------

# Save the transformed validation data
today_str = datetime.today().strftime('%Y_%m_%d')
val_df_transformed.write.partitionBy('FL_DATE').mode("overwrite").parquet(f"{team_blob_url}/val_transformed_{today_str}")

# COMMAND ----------

# MAGIC %md Creating class weight col for class imblance

# COMMAND ----------

# Compute class weights
num_negatives = train_df_transformed.filter(train_df_transformed['DEP_DEL15'] == 0).count()
total = train_df_transformed.count()
weight_ratio = num_negatives / total

# Add weights
train_df_transformed = train_df_transformed.withColumn('weight', when(train_df_transformed['DEP_DEL15'] == 1, weight_ratio).otherwise(1 - weight_ratio))

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# Find the minimum date to establish a reference point
min_date = train_df_transformed.agg({"FL_DATE": "min"}).collect()[0][0]
train_df_transformed = train_df_transformed.withColumn("date_index", F.datediff(F.col("FL_DATE"), F.lit(min_date)))

# Apply logarithmic transformation
train_df_transformed = train_df_transformed.withColumn("log_date_index", F.log(F.col("date_index") + 1))  # Adding 1 to avoid log(0)

# Compute the total sum of the log_date_index column
total_log_date_index = train_df_transformed.agg(F.sum("log_date_index")).collect()[0][0]

max_log_date_index = train_df_transformed.agg(F.max("log_date_index")).collect()[0][0]
min_log_date_index = train_df_transformed.agg(F.min("log_date_index")).collect()[0][0]

# Add normalized weights
train_df_transformed = train_df_transformed.withColumn('normalized_log_weight', 
                            (F.col('log_date_index') - min_log_date_index) /
                            (max_log_date_index - min_log_date_index) * 0.7 + 0.3
                            )

train_df_transformed = train_df_transformed.withColumn('combined_weight', F.col('weight') * F.col('normalized_log_weight'))

# COMMAND ----------

train_df_transformed.display()

# COMMAND ----------

train_df_transformed.agg(F.max('FL_DATE').alias('max_date')).collect()[0]['max_date']

# COMMAND ----------

# See feature indices and what they were mapped to

pd.set_option('display.max_rows', None)

pd.DataFrame(train_df_transformed.schema["features"].metadata["ml_attr"]["attrs"]["binary"]+train_df_transformed.schema["features"].metadata["ml_attr"]["attrs"]["numeric"]).sort_values("idx")

# COMMAND ----------

# MAGIC %md
# MAGIC Create K Cross Folds

# COMMAND ----------

BlockSplit = namedtuple("BlockSplit", ["min_date", "train_cut", "val_cut"])

def make_block_splits(min_date, max_date, train_width, val_width):
    blocks = list()
    train_min = min_date
    val_cut = min_date
    
    # Loop over and create blocks
    while True:
        train_cut = train_min + train_width
        val_cut = train_cut + val_width
        if train_cut > max_date or val_cut > max_date:
            break
        blocks.append(BlockSplit(train_min, train_cut, val_cut))
        train_min = train_cut
    
    return blocks

def make_folds(df, date_col, splits):
    """
    Make folds using the specified block splits.

    Args:
        df - the dataframe to make folds from
        date_col - the name of a date column to split on
        splits - a list of ``BlockSplit`` instances
    
    Returns:
        a list of (train_df, val_df) tuples
    """
    folds = list()
    for split in splits:
        train_df = df.filter((df[date_col] >= split.min_date) & (df[date_col] < split.train_cut))
        val_df = df.filter((df[date_col] >= split.train_cut) & (df[date_col] < split.val_cut))
        folds.append((train_df, val_df))
    return folds

def save_folds_to_blob(folds, blob_url, fold_name):
    for i, (train_df, val_df) in enumerate(folds):
        dbutils.fs.rm(f"{blob_url}/{fold_name}/train_{i}_df", recurse=True)
        dbutils.fs.rm(f"{blob_url}/{fold_name}/val_{i}_df", recurse=True)
        train_df.write.parquet(f"{blob_url}/{fold_name}/train_{i}_df")
        val_df.write.parquet(f"{blob_url}/{fold_name}/val_{i}_df")


blocks = make_block_splits(date(2015,1,1),date(2018,9,22),timedelta(days = 252),timedelta(days = 100))
folds = make_folds(train_df_transformed, 'FL_DATE', blocks)

# Save the transformed test data to blob storage
today_str = datetime.today().strftime('%Y_%m_%d')

save_folds_to_blob(folds, team_blob_url, f"k_folds_cross_val_{today_str}")


# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Function to generate dates for training and validation sets
def generate_dates(start_date, num_folds, train_days, val_days):
    dates = []
    current_start = start_date

    for _ in range(num_folds):
        train_end = current_start + timedelta(days=train_days)
        val_end = train_end + timedelta(days=val_days)
        dates.append((current_start, train_end, val_end))
        current_start = train_end
    return dates

# Define the parameters
start_date = datetime(2015,1,1)  # Arbitrary start date
num_folds = 5  # Number of folds to display
train_days = 300
val_days = 142

# Generate the dates
dates = generate_dates(start_date, num_folds, train_days, val_days)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

for i, (train_start, train_end, val_end) in enumerate(dates):
    # Plot training period
    ax.plot([train_start, train_end], [i, i], color="blue", linewidth=6, label="Training" if i == 0 else "")
    # Plot validation period
    ax.plot([train_end, val_end], [i, i], color="red", linewidth=6, label="Validation" if i == 0 else "")

# Formatting the plot
ax.set_yticks(range(num_folds))
ax.set_yticklabels([f"Fold {i+1}" for i in range(num_folds)])
ax.set_xlabel("Date")
ax.set_title("Overlapping Training and Validation Folds (Gantt Chart)")

# Format date on x-axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))

# Adding legend
ax.legend(loc='upper right')

# Rotate date labels for clarity
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
