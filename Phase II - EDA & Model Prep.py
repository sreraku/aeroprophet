# Databricks notebook source
!pip install altair

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
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

df_60M = spark.read.parquet(f"{team_blob_url}/feature_selection_2023_11_27_2").cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Handle Missing Data

# COMMAND ----------

# df_60M = df_60M.na.fill(value=0) # just filling with 0 for now because it's much faster
numerical_cols = [c for c, d_type in df_60M.dtypes if d_type in ['int','float', 'double']]

imputer = Imputer(inputCols= numerical_cols, outputCols=numerical_cols)
imputer.setStrategy("mean")

imputer_model = imputer.fit(df_60M)
df_60M = imputer_model.transform(df_60M)

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA - Full Dataset

# COMMAND ----------

# MAGIC %md
# MAGIC Examine proportions of delays greater than 15 minutes

# COMMAND ----------

def dumpInfo(df_60M):
    di = {}

    df_60M_grouped = df_60M.groupBy("DEP_DEL15").agg(F.count("*").alias("Delay_Count"))

    di['totalfl_ct'] = df_60M.count()
    di['delayed_flight_ct'] = df_60M_grouped.filter(df_60M_grouped.DEP_DEL15 == 1.0).collect()[0][1]
    di['ontime_flight_ct'] = df_60M_grouped.filter(df_60M_grouped.DEP_DEL15 == 0.0).collect()[0][1]

    return di

di = dumpInfo(df_60M)

print(f'Total Flight Count: {di["totalfl_ct"]}')
print(f'Number of Flights Delayed by More Than 15 Minutes: {di["delayed_flight_ct"]}')
print(f'Delayed Flight Percentage: {(di["delayed_flight_ct"] *100) / di["totalfl_ct"] }')
print(f'On Time Flight Percentage: {(di["ontime_flight_ct"] *100) / di["totalfl_ct"]}')

# COMMAND ----------

# MAGIC %md
# MAGIC Review distribution of outcome variable

# COMMAND ----------

hist = df_60M.select("DEP_DEL15").rdd.flatMap(lambda x: x).histogram(2)

# Histogram data
bin_edges = hist[0] 
counts = hist[1] 

categories = ['Not Delayed', 'Delayed']
df = pd.DataFrame({
    'DEP_DEL15': categories,
    'Count': counts
})

# Create the plot
plt.figure(figsize=(8, 5))
sns.barplot(x='DEP_DEL15', y='Count', data=df)

plt.xlabel('DEP_DEL15')
plt.ylabel('Count')
plt.title('Distribution of Delays')

# Set y-axis to display plain numbers
plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Examine delays by airline and airport

# COMMAND ----------

# Top 10 Most popular Origin Airports
num_dep_delay = df_60M.select( 'ORIGIN','DEP_DEL15' ).groupBy('ORIGIN').agg(F.sum('DEP_DEL15')).withColumnRenamed('sum(DEP_DEL15)', 'Num_Departure_Delay') 
df = num_dep_delay.toPandas()
df1_sort = df.sort_values(by='Num_Departure_Delay', ascending=False)
origin_airport_top10 = df1_sort.head(10)

# Top 10 Airline Carrier patterns
num_dep_delay = df_60M.select( 'OP_UNIQUE_CARRIER','DEP_DEL15' ).groupBy('OP_UNIQUE_CARRIER').agg(F.sum('DEP_DEL15')).withColumnRenamed('sum(DEP_DEL15)', 'Num_Departure_Delay') 
df = num_dep_delay.toPandas()
df_sort = df.sort_values(by='Num_Departure_Delay', ascending=False)
airline_carrier = df_sort.head(10)

chart1 = alt.Chart(airline_carrier).mark_bar(color='Red').encode(
    x=alt.X('OP_UNIQUE_CARRIER:N', title='Airline Carriers', sort='-y'),
    y=alt.Y('Num_Departure_Delay:Q', title='Number of Delays'),
    tooltip=['OP_UNIQUE_CARRIER', 'Num_Departure_Delay']
).properties(
    title="Top 10 Airline Carrier ranked by Most Number of Delays",
    width=600
)

# Origin Airports
chart2 = alt.Chart(origin_airport_top10).mark_bar(color='Blue').encode(
    x=alt.X('ORIGIN:N', title='Airports', sort='-y'),
    y=alt.Y('Num_Departure_Delay:Q', title='Number of Delays'),
    tooltip=['ORIGIN', 'Num_Departure_Delay']
).properties(
    title="Top 10 Origin Airport ranked by Most Number of Delays",
    width=600
)

chart1 & chart2

# COMMAND ----------

!pip install holoviews
!pip install geoviews
!pip install geopandas

# COMMAND ----------

# Plot the top 10 flight delayed airports
import pandas as pd
import plotly.express as px




num_dep_delay = df_60M.select( 'ORIGIN','DEP_DEL15', 'origin_airport_lon', 'origin_airport_lat' ).groupBy('ORIGIN','origin_airport_lon', 'origin_airport_lat').agg(F.sum('DEP_DEL15')).withColumnRenamed('sum(DEP_DEL15)', 'Num_Departure_Delay') 
df_pd = num_dep_delay.toPandas()

df1_sort = df_pd.sort_values(by='Num_Departure_Delay', ascending=False)
origin_airport_top10 = df1_sort.head(10)

fig = px.scatter_geo(origin_airport_top10, lat = 'origin_airport_lat', lon = 'origin_airport_lon', size = 'Num_Departure_Delay', color = 'ORIGIN', text = 'ORIGIN')

# setting title for the map
fig.update_layout(title = 'Top 10 Flight delayed Airports', geo_scope = 'usa')

fig.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA - Subset

# COMMAND ----------

# MAGIC %md
# MAGIC Limit size for EDA

# COMMAND ----------

df_3M = df_60M.filter((col('YEAR') == 2015) & (col('MONTH') < 4)).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Correlation analysis

# COMMAND ----------

try:
    corr_df = pd.read_csv('pd_data/corr_df.csv')
except:
    numerical_cols = [c for c, d_type in df_60M.dtypes if d_type in ['int','float', 'double', 'bigint']]
    selected_columns = [c for c in numerical_cols if c not in ['DEP_DEL15', 'OP_CARRIER_FL_NUM','YEAR', 'MONTH', 'DAY_OF_WEEK']]
    correlations = {c: df_3M.stat.corr("DEP_DEL15", c) for c in selected_columns}
    corr_df = pd.DataFrame(list(correlations.items()), columns=['Column', 'Correlation'])

    corr_df.to_csv('pd_data/corr_df.csv')

plt.figure(figsize=(10, 6))
sns.barplot(x='Correlation', y='Column', data=corr_df)
plt.title('Correlation with DEP_DEL15')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Columns')
plt.show()

# COMMAND ----------

corr_df = corr_df.fillna(0)

# COMMAND ----------

low_corr_features = corr_df[abs(corr_df.Correlation) < 0.01].Column.tolist()
low_corr_features

# COMMAND ----------

# MAGIC %md
# MAGIC Drop any features with less than 0.01 correlation

# COMMAND ----------

df_3M = df_3M.drop(*low_corr_features)
df_60M = df_60M.drop(*low_corr_features)

# COMMAND ----------

# MAGIC %md
# MAGIC Review correlation among features

# COMMAND ----------

numerical_cols = [c for c, d_type in df_60M.dtypes if d_type in ['int','float', 'double']]
cols = [c for c in numerical_cols if c not in ['DEP_DEL15', 'OP_CARRIER_FL_NUM','YEAR', 'MONTH', 'DAY_OF_WEEK']]
cols = ['DEP_DEL15',*cols]

assembler = VectorAssembler(inputCols= cols, outputCol='corr_features')
df_vector = assembler.transform(df_3M).select('corr_features')

# get correlation matrix
matrix = Correlation.corr(df_vector, 'corr_features').collect()[0][0] 
corr_matrix = matrix.toArray().tolist() 
corr_matrix_df = pd.DataFrame(data=corr_matrix, columns = cols, index=cols) 
corr_matrix_df .style.background_gradient(cmap='coolwarm').set_precision(2)

plt.figure(figsize=(25,10))  
sns.heatmap(corr_matrix_df, 
            xticklabels=corr_matrix_df.columns.values,
            yticklabels=corr_matrix_df.columns.values,  cmap="Greens", annot=True)

plt.show()

# COMMAND ----------

filtered_rows_df = corr_matrix_df[corr_matrix_df.apply(lambda row: any((0.9 < x < 1) for x in row), axis=1)]
filtered_rows_df.loc[:, filtered_rows_df.apply(lambda col: any((0.9 < x < 1) for x in col), axis=0)]

# COMMAND ----------

# MAGIC %md
# MAGIC Drop any highly correlated features

# COMMAND ----------

df_3M = df_3M.drop('DISTANCE_GROUP','HourlyWetBulbTemperature')
df_60M = df_60M.drop('DISTANCE_GROUP','HourlyWetBulbTemperature')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Review summary statistics

# COMMAND ----------

df_3M.describe().display()

# COMMAND ----------

# MAGIC %md
# MAGIC Drop any unwanted variables

# COMMAND ----------

df_60M = df_60M.drop('OP_CARRIER_FL_NUM','ORIGIN','DEST', 'dest_type', 'origin_type')

# COMMAND ----------

numerical_cols = [c for c, d_type in df_60M.dtypes if d_type in ['int','float', 'double']]
cols = [c for c in numerical_cols if c not in ['DEP_DEL15', 'OP_UNIQUE_CARRIER','YEAR', 'MONTH', 'DAY_OF_WEEK', 'DAY_OF_MONTH']]

result_data = []
for c in cols:
    result = df_3M.selectExpr(f"mean({c})", f"percentile_approx({c}, array(0.5))", f"stddev({c})")
    values = result.collect()[0]
    mean_val = values[0] if values[0] is not None else 0
    median_val = values[1][0] if values[1] is not None and values[1][0] is not None else 0
    stddev_val = values[2] if values[2] is not None else 0
    result_data.append({"Variable": c, "Mean": mean_val, "Median": median_val, "Stddev": stddev_val})

# Create Pandas DataFrame for mean, median, and standard deviation values
data_df = pd.DataFrame(result_data)

# Plotting
plt.figure(figsize=(14, 7))

# Set the positions for the bars
x = np.arange(len(data_df))

# Set the width of the bars
width = 0.25

# Plot mean values
plt.bar(x - width, data_df['Mean'], width, alpha=0.5, label='Mean')

# Plot median values
plt.bar(x, data_df['Median'], width, alpha=0.5, label='Median')

# Plot standard deviation values
plt.bar(x + width, data_df['Stddev'], width, alpha=0.5, label='Stddev')

# Set labels, title, and ticks
plt.xlabel('Variables')
plt.ylabel('Values')
plt.title('Mean, Median, and Standard Deviation of Specified Variables')
plt.xticks(x, data_df['Variable'], rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Review pairplots

# COMMAND ----------

# #random sample 0.2% of rows (40,000,000-> 80,000)
# eda_samp = df_60M.sample(fraction = 0.00002, seed = 1)
# # Convert Spark DataFrame to Pandas DataFrame
# pandas_df = eda_samp.toPandas()
# # Create corner pairplot, with hue's based on outcomes:
# sns.pairplot(pandas_df, hue = 'DEP_DEL15',  corner = True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Prep

# COMMAND ----------

# MAGIC %md
# MAGIC One hot encode categorical data, split into training and test, standardize

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

dummy_source_columns = ['OP_UNIQUE_CARRIER', 'DAY_OF_WEEK', 'region']

# Creating a list to store stages of the Pipeline for one-hot encoding and vector assembly
stages = []

# Process each integer column for one-hot encoding
for c in dummy_source_columns:
    # First convert integer IDs to indexed categories
    indexer = StringIndexer(inputCol=c, outputCol=c + "_indexed") # need to remove this as well <<-- don't remove, this is to translate strings to ints. For example, midwest -> index 0, southwest -> index 1 ... etc
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

test_df_transformed.write.partitionBy('FL_DATE').mode("overwrite").parquet(f"{team_blob_url}/test_transformed_{today_str}_2")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Determine feature indices for later interpretation

# COMMAND ----------

# Code Source - https://stackoverflow.com/questions/42935914/how-to-map-features-from-the-output-of-a-vectorassembler-back-to-the-column-name

# Separated them out to confirm IDs
# pd.DataFrame(train_df_transformed.schema["features"].metadata["ml_attr"]["attrs"]["binary"]).sort_values("idx")
# pd.DataFrame(train_df_transformed.schema["features"].metadata["ml_attr"]["attrs"]["numeric"]).sort_values("idx")

pd.DataFrame(train_df_transformed.schema["features"].metadata["ml_attr"]["attrs"]["binary"]+train_df_transformed.schema["features"].metadata["ml_attr"]["attrs"]["numeric"]).sort_values("idx")

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


blocks = make_block_splits(date(2015,1,1),date(2018,12,31),timedelta(days = 272),timedelta(days = 100))
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
