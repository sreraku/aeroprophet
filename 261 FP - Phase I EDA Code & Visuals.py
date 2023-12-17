# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Setup

# COMMAND ----------

# Imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql.functions import col,isnan, when, count, col, split, trim, lit, avg, sum
import seaborn as sns

# COMMAND ----------

# Connect to storage

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

# see what's in the blob storage root folder 
# display(dbutils.fs.ls(f"{team_blob_url}"))
# display(dbutils.fs.ls(f"{mids261_mount_path}/OTPW_3M"))

# COMMAND ----------

#example to load dfs
# df_airlines = spark.read.parquet(f"{mids261_mount_path}/datasets_final_project_2022/parquet_airlines_data_3m/").cache()

# Load the Jan 1st, 2015 for Weather
# df_weather =  spark.read.parquet(f"{mids261_mount_path}/datasets_final_project_2022/parquet_weather_data_3m/").filter(col('DATE') < "2015-01-02T00:00:00000").cache()
# display(df_weather)

# COMMAND ----------

#Load OTPW dataset

df_OTPW_3M = spark.read.option("header", True) \
                        .option("compression", "gzip") \
                        .option("delimiter",",") \
                        .csv(f"{mids261_mount_path}/OTPW_3M").cache()

# COMMAND ----------

# The following can write the dataframe to the team's Cloud Storage  (for checkpointing)
# Navigate back to your Storage account in https://portal.azure.com, to inspect the partitions/files.
# df.write("overwrite").parquet(f"{team_blob_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## General EDA

# COMMAND ----------

display(df_OTPW_3M)

# COMMAND ----------

print(f' num rows in OTPW table: {df_OTPW_3M.count()}')
print(f' number of OTPW columns: {len(df_OTPW_3M.columns)}')

#drop any duplicates
df_OTPW_3M = df_OTPW_3M.dropDuplicates()

#check nulls in the dataframe
from pyspark.sql.functions import col,isnan, when, count, mean, stddev
val = df_OTPW_3M.select([count(when(isnan(c) | col(c).isNull() | (col(c)=='') | col(c).contains('None') | col(c).contains('NULL'), c)).alias(c) for c in df_OTPW_3M.columns]).toPandas()

# COMMAND ----------

#convert the interested columns to int for further calculations
df_new = df_OTPW_3M.withColumn("DEP_DELAY",df_OTPW_3M.DEP_DELAY.cast('int'))
df_new = df_new.withColumn("ARR_DELAY",df_new.ARR_DELAY.cast('int'))

#pick only +ve delays
df_new = df_new.where(df_new.DEP_DELAY > 0)
df_new = df_new.where(df_new.ARR_DELAY > 0)

# COMMAND ----------

#Lets use Interquantile range (IQR) to find outliers
def calc_quantiles(df, column):
    q1,q3 = df_new.approxQuantile(column, [0.25, 0.75], 0)
    print(f'q1: {q1} q3: {q3}')
    iqr = q3-q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr    

    outliers = df.filter((col(column) < lower_limit) | (col(column) > upper_limit))
    return outliers

# COMMAND ----------

d_outliers = calc_quantiles(df_new, "DEP_DELAY")
a_outliers = calc_quantiles(df_new, "ARR_DELAY")

# Calculate total outliers, maximum, and minimum outlier values for departure delay
d_total_outliers = d_outliers.count()
d_max_outlier = d_outliers.selectExpr(f'max({"DEP_DELAY"}) as max_outlier').collect()[0]["max_outlier"]
d_min_outlier = d_outliers.selectExpr(f'min({"DEP_DELAY"}) as min_outlier').collect()[0]["min_outlier"]


print(f'number of dep outliers: {d_total_outliers}')
print(f'max dep outlier value: {d_max_outlier}')
print(f'min dep outlier value: {d_min_outlier}')

# Calculate total outliers, maximum, and minimum outlier values for arrival delay
a_total_outliers = a_outliers.count()
a_max_outlier = a_outliers.selectExpr(f'max({"ARR_DELAY"}) as max_outlier').collect()[0]["max_outlier"]
a_min_outlier = a_outliers.selectExpr(f'min({"ARR_DELAY"}) as min_outlier').collect()[0]["min_outlier"]


print(f'number of arr outliers: {a_total_outliers}')
print(f'max arr outlier value: {a_max_outlier}')
print(f'min arr outlier value: {a_min_outlier}')


# COMMAND ----------

d_filtered = d_outliers.filter((d_outliers.DEP_DELAY ==  1988) | (d_outliers.DEP_DELAY ==  112 ))
display(d_filtered)
a_filtered = a_outliers.filter((a_outliers.ARR_DELAY ==  1971) | (a_outliers.ARR_DELAY ==  116 ))
display(a_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Visuals

# COMMAND ----------

# NUMBER OF FLIGHTS BY CARRIER
# There are a lot of 'OP_UNIQUE_CARRIER' values with windings (e.g. 'Ӛ��'). I dropped those for this, but worth exploring

# Create clean pandas df
carrier_value_counts = df_OTPW_3M.groupBy('OP_UNIQUE_CARRIER').count()
carrier_value_counts_pd = carrier_value_counts.toPandas()
carrier_value_counts_pd = carrier_value_counts_pd.dropna(axis=0, how='any')
carrier_value_counts_pd = carrier_value_counts_pd[carrier_value_counts_pd['OP_UNIQUE_CARRIER'].str.match(r'^[A-Za-z0-9]+$')]
carrier_value_counts_pd = carrier_value_counts_pd[carrier_value_counts_pd['count'] > 10]

# Create bar plot
plt.figure(figsize=(12, 6))  
sns.barplot(data=carrier_value_counts_pd, x='OP_UNIQUE_CARRIER', y='count')
plt.title('Flight Counts by Carrier')
plt.xlabel('Carrier')
plt.ylabel('Count')
plt.show()

# COMMAND ----------

# BOXPLOT OF DEPARTURE DELAYS BY AIRLINE
# Nulls do not mean the same thing as 0
# 'DEP_DELAY_NEW' is like 'DEP_DELAY' but converts negative values (i.e. early departures?) to 0

# Create clean pandas df 
# Confirmed no weird carrier names
# Removed 0s otherwise boxplot hard to visualize
carrier_delays = df_OTPW_3M.select('OP_UNIQUE_CARRIER', 'DEP_DELAY_NEW')
carrier_delays_pd = carrier_delays.toPandas()
carrier_delays_pd = carrier_delays_pd.dropna(axis=0, how='any')
carrier_delays_pd['DEP_DELAY_NEW'] = carrier_delays_pd['DEP_DELAY_NEW'].astype(float)
carrier_delays_pd = carrier_delays_pd[carrier_delays_pd['DEP_DELAY_NEW'] > 0] 

# Create boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=carrier_delays_pd, x='OP_UNIQUE_CARRIER', y='DEP_DELAY_NEW')
plt.title('Departure Delays by Carrier')
plt.xlabel('Carrier')
plt.ylabel('DEP_DELAY_NEW')
plt.show()

# COMMAND ----------

# BOXPLOT OF ARRIVAL DELAYS BY TAIL NUMBER (TOP 15)

# Create clean pandas df 
# Removed 0s otherwise boxplot hard to visualize
tail_delays = df_OTPW_3M.select('TAIL_NUM', 'ARR_DELAY_NEW')
tail_delays_pd = tail_delays.toPandas()
tail_delays_pd = tail_delays_pd.dropna(axis=0, how='any')
tail_delays_pd['ARR_DELAY_NEW'] = tail_delays_pd['ARR_DELAY_NEW'].astype(float)
tail_delays_pd = tail_delays_pd[tail_delays_pd['ARR_DELAY_NEW'] > 0] 

# Subset to top 15 tail numbers by those most present in data
top_tails = tail_delays_pd['TAIL_NUM'].value_counts().index[:15] 
filtered_tail_delays_pd = tail_delays_pd[tail_delays_pd['TAIL_NUM'].isin(top_tails)]

# Create boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered_tail_delays_pd, x='TAIL_NUM', y='ARR_DELAY_NEW')
plt.title('Arrival Delays by Tail Number (Top 15 by Tail Num Value Counts)')
plt.xlabel('Tail Number')
plt.ylabel('Arrival Delay in Minutes')
plt.show()

# COMMAND ----------

# BOXPLOT OF ARRIVAL DELAYS BY AIRLINE

# Create clean pandas df 
# Removed 0s otherwise boxplot hard to visualize
carrier_arr_delays = df_OTPW_3M.select('OP_UNIQUE_CARRIER', 'ARR_DELAY_NEW')
carrier_arr_delays_pd = carrier_arr_delays.toPandas()
carrier_arr_delays_pd = carrier_arr_delays_pd.dropna(axis=0, how='any')
carrier_arr_delays_pd['ARR_DELAY_NEW'] = carrier_arr_delays_pd['ARR_DELAY_NEW'].astype(float)
carrier_arr_delays_pd = carrier_arr_delays_pd[carrier_arr_delays_pd['ARR_DELAY_NEW'] > 0] 

# Create boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=carrier_arr_delays_pd, x='OP_UNIQUE_CARRIER', y='ARR_DELAY_NEW')
plt.title('Arrival Delays by Carrier')
plt.xlabel('Carrier')
plt.ylabel('Arrival Delay in Minutes')
plt.show()

# COMMAND ----------

# HISTOGRAM OF DISTANCES
# Create clean pandas df 
distances = df_OTPW_3M.select('DISTANCE')
distances_pd = distances.toPandas()
distances_pd = distances_pd.dropna(axis=0, how='any')
distances_pd['DISTANCE'] = distances_pd['DISTANCE'].astype(float)

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(distances_pd['DISTANCE'], bins=20, color='blue', edgecolor='black')
plt.title('Flight Distances')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# SCATTERPLOT OF LAT VS LONG
# Create clean pandas df 
lat_long = df_OTPW_3M.select('LATITUDE', 'LONGITUDE')
lat_long_pd = lat_long.toPandas()
lat_long_pd = lat_long_pd.dropna(axis=0, how='any')
lat_long_pd['LATITUDE'] = lat_long_pd['LATITUDE'].astype(float)
lat_long_pd['LONGITUDE'] = lat_long_pd['LONGITUDE'].astype(float)

# Create a scatterplot
plt.scatter(lat_long_pd['LONGITUDE'], lat_long_pd['LATITUDE'])
plt.xlabel('LONGITUDE')
plt.ylabel('LATITUDE')
plt.title('LATITUDE vs LONGITUDE')
plt.grid(True)

# COMMAND ----------

# BOXPLOT OF DELAY VALUES 
# Create clean pandas df 
# Removed 0s otherwise boxplot hard to visualize
delay_columns = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']
delays = df_OTPW_3M.select(delay_columns)
delays_pd = delays.toPandas()
delays_pd = delays_pd.dropna(axis=0, how='any')
delays_pd[delay_columns] = delays_pd[delay_columns].astype(float)
delays_pd = delays_pd[delays_pd[delay_columns] > 0] 

# Create boxplot
plt.figure(figsize=(12, 6))
delays_pd.boxplot(column=delay_columns)
plt.title('Delay Categories')
plt.ylabel('Minutes') # DELETE confirm the values are in minutes
plt.grid(True)
plt.show()

# COMMAND ----------

# CANCELLATIONS BY CARRIER
# Create clean pandas df
cancellations = df_OTPW_3M.groupBy('OP_UNIQUE_CARRIER').agg({'CANCELLED': 'sum'})
cancellations_pd = cancellations.toPandas()
cancellations_pd = cancellations_pd.dropna(axis=0, how='any')

# Create bar plot
plt.figure(figsize=(12, 6))  
sns.barplot(data=cancellations_pd, x='OP_UNIQUE_CARRIER', y='sum(CANCELLED)')
plt.title('Cancellations by Carrier')
plt.xlabel('Carrier')
plt.ylabel('Count of Cancellations')
plt.show()

# COMMAND ----------

# LINE GRAPH OF FLIGHT DATE
# DELETE Seeing the consistent dips and spikes, might be worth exploring day of week
# Create clean pandas df 
date = df_OTPW_3M.select('FL_DATE')
date_pd = date.toPandas()
date_pd = date_pd.dropna(axis=0, how='any')
date_pd = date_pd[date_pd['FL_DATE'].str.match(r'^\d{4}-\d{2}-\d{2}$')]
date_pd = date_pd['FL_DATE'].value_counts().sort_index()
date_pd.index = pd.to_datetime(date_pd.index)

# Create Line Graph
plt.figure(figsize=(10, 6))
plt.plot(date_pd.index, date_pd.values)
plt.xlabel('Date')
plt.ylabel('Frequency')
plt.title('Flights per Date')
plt.grid(True)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()
plt.show()

# COMMAND ----------

# LINE GRAPH OF DEPARTURE TIMES (HOUR ONLY)
# Create clean pandas df 
dep_time = df_OTPW_3M.select('DEP_TIME')
dep_time_pd = dep_time.toPandas()
dep_time_pd = dep_time_pd.dropna(axis=0, how='any')
dep_time_pd['DEP_HOUR'] = dep_time_pd['DEP_TIME'].str[:-2]
dep_time_pd = dep_time_pd[dep_time_pd['DEP_HOUR'].str.match(r'^[0-9]+$')]
dep_time_pd['DEP_HOUR'] = dep_time_pd['DEP_HOUR'].astype(int)
dep_time_pd = dep_time_pd['DEP_HOUR'].value_counts().sort_index()

# Create a line graph
plt.figure(figsize=(8, 6))
plt.plot(dep_time_pd.index, dep_time_pd.values, marker='o', linestyle='-', color='b')
plt.xlabel('Departure Hour')
plt.ylabel('Frequency')
plt.title('Count of Departures by Hour')
plt.grid(True)
plt.show()

# COMMAND ----------

# LINE GRAPH OF ARRIVAL TIMES (HOUR ONLY)
# Create clean pandas df 
arr_time = df_OTPW_3M.select('ARR_TIME')
arr_time_pd = arr_time.toPandas()
arr_time_pd = arr_time_pd.dropna(axis=0, how='any')
arr_time_pd['ARR_HOUR'] = arr_time_pd['ARR_TIME'].str[:-2]
arr_time_pd = arr_time_pd[arr_time_pd['ARR_HOUR'].str.match(r'^[0-9]+$')]
arr_time_pd['ARR_HOUR'] = arr_time_pd['ARR_HOUR'].astype(int)
arr_time_pd = arr_time_pd['ARR_HOUR'].value_counts().sort_index()

# Create a line graph
plt.figure(figsize=(8, 6))
plt.plot(arr_time_pd.index, arr_time_pd.values, marker='o', linestyle='-', color='b')
plt.xlabel('Arrival Hour')
plt.ylabel('Frequency')
plt.title('Count of Arrivals by Hour')
plt.grid(True)
plt.show()

# COMMAND ----------

# HISTOGRAM OF TAIL_NUM
# Create clean pandas df 
tail_num = df_OTPW_3M.select('TAIL_NUM')
tail_num_pd = tail_num.toPandas()
tail_num_pd = tail_num_pd.dropna(axis=0, how='any')
tail_num_pd = tail_num_pd[tail_num_pd['TAIL_NUM'].str.match(r'^[0-9A-Za-z]+$')]
tail_num_pd = tail_num_pd['TAIL_NUM'].value_counts()

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(tail_num_pd, bins=range(1, tail_num_pd.max() + 2), edgecolor='k')
plt.xlabel('Value Count')
plt.ylabel('Frequency')
plt.title('Value Counts Distribution of Tail Number')
plt.grid(axis='y')
plt.show()


# COMMAND ----------

# STACKED BAR CHART FOR ORIGIN AND DESTINATION
# Create clean pandas df 
orig_dest = df_OTPW_3M.select('ORIGIN', 'DEST')
orig_dest_pd = orig_dest.toPandas()
orig_dest_pd = orig_dest_pd.dropna(axis=0, how='any')

# Get counts of top 10 by origin
top_origins = orig_dest_pd['ORIGIN'].value_counts().index[:10]
filtered_orig_dest_pd = orig_dest_pd[orig_dest_pd['ORIGIN'].isin(top_origins)]
categories = list(set(filtered_orig_dest_pd['ORIGIN'].unique()))
origin_counts = filtered_orig_dest_pd['ORIGIN'].value_counts().reindex(categories, fill_value=0)
dest_counts = filtered_orig_dest_pd['DEST'].value_counts().reindex(categories, fill_value=0)

# Create stacked bar chart
plt.figure(figsize=(10, 6))
plt.bar(categories, origin_counts, label='ORIGIN', color='purple')
plt.bar(categories, dest_counts, bottom=origin_counts, label='DEST', color='orange')
plt.xlabel('Airport')
plt.ylabel('Origin and Destination Frequencies')
plt.title('Origin and Destination Counts by Airport (Top 10 by Origin)')
plt.legend()
plt.grid(axis='y')
plt.show()

# COMMAND ----------

# DELAY TIME BY WEATHER CONDITIONS
# Create clean pandas df
weather_delays = df_OTPW_3M.select('DailyWeather', 'DEP_DELAY_NEW')
weather_delays_pd = weather_delays.toPandas()
weather_delays_pd = weather_delays_pd.dropna(axis=0, how='any')
weather_delays_pd['DEP_DELAY_NEW'] = weather_delays_pd['DEP_DELAY_NEW'].astype(float)
weather_delays_pd = weather_delays_pd[weather_delays_pd['DEP_DELAY_NEW'] > 0] 
grouped_weather_delays_pd = weather_delays_pd.groupby('DailyWeather')['DEP_DELAY_NEW'].sum().reset_index()
grouped_weather_delays_pd
top_10_delays_pd = grouped_weather_delays_pd.sort_values(by='DEP_DELAY_NEW', ascending=False).head(10)

# Create bar plot
# DELETE Add a legend to this for wather conditions
plt.figure(figsize=(12, 6))  
sns.barplot(data=top_10_delays_pd, x='DailyWeather', y='DEP_DELAY_NEW')
plt.title('Top 10 Total Delay Times by Weather Condition(s)')
plt.xlabel('Weather Condition(s)')
plt.ylabel('Sum of Delay Time')
plt.show()

# COMMAND ----------

# HISTOGRAM OF WIND SPEED
# Create clean pandas df 
# Dropped where HourlyWindSpeed had a letter in it (usually "s")
wind_speed = df_OTPW_3M.select('HourlyWindSpeed')
wind_speed_pd = wind_speed.toPandas()
wind_speed_pd = wind_speed_pd.dropna(axis=0, how='any')
wind_speed_pd = wind_speed_pd[wind_speed_pd['HourlyWindSpeed'].str.match(r'^[0-9]+$')]
wind_speed_pd['HourlyWindSpeed'] = wind_speed_pd['HourlyWindSpeed'].astype(float)

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(wind_speed_pd['HourlyWindSpeed'], bins=20, color='blue', edgecolor='black')
plt.title('Hourly Wind Speeds')
plt.xlabel('Hourly Wind Speed')
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

# HISTOGRAM OF VISIBILITY
# Create clean pandas df 
visibility = df_OTPW_3M.select('HourlyVisibility')
visibility_pd = visibility.toPandas()
visibility_pd = visibility_pd.dropna(axis=0, how='any')
visibility_pd = visibility_pd[visibility_pd['HourlyVisibility'].str.match(r'^\d+(\.\d+)?$')]
visibility_pd['HourlyVisibility'] = visibility_pd['HourlyVisibility'].astype(float)

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(visibility_pd['HourlyVisibility'], bins=15, color='blue', edgecolor='black')
plt.title('Hourly Visibility')
plt.xlabel('Hourly Visibility') # DELETE what is this measured in?
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

# HISTOGRAM OF PRESSURE
# Create clean pandas df 
pressure = df_OTPW_3M.select('HourlyStationPressure')
pressure_pd = pressure.toPandas()
pressure_pd = pressure_pd.dropna(axis=0, how='any')
pressure_pd = pressure_pd[pressure_pd['HourlyStationPressure'].str.match(r'^\d+(\.\d+)?$')]
pressure_pd['HourlyStationPressure'] = pressure_pd['HourlyStationPressure'].astype(float)

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(pressure_pd['HourlyStationPressure'], bins=20, color='blue', edgecolor='black')
plt.title('Hourly Station Pressure')
plt.xlabel('Pressure') # DELETE what is this measured in?
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

# HISTOGRAM OF HUMIDITY
# Create clean pandas df 
humidity = df_OTPW_3M.select('HourlyRelativeHumidity')
humidity_pd = humidity.toPandas()
humidity_pd = humidity_pd.dropna(axis=0, how='any')
humidity_pd = humidity_pd[humidity_pd['HourlyRelativeHumidity'].str.match(r'^\d+(\.\d+)?$')]
humidity_pd['HourlyRelativeHumidity'] = humidity_pd['HourlyRelativeHumidity'].astype(float)

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(humidity_pd['HourlyRelativeHumidity'], bins=20, color='blue', edgecolor='black')
plt.title('Hourly Relative Humidity')
plt.xlabel('Relative Humidity') # DELETE what is this measured in?
plt.ylabel('Frequency')
plt.show()

# COMMAND ----------

# BAR CHARTS FOR MISSING VALUES
# Variables of interest = TAIL_NUM, ARR_DELAY_NEW, OP_UNIQUE_CARRIER, DISTANCE, LATITUDE, LONGITUDE, CANCELLED, DEP_TIME, DailyWeather, ORIGIN, DEST, HourlyWindSpeed, HourlyVisibility, HourlyStationPressure, HourlyRelativeHumidity, DAY_OF_WEEK

# Create clean pandas df
variables = df_OTPW_3M.select('ARR_DELAY_NEW', 'CANCELLED', 'DailyWeather', 'DAY_OF_WEEK', 'DEP_TIME', 'DEST', 'DISTANCE', 'HourlyRelativeHumidity', 'HourlyStationPressure', 'HourlyVisibility', 'HourlyWindSpeed', 'LATITUDE', 'LONGITUDE', 'OP_UNIQUE_CARRIER', 'ORIGIN', 'TAIL_NUM')
variables_pd = variables.toPandas()

# Calculate missing proportion
missing_proportion = (variables_pd.isnull().sum() / len(variables_pd)).round(2)

# Create a bar chart
plt.figure(figsize=(12, 6))
missing_proportion.plot(kind='bar', color='skyblue')
plt.xlabel('Variables of Interest')
plt.ylabel('Proportion of Missing Values')
plt.title('Proportion of Missing Values by Variable of Interest')
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------

# Other ideas

# Histogram of arrival delays ('ARR_DELAY','ARR_DELAY_NEW',) - compare to departure delays

# COMMAND ----------

temp = df_OTPW_3M.select("FL_DATE")
temp_df = temp.toPandas()


# COMMAND ----------

temp_df['Flight_date'] = pd.to_datetime(temp_df.FL_DATE, format = '%Y-%m-%d', errors = 'coerce')

# COMMAND ----------

temp_df.describe

# COMMAND ----------

