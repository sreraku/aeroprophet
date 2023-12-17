# Databricks notebook source
# MAGIC %md
# MAGIC # Extra Credit - Data Join

# COMMAND ----------

#Imports

from pyspark.sql.functions import col, lit, date_format, to_utc_timestamp, expr, hour
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql.functions import col,isnan, when, count, col, split, trim, lit, avg, sum, expr, udf
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType
import holidays
import seaborn as sns

pd.set_option('display.max_rows', 400)
pd.set_option('display.max_columns', 40)

# COMMAND ----------

#Connecting to blob storage

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

# COMMAND ----------

#loading custom 3 month dataset from finalized data join 
# Used this after completing the join to calculate memory and table size.

try:
    df_FS_3M = spark.read.parquet(f"{team_blob_url}/df_FS_3M")
    
except:                
  print('error loading file')

# COMMAND ----------

#Checking available dataset files 
data_BASE_DIR = "dbfs:/mnt/mids-w261/datasets_final_project_2022/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Loading datasets 

# COMMAND ----------

# Airline Data    
df_flights = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/")
# Weather data
df_weather = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data/")
# Stations data      
df_stations = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/stations_data/stations_with_neighbors.parquet/")
#airport_station_name data
df_airports = spark.read.csv(f"file:///Workspace/Repos/nickjohnson@berkeley.edu/datasci_261_team_2-2/Data/airport-codes.csv", header = True, inferSchema = True )

# COMMAND ----------

display(df_flights)

# COMMAND ----------

import requests

url = "https://www.ncei.noaa.gov/access/services/data/v1"
params = {
    'dataset': 'LCD',
    'startDate': '2020-01-01',
    'endDate': '2022-12-31',

    # Define your parameters here, e.g., 'dataset': 'your-dataset-id',
    # 'startDate': 'YYYY-MM-DD', 'endDate': 'YYYY-MM-DD', ...
}

response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    data = response.jtext()  # Or response.text if data is in CSV or other format
    # Process the data as needed
else:
    print(f"Error: {response.status_code}")


# COMMAND ----------

# MAGIC %md ## Initial exploration of datasets

# COMMAND ----------

df_airports.columns

# COMMAND ----------

df_airports.dtypes

# COMMAND ----------

for i in df_weather.dtypes:
  name, data_type = i 
  print(name, data_type)

# COMMAND ----------

total_ct  = df_flights.count()
distinct_ct = df_flights.distinct().count()

print(total_ct,distinct_ct)

# COMMAND ----------

def table_size_calculator(df_list, df_names):
  table_names = df_names
  table_size = {}
  for i in range(len(df_list)):
    table_size[table_names[i]] = (f"# of rows: {df_list[i].count()}",\
       f"#of cols: {len(df_list[i].columns)}")
  print(table_size)


def mem_calculator(df):
  df.cache().select(df.columns) 
  size_in_bytes = df._jdf.queryExecution().optimizedPlan().stats().sizeInBytes()
  df.unpersist(blocking = True)
  print(round(size_in_bytes/1000000000,2))


# COMMAND ----------

# table_size_calculator([df_flights,df_stations,df_weather, df_airports], ['df_flights','df_stations','df_weather', 'df_airports'])

# COMMAND ----------

table_size_calculator([df_weather], ['df_weather'])

# COMMAND ----------

# MAGIC %md
# MAGIC Table details and final join time
# MAGIC |Table | Rows | Columns | Memory (GB) | Time to Join (HH:MM:SS) |
# MAGIC | -------- | ------- | -------- | ------- | ------- |
# MAGIC | df_flights | 74,177,433| 109| 2.93| None |
# MAGIC | df_weather | 898,983,399 | 124 |35.05 | None|
# MAGIC | df_stations | 5,004,169 | 12|1.3 | None|
# MAGIC | df_airports | 57,421| 12 |0.01 | None|
# MAGIC | df_FSW | 72,515,921| 292 |43.5  | 02:13:00|
# MAGIC
# MAGIC Cluster Details:
# MAGIC DBR 13.3 LTS ML, Spark 3.4.1, Scala 2.12,
# MAGIC Standaard_DS3_v2, 14GB, 4 Cores, Standard_DS3_v2, 84GB, 24 Cores, 1-6 workers
# MAGIC

# COMMAND ----------

df_flights = df_flights.withColumnRenamed('YEAR','year_flights')

# COMMAND ----------

df_OTPW_3M = spark.read.option("header", True) \
                        .option("compression", "gzip") \
                        .option("delimiter",",") \
                        .csv(f"{mids261_mount_path}/OTPW_3M")

# COMMAND ----------

# comopare column names in datasets to evalute overlap
flights_weather = []
flights_stations = []
weather_stations = []
for i in df_OTPW_3M.columns:
  if i in df_flights.columns and i in df_weather.columns:
    flights_weather.append(i)
  if i in df_flights.columns and i in df_stations.columns:
    flights_stations.append(i)
  if i in df_weather.columns and i in df_stations.columns:
    weather_stations.append(i)
print(flights_weather)
print(flights_stations)
print(weather_stations)

# COMMAND ----------

#compare col names to compare overlap
flights = []
stations = []
weather = []
created = []
for i in df_OTPW_3M.columns:
  if i in df_flights.columns:
    flights.append(i)
  elif i in df_stations.columns:
    stations.append(i)
  elif i in df_weather.columns:
    weather.append(i)
  else:
    created.append(i)
print(weather)
print(stations)
print(flights)
print(created)

# COMMAND ----------

# MAGIC %md # Flights & Stations & Airports join

# COMMAND ----------

# find weather stations with min distance to airport
df_closest_stations = df_stations.groupBy("neighbor_call")
df_helper = df_closest_stations.min('distance_to_neighbor')
df_stations = df_stations.join(df_helper,['neighbor_call'], 'left')

# filter out stations that are farther away
df_closest_stations = df_stations.filter(col('distance_to_neighbor') == col('min(distance_to_neighbor)'))

display(df_closest_stations)


# COMMAND ----------

# get airport iata code 
df_sta_air = df_closest_stations.join(df_airports, df_airports['ident'] == df_closest_stations['neighbor_call'], 'left')
table_size_calculator([df_sta_air],['df_sta_air'])

# COMMAND ----------

# find weather stations with min distance to airport
df_closest_stations = df_stations.groupBy("neighbor_call")
df_helper = df_closest_stations.min('distance_to_neighbor')
df_stations = df_stations.join(df_helper,['neighbor_call'], 'left')

# filter out stations that are farther away
df_closest_stations = df_stations.filter(col('distance_to_neighbor') == col('min(distance_to_neighbor)'))

display(df_closest_stations)


# join closest stations to airports

#getting weather station data for all destination airports
df_FS = df_flights.join(df_sta_air, df_flights["DEST"] == df_sta_air["iata_code"], "left")

#renaming destination station columns with dest_station prefix
df_FS = df_FS.select([df_FS[col].alias(f"{col}") for col in df_flights.columns] + [df_FS[col].alias(f"dest_station_{col}") for col in df_closest_stations.columns])

#getting origin weather station data fo ral lorigin airports
df_FS = df_FS.join(df_sta_air, df_FS["ORIGIN"] == df_sta_air["iata_code"], "left")

#renaming origin station columns with origin_station prefix
col_names = df_sta_air.columns
for i in col_names:
  df_FS = df_FS.withColumnRenamed(i,f"origin_station_{i}")

display(df_FS)

# COMMAND ----------

# MAGIC %md ### To do:
# MAGIC - need to convert everything to common time zone - utc
# MAGIC - watch out of daylight savings time
# MAGIC - get time zones from state abrevations 
# MAGIC

# COMMAND ----------

# MAGIC %md # (Flights + Stations + Airports) & Weather join

# COMMAND ----------


state_time_zones = {
  'AL': 'America/Chicago',   # Central Time Zone
  'AK': 'America/Anchorage', # Alaska Time Zone
  'AZ': 'America/Phoenix',   # Mountain Standard Time (no DST)
  'AR': 'America/Chicago',   # Central Time Zone
  'CA': 'America/Los_Angeles',# Pacific Time Zone
  'CO': 'America/Denver',    # Mountain Time Zone
  'CT': 'America/New_York',  # Eastern Time Zone
  'DE': 'America/New_York',  # Eastern Time Zone
  'FL': 'America/New_York',  # Eastern Time Zone
  'GA': 'America/New_York',  # Eastern Time Zone
  'HI': 'Pacific/Honolulu',  # Hawaii-Aleutian Time Zone
  'ID': 'America/Boise',     # Mountain Time Zone
  'IL': 'America/Chicago',   # Central Time Zone
  'IN': 'America/Indiana/Indianapolis', # Eastern Time Zone
  'IA': 'America/Chicago',   # Central Time Zone
  'KS': 'America/Chicago',   # Central Time Zone
  'KY': 'America/New_York',  # Eastern Time Zone
  'LA': 'America/Chicago',   # Central Time Zone
  'ME': 'America/New_York',  # Eastern Time Zone
  'MD': 'America/New_York',  # Eastern Time Zone
  'MA': 'America/New_York',  # Eastern Time Zone
  'MI': 'America/Detroit',   # Eastern Time Zone
  'MN': 'America/Chicago',   # Central Time Zone
  'MS': 'America/Chicago',   # Central Time Zone
  'MO': 'America/Chicago',   # Central Time Zone
  'MT': 'America/Denver',    # Mountain Time Zone
  'NE': 'America/Chicago',   # Central Time Zone
  'NV': 'America/Los_Angeles',# Pacific Time Zone
  'NH': 'America/New_York',  # Eastern Time Zone
  'NJ': 'America/New_York',  # Eastern Time Zone
  'NM': 'America/Denver',    # Mountain Time Zone
  'NY': 'America/New_York',  # Eastern Time Zone
  'NC': 'America/New_York',  # Eastern Time Zone
  'ND': 'America/Chicago',   # Central Time Zone
  'OH': 'America/New_York',  # Eastern Time Zone
  'OK': 'America/Chicago',   # Central Time Zone
  'OR': 'America/Los_Angeles',# Pacific Time Zone
  'PA': 'America/New_York',  # Eastern Time Zone
  'RI': 'America/New_York',  # Eastern Time Zone
  'SC': 'America/New_York',  # Eastern Time Zone
  'SD': 'America/Chicago',   # Central Time Zone
  'TN': 'America/Chicago',   # Central Time Zone
  'TX': 'America/Chicago',   # Central Time Zone
  'UT': 'America/Denver',    # Mountain Time Zone
  'VT': 'America/New_York',  # Eastern Time Zone
  'VA': 'America/New_York',  # Eastern Time Zone
  'WA': 'America/Los_Angeles',# Pacific Time Zone
  'WV': 'America/New_York',  # Eastern Time Zone
  'WI': 'America/Chicago',   # Central Time Zone
  'WY': 'America/Denver',    # Mountain Time Zone
  'VI': 'America/St_Thomas',  # Atlantic Time Zone
  'AS': 'Pacific/Pago_Pago', # Samoa Time Zone
  'GU': 'Pacific/Guam',      # Chamorro Time Zone
  'MP': 'Pacific/Saipan',     # Chamorro Time Zone
  'PR': 'America/Puerto_Rico'  # Atlantic Standard Time
}


df_FS = df_FS.withColumn("time_zone", col("ORIGIN_STATE_ABR"))


# COMMAND ----------

df_FS = df_FS.replace(state_time_zones, subset = ['time_zone'])

# COMMAND ----------



# defining a custom function to extract hours/minutes

def min_getter(time):
  time = str(time)
  minutes = time[-2:]
  # hours = time[0:-2]
  return minutes


min_getter_udf = udf(min_getter,StringType())

def hour_getter(time):
  time = str(time)
  # minutes = time[-2:]
  hours = time[0:-2]
  return hours


hour_getter_udf = udf(hour_getter,StringType())


# COMMAND ----------

df_FS = df_FS.withColumn('sch_depart_minute',min_getter_udf(col('CRS_DEP_TIME')))
df_FS = df_FS.withColumn('sch_depart_hour',hour_getter_udf(col('CRS_DEP_TIME')))

# COMMAND ----------

# merge cols together to get a date time
def time_merge(ymd,hour,minute):
  ymd = str(ymd)
  hour = str(hour)
  minute = str(minute)
  return ymd+'T'+hour+':'+minute

time_merge_udf = udf(time_merge,StringType())

# COMMAND ----------

df_FS = df_FS.withColumn('departing_local_time',time_merge_udf(col('FL_DATE'),col('sch_depart_hour'), col('sch_depart_minute')))

# COMMAND ----------

# dropping data missing outcome variable
# this is impacting utc time conversion negatively (i.e. getting 'none' for hour/minutes)
# losing 84710 records

df_FS = df_FS.filter(df_FS.DEP_DEL15 == df_FS.DEP_DEL15)
df_FS.count()

# COMMAND ----------

#creating col of utc depart time
df_FS = df_FS.withColumn("depart_UTC", to_utc_timestamp(col("departing_local_time"), col('time_zone')))
# Is it fair to assume the date for weather is in UTC?

#converting weather dates to utc:

df_weather = df_weather.withColumn("DATE_UTC",to_utc_timestamp(col('DATE'),"UTC"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC - need to join on composite key ( station, time-window)
# MAGIC - reports are at fixed interval : 2, 5, 8, 11, 14, 17, 20, 23
# MAGIC - calculate window that is AT LEAST 2 hours before
# MAGIC - pull data in from station window.
# MAGIC - 7:59 will end as hour 7 even though it's closer to hour 8. If we have time to play with it we can explore improving.
# MAGIC ###bigger questions:
# MAGIC - how long does bad weather impact airport traffic?
# MAGIC - is it more important to have most recent report
# MAGIC - or is it better to have report of previous 24 hours?
# MAGIC
# MAGIC

# COMMAND ----------

#going back 3 hours for analysis:

df_FS= df_FS.withColumn('Depart_3hrs_before', expr("depart_UTC - interval 3 hours"))

#creating cols for time component to reconstruct date.time to match composite key pulls

df_FS = df_FS.withColumn('hour_3prior', hour(col('Depart_3hrs_before')))
df_FS = df_FS.withColumn('month_3prior', F.month(col('Depart_3hrs_before')))
df_FS= df_FS.withColumn('day_3prior', F.day(col('Depart_3hrs_before')))
df_FS= df_FS.withColumn('year_3prior', F.year(col('Depart_3hrs_before')))


# COMMAND ----------

# create function to select closest hour to weather report without getting info from beyond window.
def time_window(hour):
  """Creating composite key to match with weather report time intervals. Changing the hour to match available reports"""
  if hour is not None:
    hour = int(hour)
    weather = { 0: 23,
      1: 23,
      2: 2,
      3: 2,
      4: 2,
      5: 5,
      6: 5,
      7: 5,
      8: 8,
      9: 8,
      10: 8,
      11: 11,
      12: 11,
      13: 11,
      14: 14,
      15: 14,
      16: 14,
      17: 17,
      18: 17,
      19: 17,
      20: 20,
      21: 20,
      22: 20,
      23: 23}
    return weather[hour]
  else:
    return None
  

# COMMAND ----------

time_window_udf = udf(time_window, IntegerType())
df_FS = df_FS.withColumn('time_key_3hr', time_window_udf(col('hour_3prior')))

# COMMAND ----------

# MAGIC %md next steps:
# MAGIC - merge year, month, day , hour back together
# MAGIC - convert to utc
# MAGIC - make composite key (station id, time_key)
# MAGIC - left join df_FS and weather.

# COMMAND ----------

def time_merge2(year,month,day,hour):
  """create time key for composite key by reassembling utc date"""
  year = str(year)
  month = str(month)
  day = str(day)
  hour = str(hour)
  return year+'-'+month+'-'+day+'T'+hour+':'+'00'

time_merge2_udf = udf(time_merge2,StringType())

# COMMAND ----------

#get utc time stamp that will match with weather station as part of composite key
df_FS = df_FS.withColumn('time_key_3hr_helper', time_merge2_udf(col('year_3prior'),col('month_3prior'),col('day_3prior'), col('time_key_3hr')))

#convert back to utc
df_FS = df_FS.withColumn('time_key_3hr_utc', to_utc_timestamp(col('time_key_3hr_helper'),'UTC'))

# COMMAND ----------

#extracting date information from weather
df_weather = df_weather.withColumn('weather_year', F.year(col('DATE_UTC')))
df_weather = df_weather.withColumn('weather_month', F.month(col('DATE_UTC')))
df_weather = df_weather.withColumn('weather_day', F.day(col('DATE_UTC')))
df_weather = df_weather.withColumn('weather_hour', F.hour(col('DATE_UTC')))

# COMMAND ----------

# MAGIC %md can engineer date_utc in weather so that it's format matches
# MAGIC - how do we handle duplicate records?
# MAGIC - is that the origin of the time zone parsing error?
# MAGIC

# COMMAND ----------

def time_window2(hour):
  """Creating composite key to match with flight depart time intervals. Changing the hour to match flight departures"""
  if hour is not None:
    hour = int(hour)
    weather = {
      0: 2,
      1: 2,
      2: 2,
      3: 5,
      4: 5,
      5: 5,
      6: 8,
      7: 8,
      8: 8,
      9: 11,
      10: 11,
      11: 11,
      12: 14,
      13: 14,
      14: 14,
      15: 17,
      16: 17,
      17: 17,
      18: 20,
      19: 20,
      20: 20,
      21: 23,
      22: 23,
      23: 23}
    return weather[hour]
  else:
    return None

time_window2_udf = udf(time_window2,IntegerType())


# COMMAND ----------

#went down this route of trying to sync weather station times with flight times. 
#since like 30% of the the station data was in a fixed increments it seemed sensible to 
#convert everything to those levels,
#flights go to prior levels
#weather goes to future levels
#prevent data leakage of future weather 
#being associated with a flight

#change hour to match exisiting levels for flights/weather stations
df_weather = df_weather.withColumn('hour_key', time_window2_udf(col('weather_hour')))

#creating time key
df_weather = df_weather.withColumn('Time_key', time_merge2_udf(col('weather_year'),col('weather_month'),col('weather_day'),col('hour_key')))

#converting to utc
df_weather = df_weather.withColumn('Time_key',to_utc_timestamp(col('Time_key'),'UTC'))

#creating unique rows

df_weather = df_weather.withColumn('unique_rows', F.concat(col('STATION'),col('Time_key').cast('string')))

#dropping duplicate weather reports results in losing 22788338 records.
# collapses data to 25% of the rows. losing a lot of granularity
df_weather_deduped = df_weather.dropDuplicates(['unique_rows'])
# running into error, maybe from missing origin station information
# lose 20,000ish records when dropping missing origin station id
df_FS  = df_FS.filter(df_FS['origin_station_station_id']== df_FS['origin_station_station_id'])
# display(df_FS.join(df_weather_deduped,(df_FS['origin_station_station_id']==df_weather['STATION']), 'left'))


# COMMAND ----------

# display(df_FS.join(df_weather_deduped,(df_FS['origin_station_station_id']==df_weather['STATION'])& (df_weather['Time_key'] == df_FS['time_key_3hr_utc']), 'left'))
display(df_FS)

# COMMAND ----------

df_FSW = df_FS.join(df_weather_deduped,(df_FS['origin_station_station_id']==df_weather['STATION'])& (df_weather['Time_key'] == df_FS['time_key_3hr_utc']), 'left')

# COMMAND ----------



# COMMAND ----------

df_FSW.columns

# COMMAND ----------

#saving joined dataset to blob

# df_FSW.write.mode('overwrite').parquet(f"{team_blob_url}/df_FS_3M")
# df_FSW.write.parquet(f"{team_blob_url}/df_FS_60M")

df_FSW_60M = spark.read.parquet(f"{team_blob_url}/df_FS_60M")

# COMMAND ----------

#evaluating missing data % from joined dataset to compare with provided joined dataset for sanity check. Looks good :) 

df_FSW_60M_row_ct = 72515921

results_df = df_FSW_60M.select([col(c).cast("string") for c in df_FSW_60M.columns]).select( \
                    [(count(when(isnan(c) | col(c).isNull(), c)) / df_FSW_60M_row_ct * 100).alias(c) for c in df_FSW_60M.columns]) \
                    .toPandas().T.rename(columns={0: 'Missing_Value_Percentage'}).reset_index().merge(pd.DataFrame(df_FSW_60M.dtypes, columns=['index', 'DataType']), on = 'index') \
                    .sort_values('Missing_Value_Percentage', ascending = False)
print(results_df)

# COMMAND ----------

print(results_df)

# COMMAND ----------

