# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering

# COMMAND ----------

#Imports

from pyspark.sql.functions import col, lit, date_format, to_utc_timestamp, expr, hour
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql.functions import col,isnan, when, count, col, split, trim, lit, avg, sum, expr, udf
from pyspark.sql import functions as F
from pyspark.sql import Window
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

# try:
#     df_FS_3M = spark.read.parquet(f"{team_blob_url}/df_FS_3M")
    
# except:                
#   print('error loading file')

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

df_flights.columns


# COMMAND ----------

df_weather.columns

# COMMAND ----------

df_stations.columns

# COMMAND ----------

df_airports.columns

# COMMAND ----------

#dropping duplicate records
df_flights = df_flights.dropDuplicates()
df_weather = df_weather.dropDuplicates()

#dropping columns with major amounts of missing data

pd.set_option('display.max_rows', None)

try:
  df_col_analysis = pd.read_parquet('pd_data/df_fli_col_analysis.parquet')
except:
  df_flights_row_ct = df_flights.count()
  df_col_analysis = df_flights.select( \
                      [(count(when(col(c).isNull(), c)) / df_flights_row_ct * 100).alias(c) for c in df_flights.columns]) \
                      .toPandas().T.rename(columns={0: 'Missing_Value_Percentage'}).reset_index().merge(pd.DataFrame(df_flights.dtypes, columns=['index', 'DataType']), on = 'index') \
                      .sort_values('Missing_Value_Percentage', ascending = False)
  df_col_analysis.to_parquet('pd_data/df_fli_col_analysis.parquet', index = False)

df_col_analysis
df_flights = df_flights.drop(*df_col_analysis[df_col_analysis.Missing_Value_Percentage >= 99]['index'].tolist())


# COMMAND ----------

try:
  df_col_analysis = pd.read_parquet('pd_data/df_wea_col_analysis.parquet')
except:
  df_weather_row_ct = df_weather.count()
  df_col_analysis = df_weather.select( \
                      [(count(when(col(c).isNull(), c)) / df_weather_row_ct * 100).alias(c) for c in df_weather.columns]) \
                      .toPandas().T.rename(columns={0: 'Missing_Value_Percentage'}).reset_index().merge(pd.DataFrame(df_weather.dtypes, columns=['index', 'DataType']), on = 'index') \
                      .sort_values('Missing_Value_Percentage', ascending = False)
  df_col_analysis.to_parquet('pd_data/df_wea_col_analysis.parquet', index = False)

df_col_analysis
df_weather = df_weather.drop(*df_col_analysis[df_col_analysis.Missing_Value_Percentage >= 99]['index'].tolist())

# COMMAND ----------

from graphframes import *

def page_rank(df):
    flight_matrix = df.groupBy(['ORIGIN','DEST']).count()
    # vertices DataFrame
    vertices = flight_matrix.select("ORIGIN").union(flight_matrix.select("DEST")).distinct().withColumnRenamed("ORIGIN", "id")

    # Rename the dataframe cols to match GraphFrames requirements
    edges = flight_matrix.withColumnRenamed("ORIGIN", "src").withColumnRenamed("DEST", "dst").withColumnRenamed("count", "relationship")

    # Create a GraphFrame and run pagerank
    g = GraphFrame(vertices, edges)

    results = g.pageRank(resetProbability=0.15, tol=0.01)
    df = df.join(results.vertices,  df.ORIGIN == results.vertices.id, 'left')

    return df
  
df_flights = page_rank(df_flights)

display(df_flights)

# COMMAND ----------

df_flights = df_flights.withColumnRenamed('YEAR','year_flights')

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

# COMMAND ----------


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

# MAGIC %md # (Flights + Stations + Airports) & Weather join

# COMMAND ----------

#using these times zones will automatically account for daylight savings time when using to_UTC function

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
  'PR': 'America/Puerto_Rico',  # Atlantic Standard Time
  'TT': 'Pacific/Chatham' 
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
# df_FS.count()

# COMMAND ----------

#creating col of utc depart time, 'timezone' will correct for daylight savings time.
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

#going back 2 hours for analysis:

df_FS= df_FS.withColumn('Depart_2hrs_before', expr("depart_UTC - interval 2 hours"))

#creating cols for time component to reconstruct date.time to match composite key pulls

df_FS = df_FS.withColumn('hour_2prior', hour(col('Depart_2hrs_before')))
df_FS = df_FS.withColumn('month_2prior', F.month(col('Depart_2hrs_before')))
df_FS= df_FS.withColumn('day_2prior', F.day(col('Depart_2hrs_before')))
df_FS= df_FS.withColumn('year_2prior', F.year(col('Depart_2hrs_before')))


# COMMAND ----------

# create function to select closest hour to weather report without getting info from beyond window.
def time_window(hour):
  """Creating composite key to match with weather report time intervals. Changing the hour to match available reports"""
  if hour is not None:
    hour = int(hour)
    weather = { 0: 0,
      1: 0,
      2: 2,
      3: 2,
      4: 4,
      5: 4,
      6: 6,
      7: 6,
      8: 8,
      9: 8,
      10: 10,
      11: 10,
      12: 12,
      13: 12,
      14: 14,
      15: 14,
      16: 16,
      17: 16,
      18: 18,
      19: 118,
      20: 20,
      21: 20,
      22: 22,
      23: 22}
    return weather[hour]
  else:
    return None
  

# COMMAND ----------

time_window_udf = udf(time_window, IntegerType())
df_FS = df_FS.withColumn('time_key_2hr', time_window_udf(col('hour_2prior')))

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
df_FS = df_FS.withColumn('time_key_2hr_helper', time_merge2_udf(col('year_2prior'),col('month_2prior'),col('day_2prior'), col('time_key_2hr')))

#convert back to utc
df_FS = df_FS.withColumn('time_key_2hr_utc', to_utc_timestamp(col('time_key_2hr_helper'),'UTC'))

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
  """Creating composite key to match with flight depart time intervals. Changing the hour to match flight departures
  flight departure times moved backwards 5pm flights -> appears as if it departs at 4pm.
  This is so when we join weather data, it pulls earlier weather reports * NOT later weather reports which might leak data.
  Weather time key needs to match with flight time keys. To do this. Weather times are pushed forward. 5pm weather -> appears as if it happend at 6pm.
  
  degenerate scenario: a 5pm flight pulls weather from 2pm (2 hours prior: 5pm->3pm, 1 hour offset ->2pm).

  Without this offset/buffer, you might get weather from 1.5 hours prior to a flight. 


  """
  if hour is not None:
    hour = int(hour)
    weather = {
      0: 0,
      1: 2,
      2: 2,
      3: 4,
      4: 4,
      5: 6,
      6: 6,
      7: 8,
      8: 8,
      9: 10,
      10: 10,
      11: 12,
      12: 12,
      13: 14,
      14: 14,
      15: 16,
      16: 16,
      17: 18,
      18: 18,
      19: 20,
      20: 20,
      21: 22,
      22: 22,
      23: 0}
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




# COMMAND ----------

col_check_list = df_weather.columns
cols = ['HourlyAltimeterSetting',
'HourlyDewPointTemperature',
'HourlyDryBulbTemperature',
'HourlyPrecipitation',
'HourlyPresentWeatherType',
'HourlyPressureChange',
'HourlyPressureTendency',
'HourlyRelativeHumidity',
'HourlySkyConditions',
'HourlySeaLevelPressure',
'HourlyStationPressure',
'HourlyVisibility',
'HourlyWetBulbTemperature',
'HourlyWindDirection',
'HourlyWindGustSpeed',
'HourlyWindSpeed']

for i in cols:
  if i in col_check_list:
    print(str(i))

# COMMAND ----------

cols = ['HourlyAltimeterSetting',
'HourlyDewPointTemperature',
'HourlyDryBulbTemperature',
'HourlyPrecipitation',
'HourlyPresentWeatherType',
'HourlyPressureChange',
'HourlyPressureTendency',
'HourlyRelativeHumidity',
'HourlySkyConditions',
'HourlySeaLevelPressure',
'HourlyStationPressure',
'HourlyVisibility',
'HourlyWetBulbTemperature',
'HourlyWindDirection',
'HourlyWindGustSpeed',
'HourlyWindSpeed']


## idea -> do I merge the weather with flights first and then run the agg since there will be less records? or do I do this agg first?

# display(df_weather)

df_test = df_weather.groupBy(col('unique_rows')).agg(F.max(col('HourlyAltimeterSetting')).alias('Max_HourlyAltimeterSetting'),
                                                           F.min(col('HourlyAltimeterSetting')).alias('Min_HourlyAltimeterSetting'),
                                                            F.avg(col('HourlyAltimeterSetting')).alias('Avg_HourlyAltimeterSetting'),
                                                            F.max(col('HourlyDewPointTemperature')).alias('Max_HourlyDewPointTemperature'),
                                                           F.min(col('HourlyDewPointTemperature')).alias('Min_HourlyDewPointTemperature'),
                                                            F.avg(col('HourlyDewPointTemperature')).alias('Avg_HourlyDewPointTemperature'),
                                                            F.max(col('HourlyDryBulbTemperature')).alias('Max_HourlyDryBulbTemperature'),
                                                           F.min(col('HourlyDryBulbTemperature')).alias('Min_HourlyDryBulbTemperature'),
                                                            F.avg(col('HourlyDryBulbTemperature')).alias('Avg_HourlyDryBulbTemperature'),
                                                            F.max(col('HourlyPrecipitation')).alias('Max_HourlyPrecipitation'),
                                                           F.min(col('HourlyPrecipitation')).alias('Min_HourlyPrecipitation'),
                                                            F.avg(col('HourlyPrecipitation')).alias('Avg_HourlyPrecipitation'),
                                                            F.max(col('HourlyPresentWeatherType')).alias('Max_HourlyPresentWeatherType'),
                                                           F.min(col('HourlyPresentWeatherType')).alias('Min_HourlyPresentWeatherType'),
                                                            F.avg(col('HourlyPresentWeatherType')).alias('Avg_HourlyPresentWeatherType'),
                                                            F.max(col('HourlyPressureChange')).alias('Max_HourlyPressureChange'),
                                                           F.min(col('HourlyPressureChange')).alias('Min_HourlyPressureChange'),
                                                            F.avg(col('HourlyPressureChange')).alias('Avg_HourlyPressureChange'),
                                                            F.max(col('HourlyPressureTendency')).alias('Max_HourlyPressureTendency'),
                                                           F.min(col('HourlyPressureTendency')).alias('Min_HourlyPressureTendency'),
                                                            F.avg(col('HourlyPressureTendency')).alias('Avg_HourlyPressureTendency'),
                                                            F.max(col('HourlyRelativeHumidity')).alias('Max_HourlyRelativeHumidity'),
                                                           F.min(col('HourlyRelativeHumidity')).alias('Min_HourlyRelativeHumidity'),
                                                            F.avg(col('HourlyRelativeHumidity')).alias('Avg_HourlyRelativeHumidity'),
                                                            F.max(col('HourlySkyConditions')).alias('Max_HourlySkyConditions'),
                                                           F.min(col('HourlySkyConditions')).alias('Min_HourlySkyConditions'),
                                                            F.avg(col('HourlySkyConditions')).alias('Avg_HourlySkyConditions'),
                                                            F.max(col('HourlySeaLevelPressure')).alias('Max_HourlySeaLevelPressure'),
                                                           F.min(col('HourlySeaLevelPressure')).alias('Min_HourlySeaLevelPressure'),
                                                            F.avg(col('HourlySeaLevelPressure')).alias('Avg_HourlySeaLevelPressure'),
                                                            F.max(col('HourlyStationPressure')).alias('Max_HourlyStationPressure'),
                                                           F.min(col('HourlyStationPressure')).alias('Min_HourlyStationPressure'),
                                                            F.avg(col('HourlyStationPressure')).alias('Avg_HourlyStationPressure'),
                                                            F.max(col('HourlyVisibility')).alias('Max_HourlyVisibility'),
                                                           F.min(col('HourlyVisibility')).alias('Min_HourlyVisibility'),
                                                            F.avg(col('HourlyVisibility')).alias('Avg_HourlyVisibility'),
                                                           F.max(col('HourlyWetBulbTemperature')).alias('Max_HourlyWetBulbTemperature'),
                                                           F.min(col('HourlyWetBulbTemperature')).alias('Min_HourlyWetBulbTemperature'),
                                                            F.avg(col('HourlyWetBulbTemperature')).alias('Avg_HourlyWetBulbTemperature'),
                                                           F.max(col('HourlyWindDirection')).alias('Max_HourlyWindDirection'),
                                                           F.min(col('HourlyWindDirection')).alias('Min_HourlyWindDirection'),
                                                            F.avg(col('HourlyWindDirection')).alias('Avg_HourlyWindDirection'),
                                                            F.max(col('HourlyWindGustSpeed')).alias('Max_HourlyWindGustSpeed'),
                                                           F.min(col('HourlyWindGustSpeed')).alias('Min_HourlyWindGustSpeed'),
                                                            F.avg(col('HourlyWindGustSpeed')).alias('Avg_HourlyWindGustSpeed'),
                                                            F.max(col('HourlyWindSpeed')).alias('Max_HourlyWindSpeed'),
                                                           F.min(col('HourlyWindSpeed')).alias('Min_HourlyWindSpeed'),
                                                            F.avg(col('HourlyWindSpeed')).alias('Avg_HourlyWindSpeed')
                                                           
                                                           
                                                           )
display(df_test)

# COMMAND ----------

#originally used windowing, but noticed that took 30ish minutes per agg function. Doing groupby aggs resulted in 10 minute aggs, but we'll have to join the new df on weather which will take some time.

# window = Window.partitionBy('unique_rows')
# df_weather =  df_weather.withColumn('max_HourlyAltimeterSetting', F.max('HourlyAltimeterSetting').over(window))

# df_weather =  df_weather.withColumn('max_HourlyDewPointTemperature', F.max('HourlyDewPointTemperature').over(window))

# ## idea -> do I merge the weather with flights first and then run the agg since there will be less records? or do I do this agg first?

# #Window commands took 30-ish minutes each.
# display(df_weather)

# COMMAND ----------

df_weather = df_weather.join(df_test, on = 'unique_rows', how = 'left')

display(df_weather)

# COMMAND ----------

#dropping duplicate weather reports results in losing 22788338 records.
# collapses data to 25% of the rows. losing a lot of granularity
df_weather = df_weather.dropDuplicates(['unique_rows'])
# running into error, maybe from missing origin station information
# lose 20,000ish records when dropping missing origin station id
df_FS  = df_FS.filter(df_FS['origin_station_station_id']== df_FS['origin_station_station_id'])

# COMMAND ----------

df_FSW = df_FS.join(df_weather,(df_FS['origin_station_station_id']==df_weather['STATION'])& (df_weather['Time_key'] == df_FS['time_key_2hr_utc']), 'left')

# COMMAND ----------

#saving joined dataset to blob

# df_FSW.write.mode('overwrite').parquet(f"{team_blob_url}/df_FS_3M")
df_FSW.write.parquet(f"{team_blob_url}/df_FSW_2023_12_8")

df_FSW_60M = spark.read.parquet(f"{team_blob_url}/df_FSW_2023_12_8")

# COMMAND ----------

df_FSW_60M.count()

# COMMAND ----------

