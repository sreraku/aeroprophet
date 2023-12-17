# Databricks notebook source
# MAGIC %md
# MAGIC ### Imports

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql import functions as F
from pyspark.sql.functions import col, isnan, when, count, split, trim, lit, avg, sum, expr
from pyspark.sql.types import IntegerType, FloatType, StringType, TimestampType
from pyspark.ml.feature import Imputer
from pyspark.sql.window import Window
import ast
import holidays
from dateutil.parser import parse
from datetime import datetime

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

df_OTPW_60M = spark.read.parquet(f"{team_blob_url}/OTPW_60M").cache() #provided data
df_teamjoined_60M = spark.read.parquet(f"{team_blob_url}/df_FSW_2023_12_8").cache() #joined data created by our team

# COMMAND ----------

# MAGIC %md
# MAGIC ### Examine Features

# COMMAND ----------

# MAGIC %md
# MAGIC Start by reviewing columns present in the provided data that are missing from our joined data and attempt to map them to extra columns in our joined data

# COMMAND ----------

missing_from_join = [c for c in df_OTPW_60M.columns if c not in [*df_teamjoined_60M.columns]]
                
missing_from_join

# COMMAND ----------

extra_cols_in_join = [c for c in df_teamjoined_60M.columns if c not in [*df_OTPW_60M.columns]]
                
extra_cols_in_join

# COMMAND ----------

column_mappings = {'origin_station_station_id': 'origin_station_id',
                    'origin_station_iata_code': 'origin_iata_code',
                    'origin_station_type': 'origin_type',
                    'origin_station_iso_region': 'origin_region',
                    'origin_station_neighbor_lat': 'origin_airport_lat',
                    'origin_station_neighbor_lon': 'origin_airport_lon',
                    'origin_station_distance_to_neighbor': 'origin_station_dis',
                    'dest_station_station_id': 'dest_station_id',
                    'dest_station_iata_code': 'dest_iata_code',
                    'dest_station_type': 'dest_type',
                    'dest_station_iso_region': 'dest_region',
                    'dest_station_neighbor_lat': 'dest_airport_lat',
                    'dest_station_neighbor_lon': 'dest_airport_lon',
                    'dest_station_distance_to_neighbor': 'dest_station_dis',
                    'depart_UTC': 'sched_depart_date_time_UTC'}

for c in column_mappings:
    df_teamjoined_60M = df_teamjoined_60M.withColumnRenamed(c,column_mappings[c])

# COMMAND ----------

# MAGIC %md
# MAGIC The following do not have equivalents in our team's joined data

# COMMAND ----------

missing_from_join = [c for c in df_OTPW_60M.columns if c not in [*df_teamjoined_60M.columns]]
                
missing_from_join

# COMMAND ----------

# MAGIC %md
# MAGIC The following extras do not have equivalents in the provided data

# COMMAND ----------

extra_cols_in_join = [c for c in df_teamjoined_60M.columns if c not in [*df_OTPW_60M.columns]]
                
extra_cols_in_join

# COMMAND ----------

# MAGIC %md
# MAGIC Many of the remaining missing features relate to diverted flights. Since diversions can happen after the scheduled flight time, these can lead to data leakage with our desired prediction window (two hours before take off). The other missing feature are redundant, uninformative, or unclear. Let's drop them and use our team's joined data going forward.  

# COMMAND ----------

#seperating out derived features  to keep:
cols_to_keep = [
    'Depart_2hrs_before',
    'pagerank',
    'Max_HourlyAltimeterSetting',
 'Min_HourlyAltimeterSetting',
 'Avg_HourlyAltimeterSetting',
 'Max_HourlyDewPointTemperature',
 'Min_HourlyDewPointTemperature',
 'Avg_HourlyDewPointTemperature',
 'Max_HourlyDryBulbTemperature',
 'Min_HourlyDryBulbTemperature',
 'Avg_HourlyDryBulbTemperature',
 'Max_HourlyPrecipitation',
 'Min_HourlyPrecipitation',
 'Avg_HourlyPrecipitation',
 'Max_HourlyPressureChange',
 'Min_HourlyPressureChange',
 'Avg_HourlyPressureChange',
 'Max_HourlyPressureTendency',
 'Min_HourlyPressureTendency',
 'Avg_HourlyPressureTendency',
 'Max_HourlyRelativeHumidity',
 'Min_HourlyRelativeHumidity',
 'Avg_HourlyRelativeHumidity',
 'Max_HourlySeaLevelPressure',
 'Min_HourlySeaLevelPressure',
 'Avg_HourlySeaLevelPressure',
 'Max_HourlyStationPressure',
 'Min_HourlyStationPressure',
 'Avg_HourlyStationPressure',
 'Max_HourlyVisibility',
 'Min_HourlyVisibility',
 'Avg_HourlyVisibility',
 'Max_HourlyWetBulbTemperature',
 'Min_HourlyWetBulbTemperature',
 'Avg_HourlyWetBulbTemperature',
 'Max_HourlyWindDirection',
 'Min_HourlyWindDirection',
 'Avg_HourlyWindDirection',
 'Max_HourlyWindGustSpeed',
 'Min_HourlyWindGustSpeed',
 'Avg_HourlyWindGustSpeed',
 'Max_HourlyWindSpeed',
 'Min_HourlyWindSpeed',
 'Avg_HourlyWindSpeed',
 'sch_depart_hour'
]


# COMMAND ----------

df_60M = df_teamjoined_60M.drop(*[c for c in extra_cols_in_join if c not in cols_to_keep])

# COMMAND ----------

# MAGIC %md
# MAGIC There are many fields that contain only missing-values. Namely, all of the monthly level and much of the daily weather data (https://www.ncei.noaa.gov/pub/data/cdo/documentation/LCD_documentation.pdf). This should not be an issue for our purposes given we want to predict at a much more granular level (2 hrs before departure). We can safely drop fields with greater than 99% missing values given then contain little to no information.

# COMMAND ----------

# MAGIC %md
# MAGIC We must also be careful to avoid data leakage. Any variable that is known **after** our desired prediction time should be dropped - with a few exceptions. First, we need an outcome variable that will serve as a label. 'DEP_DEL15' is such a variable and provides a binary indicator for delays greater than 15 minutes. This will assist us in minimizing loss during training and for evaluation during testing. Next, while we can't use many variables for a given flight at prediction time, we can use those variables from prior flights. For example, consider that an aircraft must be located in a depature location in order to leave. If we identify delays on previous flights (DEP_DELAY), they could assist in predicting future delays.

# COMMAND ----------

data_leakage_cols = ['DEP_TIME',
                    'DEP_DELAY_NEW',
                    'DEP_DELAY_GROUP',
                    'DEP_TIME_BLK',
                    'TAXI_OUT',
                    'WHEELS_OFF',
                    'WHEELS_ON',
                    'TAXI_IN',
                    'ACTUAL_ELAPSED_TIME',
                    'CRS_ELAPSED_TIME',
                    'ARR_TIME',
                    'ARR_DELAY',
                    'ARR_DELAY_NEW',
                    'ARR_DEL15',
                    'ARR_DELAY_GROUP',
                    'ARR_TIME_BLK',
                    'CANCELLED',
                    'CANCELLATION_CODE',
                    'DIVERTED',
                    'AIR_TIME',
                    'CARRIER_DELAY',
                    'WEATHER_DELAY',
                    'NAS_DELAY',
                    'SECURITY_DELAY',
                    'LATE_AIRCRAFT_DELAY']

df_60M = df_60M.drop(*data_leakage_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC Since each flight should only have a single record, let's drop the duplicates

# COMMAND ----------

df_60M = df_60M.na.drop(subset=["DEP_DEL15"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Type Conversion

# COMMAND ----------

type_dict = {}
for c, dtype in df_60M.dtypes:
    if dtype == 'string':
        value = df_60M.select(c).filter(col(c).isNotNull()).first()[c]

        try:
            literal_value = ast.literal_eval(value)
            if isinstance(literal_value, int):
                type_dict[c] = IntegerType()
            elif isinstance(literal_value, float):
                type_dict[c] = FloatType()
        except:
            try:
                parse(value)
                type_dict[c] = TimestampType()
            except:
                continue

for c in type_dict:
    df_60M = df_60M.withColumn(c, col(c).cast(type_dict[c]))

# COMMAND ----------

df_60M.cache()
display(df_60M)

# COMMAND ----------

display(df_60M.filter(col('FL_DATE')> '2019-12-31T00:00:00Z').select(col('FL_DATE'),col('YEAR')))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC Let's add a feature to calculate the delay of the previous flight for the same aircraft

# COMMAND ----------

df_60M.createOrReplaceTempView("df_60M_view")

df_60M = spark.sql("""
    SELECT t1.*,
        FIRST_VALUE(t2.DEP_DELAY) OVER (
            PARTITION BY t1.TAIL_NUM, t1.ORIGIN_AIRPORT_ID, t1.sched_depart_date_time_UTC
            ORDER BY t2.sched_depart_date_time_UTC DESC
        ) AS previous_flight_delay
    FROM df_60M_view t1
    LEFT JOIN (
        SELECT TAIL_NUM, DEST_AIRPORT_ID, sched_depart_date_time_UTC, DEP_DELAY, FL_DATE
        FROM df_60M_view
    ) t2
    ON t1.TAIL_NUM = t2.TAIL_NUM
    AND t1.ORIGIN_AIRPORT_ID = t2.DEST_AIRPORT_ID
    AND t1.Depart_2hrs_before > t2.sched_depart_date_time_UTC
    AND (DATEDIFF(t1.FL_DATE, t2.FL_DATE) BETWEEN -1 AND 1)
""")

# COMMAND ----------

# MAGIC %md
# MAGIC Let's create a feature that represents the distance of a given day to the closest holiday

# COMMAND ----------

# Define UDF
def days_to_nearest_holiday(current_date):
    current_date = current_date.date()
    years = [current_date.year - 1, current_date.year, current_date.year + 1]
    us_holidays = holidays.US(years=years)
    dist_to_holiday = min(abs(current_date - d).days for d in us_holidays.keys())

    return dist_to_holiday

# Register the UDF
days_to_nearest_holiday_udf = F.udf(days_to_nearest_holiday, IntegerType())

# Apply the UDF to create a new column
df_60M = df_60M.withColumn('days_to_nearest_holiday', days_to_nearest_holiday_udf(col('FL_DATE')))


# COMMAND ----------

# MAGIC %md
# MAGIC Extract hour component of local time string

# COMMAND ----------

# Convert scheduled dep and arr integers to 'hhmm' format
df_60M = df_60M.withColumn('CRS_DEP_TIME_Str', F.lpad(col('CRS_DEP_TIME'), 4, '0'))
df_60M = df_60M.withColumn('CRS_ARR_TIME_Str', F.lpad(col('CRS_ARR_TIME'), 4, '0'))

# Extract the hour part
df_60M = df_60M.withColumn('CRS_DEP_HOUR', F.substring('CRS_DEP_TIME_Str', 1, 2))
df_60M = df_60M.withColumn('CRS_ARR_HOUR', F.substring('CRS_ARR_TIME_Str', 1, 2))

# Convert these hour columns to integers
df_60M = df_60M.withColumn('CRS_DEP_HOUR', df_60M['CRS_DEP_HOUR'].cast('int'))
df_60M = df_60M.withColumn('CRS_ARR_HOUR', df_60M['CRS_ARR_HOUR'].cast('int'))

# COMMAND ----------

# MAGIC %md
# MAGIC add a feature for geographic region

# COMMAND ----------

# adding feature, state region
def state_to_region(state_abrv):
    """converts state abreviations to geographic region """
    region_mapping = {
    'AL': 'Southeast',
    'AK': 'West',
    'AZ': 'West',
    'AR': 'South',
    'CA': 'West',
    'CO': 'West',
    'CT': 'Northeast',
    'DE': 'Northeast',
    'FL': 'Southeast',
    'GA': 'Southeast',
    'HI': 'West',
    'ID': 'West',
    'IL': 'Midwest',
    'IN': 'Midwest',
    'IA': 'Midwest',
    'KS': 'Midwest',
    'KY': 'South',
    'LA': 'South',
    'ME': 'Northeast',
    'MD': 'Northeast',
    'MA': 'Northeast',
    'MI': 'Midwest',
    'MN': 'Midwest',
    'MS': 'South',
    'MO': 'Midwest',
    'MT': 'West',
    'NE': 'Midwest',
    'NV': 'West',
    'NH': 'Northeast',
    'NJ': 'Northeast',
    'NM': 'West',
    'NY': 'Northeast',
    'NC': 'Southeast',
    'ND': 'Midwest',
    'OH': 'Midwest',
    'OK': 'South',
    'OR': 'West',
    'PA': 'Northeast',
    'RI': 'Northeast',
    'SC': 'Southeast',
    'SD': 'Midwest',
    'TN': 'South',
    'TX': 'South',
    'UT': 'West',
    'VT': 'Northeast',
    'VA': 'Southeast',
    'WA': 'West',
    'WV': 'South',
    'WI': 'Midwest',
    'WY': 'West',
    'VI': 'Atlantic',  
    'AS': 'Pacific', 
    'GU': 'Pacific',      
    'MP': 'Pacific',     
    'PR': 'Atlantic',  
    'TT': 'Pacific' 
}
    return region_mapping.get(state_abrv, 'Unknown')
state_to_region_udf = udf(state_to_region, StringType())
df_60M = df_60M.withColumn('region',state_to_region_udf(col("ORIGIN_STATE_ABR")))

# COMMAND ----------

# MAGIC %md
# MAGIC add feature that checks # of flights in last 24 hours of tail number that were delayed

# COMMAND ----------

df_60M = df_60M.withColumn('sch_dep_time_linux', F.unix_timestamp("sched_depart_date_time_UTC"))

windowSpec = Window().partitionBy('TAIL_NUM').orderBy('sch_dep_time_linux')\
    .rangeBetween(-86400,-10800)
df_60M = df_60M.withColumn('Plane_Delays_last_24H', F.sum(F.col("DEP_DEL15").cast("int")).over(windowSpec))

# COMMAND ----------

# MAGIC %md
# MAGIC add feature that checks # flights delayed at airport in last 24 hours

# COMMAND ----------

windowSpec = Window().partitionBy('ORIGIN').orderBy('sch_dep_time_linux')\
    .rangeBetween(-86400,-10800)
df_60M = df_60M.withColumn('airport_delays_in_previous_24_hours', F.sum(F.col("DEP_DEL15").cast("int")).over(windowSpec))
df_60M = df_60M.fillna(0,subset=['Plane_Delays_last_24H','airport_delays_in_previous_24_hours'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Review Remaining Columns & Drop Uninformative or Redundant Features

# COMMAND ----------

# MAGIC %md
# MAGIC Let's review remaining variables and drop those that are unhelpful. This could be because it is categorical data that is too expensive to convert to one hot encoding (e.g. tail number has too many categories), it redudant of other information (e.g. LATITUDE is similar to dest_station_lat and less clear) or is a type not directly interpretable by ML models (e.g. datetime objects)

# COMMAND ----------

cols_to_drop = [
                        'OP_CARRIER',
                        'TAIL_NUM',
                        'ORIGIN_CITY_NAME',
                        'ORIGIN_STATE_ABR',
                        'ORIGIN_STATE_NM',
                        'DEST_CITY_NAME',
                        'DEST_STATE_ABR',
                        'DEST_STATE_NM',
                        'origin_station_name',
                        'origin_station_id',
                        'origin_iata_code',
                        'origin_region',
                        'dest_station_name',
                        'dest_station_id',
                        'dest_iata_code',
                        'dest_region',
                        'STATION',
                        'NAME',
                        'REPORT_TYPE',
                        'REM',
                        'BackupDirection',
                        'BackupDistanceUnit',
                        'BackupElements',
                        'BackupEquipment',
                        'BackupName',
                        'ORIGIN_AIRPORT_SEQ_ID',
                        'ORIGIN_CITY_MARKET_ID',
                        'ORIGIN_STATE_FIPS',
                        'ORIGIN_WAC',
                        'DEST_AIRPORT_SEQ_ID',
                        'DEST_CITY_MARKET_ID',
                        'DEST_STATE_FIPS',
                        'DEST_WAC',
                        'SOURCE',
                        'BackupElevation',
                        'LATITUDE',
                        'LONGITUDE',
                        'BackupLatitude',
                        'BackupLongitude',
                        'BackupDistance',
                        'QUARTER',
                        'OP_CARRIER_AIRLINE_ID',
                        'ORIGIN_AIRPORT_ID',
                        'DEST_AIRPORT_ID',
                        'CRS_DEP_TIME',
                        'CRS_ARR_TIME',
                        'dest_station_lat',
                        'dest_station_lon',
                        'sched_depart_date_time_UTC',
                        'Depart_2hrs_before',
                        'DATE',
                        'WindEquipmentChangeDate',
                        'two_hours_prior_depart_UTC',
                        'CRS_DEP_TIME_Str',
                        'CRS_ARR_TIME_Str',
                        'sch_dep_time_linux',
                        'origin_station_lon',
                        'origin_station_lat',
                        'HourlySkyConditions',
                        'HourlyPresentWeatherType']

df_60M = df_60M.drop(*cols_to_drop)

# COMMAND ----------

df_60M.cache()
display(df_60M)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checkpoint the Cleaned Data

# COMMAND ----------

# Get today's date in the desired format (e.g., YYYYMMDD)
today_str = datetime.today().strftime('%Y_%m_%d')

# Append today's date to the file path
df_60M.write.mode('overwrite').parquet(f"{team_blob_url}/feature_selection_{today_str}")

# COMMAND ----------

