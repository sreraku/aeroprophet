# Databricks notebook source
# MAGIC %md
# MAGIC ### Imports

# COMMAND ----------

!pip install altair

# COMMAND ----------

import pandas as pd
import numpy as np
import altair as alt
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
df_teamjoined_60M = spark.read.parquet(f"{team_blob_url}/df_FS_60M").cache() #joined data created by our team

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

df_60M = df_teamjoined_60M.drop(*[c for c in extra_cols_in_join if c not in ['Depart_3hrs_before']])

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's review the rest of the features to see what we have, how much data they are missing, and if they have the proper data types (SKIP NEXT CELL, DO NOT RERUN)

# COMMAND ----------

pd.set_option('display.max_rows', None)

try:
    df_col_analysis = pd.read_csv('pd_data/df_col_analysis.csv')
except:
    df_60M_row_ct = df_60M.count()
    df_col_analysis = df_60M.select( \
                        [(count(when(col(c).isNull(), c)) / df_60M_row_ct * 100).alias(c) for c in df_60M.columns]) \
                        .toPandas().T.rename(columns={0: 'Missing_Value_Percentage'}).reset_index().merge(pd.DataFrame(df_60M.dtypes, columns=['index', 'DataType']), on = 'index') \
                        .sort_values('Missing_Value_Percentage', ascending = False)
    df_col_analysis.to_csv('pd_data/df_col_analysis.csv', index = False)

df_col_analysis

# COMMAND ----------

# MAGIC %md
# MAGIC There are many fields that contain only missing-values. Namely, all of the monthly level and much of the daily weather data (https://www.ncei.noaa.gov/pub/data/cdo/documentation/LCD_documentation.pdf). This should not be an issue for our purposes given we want to predict at a much more granular level (2 hrs before departure). We can safely drop fields with greater than 99% missing values given then contain little to no information.

# COMMAND ----------

df_60M = df_60M.drop(*df_col_analysis[df_col_analysis.Missing_Value_Percentage >= 99]['index'].tolist())

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
# MAGIC ### Check for Duplicates

# COMMAND ----------


try:
    df_dups = pd.read_csv('pd_data/df_dups.csv')
except:
    total_ct = df_60M.count()
    distinct_ct = df_60M.distinct().count()

    df_dups = pd.DataFrame({
        'Metric': ['Total Row Count', 'Distinct Row Count', 'Duplicate Row Count'],
        'Value': [total_ct, distinct_ct, total_ct - distinct_ct]
    })

    df_dups.to_csv('pd_data/df_dups.csv', index = False)

df_dups


# COMMAND ----------

# MAGIC %md
# MAGIC Since each flight should only have a single record, let's drop the duplicates

# COMMAND ----------

df_60M = df_60M.dropDuplicates()

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

# MAGIC %md
# MAGIC ### Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC We currently only have a three hour before departure timestamp. Let's add a two hour before depature feature

# COMMAND ----------

df_60M = df_60M.withColumn("two_hours_prior_depart_UTC", expr("Depart_3hrs_before + INTERVAL 1 HOURS"))

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
    AND t1.two_hours_prior_depart_UTC > t2.sched_depart_date_time_UTC
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
    'WY': 'West'
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
                        'HourlyPresentWeatherType',
                        'HourlySkyConditions',
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
                        'DEP_DELAY',
                        'CRS_ARR_TIME',
                        'dest_station_lat',
                        'dest_station_lon',
                        'sched_depart_date_time_UTC',
                        'Depart_3hrs_before',
                        'DATE',
                        'WindEquipmentChangeDate',
                        'two_hours_prior_depart_UTC',
                        'CRS_DEP_TIME_Str',
                        'CRS_ARR_TIME_Str',
                        'sch_dep_time_linux',
                        'origin_station_lon',
                        'origin_station_lat']

df_60M = df_60M.drop(*cols_to_drop)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Handle Remaining Missing Outcome & String Values

# COMMAND ----------

# MAGIC %md
# MAGIC Remove rows where the outcome variable is null. Ensures we do not train our model on delays that never occured

# COMMAND ----------

df_60M = df_60M.na.drop(subset=["DEP_DEL15"])

# COMMAND ----------

# MAGIC %md
# MAGIC Fill remaining missing categorical values with an empty string

# COMMAND ----------

df_60M = df_60M.na.fill(value='')

# COMMAND ----------

# MAGIC %md
# MAGIC Note we have not dealt with missing numerical information. Mean impuation is taking considerable time. Let's checkpoint the data and reattempt with EDA

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checkpoint the Cleaned Data

# COMMAND ----------

# Get today's date in the desired format (e.g., YYYYMMDD)
today_str = datetime.today().strftime('%Y_%m_%d')

# Append today's date to the file path
df_60M.write.mode("overwrite").parquet(f"{team_blob_url}/feature_selection_{today_str}")