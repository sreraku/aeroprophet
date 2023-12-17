# Databricks notebook source
# MAGIC %md
# MAGIC ## Resources & Notes
# MAGIC #### Delete this chunk before submission
# MAGIC - Description of what should be in each section below: https://digitalcampus.instructure.com/courses/14487/pages/phase-descriptions-and-deliverables?module_item_id=1711798
# MAGIC - This is the main notebook
# MAGIC   - There should be no code in this notebook
# MAGIC   - Include links in this notebook to separate coding notebooks
# MAGIC - Clone w261_final_project_starter_nb_fp BUT DON'T MODIFY IT DIRECTLY
# MAGIC   - https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/365886751420832/command/365886751420833
# MAGIC   - The starter notebook has everything you need to access the data. All data is accessible via Databricks and is stored on the Azure Cloud.
# MAGIC - Flights data dictionary: https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ
# MAGIC - Flight data
# MAGIC   - dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/  # All flights during 2015 to 2021 inclusive
# MAGIC   - dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_3m/   #2015 Q1 
# MAGIC   - dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_6m/   #2015 Q1 and Q2
# MAGIC   - dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_1y/    # 2019 (correct 2019!)
# MAGIC - Weather data dictionary p.8-12: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf
# MAGIC - Weather data
# MAGIC   - dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data/          #weather data for 2015-2021 
# MAGIC   - dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_1y/    #weather data for all of 2019
# MAGIC   - dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_3m/   #weather data for 2015 Q1
# MAGIC   - dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_6m/   #weather data for 2015 Q1 and Q2
# MAGIC - Airport data
# MAGIC   - dbfs:/mnt/mids-w261/datasets_final_project_2022/stations_data
# MAGIC - Airport code conversions
# MAGIC   - https://datahub.io/core/airport-codes
# MAGIC - We have rejoined joined Arrival Time Performance with local Weather  data. This produces the ATPW dataset. Please see Azure storage bucket via the Databricks starter notebook on how to access this data:
# MAGIC   - 3 months ATPW
# MAGIC   - 1 year ATPW
# MAGIC   - 3-5 years ATPW

# COMMAND ----------

# MAGIC %md
# MAGIC # Phase Summaries
# MAGIC
# MAGIC ### Phase 1
# MAGIC
# MAGIC Link to full Phase 1 document: https://adb-4248444930383559.19.azuredatabricks.net/?o=4248444930383559#notebook/930177584947455
# MAGIC
# MAGIC ### Phase 2
# MAGIC
# MAGIC ### Phase 3

# COMMAND ----------

# MAGIC %md
# MAGIC # Question Formulation

# COMMAND ----------

# MAGIC %md
# MAGIC # EDA & Discussion of Challenges

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC # Algorithm Exploration

# COMMAND ----------

# MAGIC %md
# MAGIC # Algorithm Implementation

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusions

# COMMAND ----------

