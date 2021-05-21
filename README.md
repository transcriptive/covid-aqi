# Visualizing the impact of COVID-related transportation shutdowns on Air Quality

## Problem Statement

How did COVID-19 shutdowns affect air quality in Los Angeles and New York City? Use predictive models to forecast a “normal” 2020, and compare this to reality. Present findings to a group of environmentalists.

## Table of Contents

1. [EDA.ipynb](https://github.com/transcriptive/covid-aqi/blob/main/EDA.ipynb) — Initial Exploratory Data Analysis of the AQI and Traffic data
2. [covid-data-cleaning.ipynb](https://github.com/transcriptive/covid-aqi/blob/main/covid-data-cleaning.ipynb) — Data cleaning and preprocessing for the covid case information
3. [traffic-data-cleaning.ipynb](https://github.com/transcriptive/covid-aqi/blob/main/traffic-data-cleaning.ipynb) — Data cleaning and preprocessing for the traffic data
4. [datasets](https://github.com/transcriptive/covid-aqi/tree/main/datasets) — folder containing the datasets used during the modeling process
5. [LAX-notebooks](https://github.com/transcriptive/covid-aqi/tree/main/LAX-notebooks) — folder containing the notebooks used to process and predict the LA data
    1. [LA-aqi-data-preprocessing.ipynb](https://github.com/transcriptive/covid-aqi/blob/main/LAX-notebooks/LA-aqi-data-preprocessing.ipynb) — notebook containing the data cleaning and preprocessing for LA
    2. [LA-time-series-work.ipynb](https://github.com/transcriptive/covid-aqi/blob/main/LAX-notebooks/LA-time-series-work.ipynb) — notebook containing the process to create a Holt-Winters model for LA AQI
    3. [LA_time_series_work_colab.ipynb](https://github.com/transcriptive/covid-aqi/blob/main/LAX-notebooks/LA_time_series_work_colab.ipynb) — a fork of the LA-time-series-work notebook uploaded to colab, which was used to create an Auto ARIMA model and to export LA datasets
    4. [difference-between-prediction-and-actual.ipynb](https://github.com/transcriptive/covid-aqi/blob/main/LAX-notebooks/Difference-between-prediction-and-actual.ipynb) — notebook used to examine the differences between predicted and actual AQI values
6. [NYC-notebooks](https://github.com/transcriptive/covid-aqi/tree/main/NYC-notebooks) — folder containing the notebooks used to process and predict the NYC data
    1. [NYC-aqi-data-preprocessing.ipynb](https://github.com/transcriptive/covid-aqi/blob/main/NYC-notebooks/NYC-aqi-data-preprocessing.ipynb) — notebook containing the data cleaning and preprocessing for NYC
    2. [NYC-time-series-work.ipynb](https://github.com/transcriptive/covid-aqi/blob/main/NYC-notebooks/NYC-time-series-work.ipynb) — notebook containing the process to create a Holt-Winters model for NYC AQI
7. [predictions](https://github.com/transcriptive/covid-aqi/tree/main/predictions) — folder containing the predictions for LA and NYC AQI
    1. [lax_daily_predictions.csv](https://github.com/transcriptive/covid-aqi/blob/main/predictions/lax_daily_predictions.csv) — daily Holt Winters AQI predictions for LA
    2. [lax_weekly_predictions.csv](https://github.com/transcriptive/covid-aqi/blob/main/predictions/lax_weekly_predictions.csv) — weekly Auto ARIMA and Holt Winters AQI predictions for LA
    3. [nyc_daily_predictions.csv](https://github.com/transcriptive/covid-aqi/blob/main/predictions/nyc_daily_predictions.csv) — daily Holt Winters AQI predictions for NYC
    4. [nyc_weekly_predictions.csv](https://github.com/transcriptive/covid-aqi/blob/main/predictions/nyc_weekly_predictions.csv) — weekly Holt Winters AQI predictions for NYC
    5. [difference.csv](https://github.com/transcriptive/covid-aqi/blob/main/predictions/difference.csv) — the difference between predicted and actual Holt Winters and Auto ARIMA weekly AQI values for LA
8. [presentation.pdf](https://github.com/transcriptive/covid-aqi/blob/main/presentation.pdf) — a copy of the slide deck we presented from
9. scratch — a folder containing outdated notebooks not used for final modeling.


## Executive Summary

In 2019 a new highly infectious virus, which is now known as SARS-CoV-2, was identified. It originated in East Asia and quickly spread all over the world. The first case of COVID-19, the disease caused by SARS-CoV-2, was reported on January 21, 2020, and by March it became a problem big enough to declare a national emergency. New York and California were the first states to go on lockdown to prevent the spread of the virus, which drastically reduced the amount of car trips people took during the shutdown. In turn, this affected the air quality as transportation-related air pollution levels fell. 

For this project we visualized the AQI, COVID-19 and traffic data from LA county in 2020. Upon investigation, we saw a clear drop in air pollution in March at the top of the shutdown. For comparison, we then predicted what the air quality would have been in 2020 if the pandemic were never to happen. While Los Angeles experienced a drop in air pollution in March, we also noticed a clear spike in AQI in May, which could be attributed to the beginning of a record-setting fire season in California. 

Visualizing all the data together revealed some strong correlations between the three factors we explored.  There was a dramatic drop in car trips in Los Angeles county starting in March 2020, which is right around when Covid-related restrictions were put into place, also the month with the biggest difference in actual vs. predicted AQI. It is important to note that although these correlate, there are many more factors that went into the AQI index than just car trips/travel, and correlation does not always imply causation.

## Data Sources
### Air Quality 
[Air Quality Index Daily Values Report, Environmental Protection Agency](https://www.epa.gov/outdoor-air-quality-data/air-quality-index-daily-values-report)
* Daily AQI data per county, 2015-2020

### COVID-19 Data:
[Los Angeles, Los Angeles Times Data and Graphics Department
](https://github.com/datadesk/california-coronavirus-data) 
* Daily COVID-19 cases, deaths, and hospitalizations, January 2020 - present

[New York City, NYC Open Data](https://data.cityofnewyork.us/Health/COVID-19-Daily-Counts-of-Cases-Hospitalizations-an/rc75-m7u3) 
* Daily COVID-19 cases, deaths, and hospitalizations for all boroughs of NYC, February 2020 - present

### Wildfires
[CAL FIRE 2020 Incident Archive](https://www.fire.ca.gov/incidents/2020/)
* Database of California wildfires, 2020

### Traffic
[U.S. Department of Transportation, Bureau of Transportation Statistics](https://data.bts.gov/Research-and-Statistics/Trips-by-Distance-2020/dac6-p3ut)
* Total number of car trips grouped by distance and location, 2019-2021

## Data Cleaning
Data was cleaned by importing it into a jupyter lab notebook and performing necessary cleaning and processing. In the case of the Air Quality Index (AQI) data, six individual spreadsheets containing the daily AQI values for each county in the United states for a given year were downloaded, corresponding to 2015-2020.

### AQI
LA AQI data was prepared by combining the six years of information and selecting only the rows in the County column corresponding to Los Angeles county and exporting this daily information. This was later resampled to a weekly average for easier modeling.

NYC AQI data was prepared by combining the six years of information and selecting only the rows in the State column corresponding to New York State. This state-level data was then further filtered, by selecting only the rows in the County column which corresponded to New York, Kings, Queens, Bronx, and Richmond counties. We then took a mean of the daily data for the 5 counties to create an average daily AQI for New York City. This was later resampled to a weekly average for easier modeling. 

### COVID-19
California data was pulled from LA Times source, and contained COVID-19 data on cases for all counties. The data was filtered by counties to keep only Los Angeles county and resampled to contain data for the year of 2020. Once resampled, two NaNs for the row of the very first case were replaced as 0 for `new_confirmed_cases` and `new_deaths` features. New York Data was pulled from NYC Open Data and also was resampled to contain information on 2020 only.

### Traffic
The traffic data cleaning process consisted of downloading the county, state, and countrywide data from the data source. There were no null values and the data was all in the correct format, except the date. The index was set to be the column of dates after converting the column to datetime type. Then we created a new dataframe with just the data from Los Angeles County for the years 2019 through 2021. We exported and saved the dataframe as a csv file.


## Conclusion
The most noticeable drop in AQI occurred in Los Angeles, at the beginning of their lockdown period. New York City, with a much lower reliance on personal vehicles for transportation, did not see as many notable drops. What drops did occur were temporary, however, and the AQI rebounded to more normal levels as commercial traffic started up again. Additional forms of pollution, such as wildfires, have a significant additive property and are not accounted for by decreased levels of mobility during lockdown.

### Recommendations
Encourage people to drive less frequently, or to use public transportation, to decrease the AQI and increase overall air quality.
