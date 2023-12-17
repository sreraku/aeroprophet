The ability to anticipate delays will usher in a new era of empowered decision-making for passengers and seamless operations for airlines.

Armed with this foresight, passengers will be able to adapt their plans in real-time and make more informed choices from the outset.

Meanwhile, airlines will be able to more proactively address issues and come up with timely resolutions if they're able to foresee delays. Beyond these immediate benefits, this predictive analysis will also shed light on the major causes of delays, which airlines can use to improve their operational efficiency and elevate the customer experience.

In a world that puts an increasing emphasis on efficiency, the power of real-time insights will transform the travel landscape.

Our model is built on publicly available data from the Bureau of Labor Statistics on US flights from 2015 to 2019, and weather station data from the National Center for Environmental Information during the same time period.

The 4 datasets are summarized in both tables below:

*Dataset	Source	Description*
Flights	Bureau of Labor Statistics	US flight data from 2015 to 2019, containing information on airlines, date, origin and destination airports, flight delay, time of departure, arrival. This is a historical data that can be used to figure out relevant features required to model to predict or classify if and when a flight is delayed.
Weather	National Center for Environmental Information	This dataset contains weather information including wind speed, dewpoint, visibility, elevation, humidity, precipitation, in hourly, weekly, and monthly intervals from weather stations located around the world.
Airport	DataHub	This dataset contains location, size, and identification information about US airports.
Weather Stations	DataHub	This dataset contains location and identification information about weather stations and their distance to nearby airports.
Dataset dimensions and memory requirements

Table	Rows	Columns	Memory (GB)
df_flights	74,177,433	109	2.93
df_weather	898,983,399	124	35.05
df_stations	5,004,169	12	1.3
df_airports	57,421	12	0.01
