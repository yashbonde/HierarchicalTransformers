# Data

[In part taken from paper]

We use ground sensor data instead of satellite images, one of the largest available datasets is from the National Meteorological Institute - Brazil. Historical data is available at from year 2000 to the day downloaded. [link](https://portal.inmet.gov.br/dadoshistoricos).

## Features

Each weather station has these features that are trained and predicted regressively:
```
total_precipitation(mm): amount of rainfall in past one hour
pressure(mB): pressure at the time of logging
max_pressure(mB): Maximum pressure in past one hour
min_pressure(mB): Maximum pressure in past one hour
radiation(KJ/m2): Radiation at the time of logging
temp(◦C): Temperature at the time of logging
max_temp(◦C): Maximum Temperature in past one hour
min_temp(◦C): Minimum Temperature in past one hour
max_dew(◦C): Maximum dew temperature in past one hour
min_dew(◦C): Minimum dew temperature in past one hour
humidity(%): Humidity percentage at the time of logging
max_humidity(%): Maximum humidity percentage in past one hour
min_humidity(%): Minimum humidity percentage in past one hour
wind_gust(m/s): Gust (maximum wind speed) in past one hour
wind_speed(m/s): Wind speed at the time of logging
wind_direction(◦deg): Wind direction in degrees8
```

For each station location metrics:
```
longitude(◦deg): longitude of recording weather station
latitude(◦deg): latitude of recording weather station
elevation(m): elevation from sea level of recording weather station
```

For each time step we have the features:
```
month: month at the time of recording [0-11]
day: day of recording [0-30]
hour: recording hour [0-23]
```

## Normalisation of features [TODO]

Features are normalised as shown below, all operations are keeping in mind that we will have to de-normalise the predictions from the model.
