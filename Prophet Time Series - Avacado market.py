'''Predicting Avocado Prices Using Facebook Prophet'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
from fbprophet import Prophet

# Loading the dataset
avocado_df = pd.read_csv('avocado.csv')

# Exploring the dataset
print(avocado_df.head())
print(avocado_df.tail())

# sorting the dataframe with the date column
avocado_df = avocado_df.sort_values('Date')
print(avocado_df)

# Visualizing the dates vrs average price
plt.figure(figsize=(20,20))
plt.plot(avocado_df['Date'],avocado_df['AveragePrice'], color= 'red')
plt.xticks(rotation=90, size =3)
plt.show()

# Visualizing the region
plt.figure(figsize=(20,12))
sns.countplot(x= 'region', data= avocado_df)
plt.xticks(rotation= 90, size = 5)
plt.show()

# visualizing the year

plt.figure(figsize=(15,15))
sns.countplot(x='year', data=avocado_df)
plt.show()

"""
N/B: In this dataframe, i am concentrating on the date and average price for the prophet 
time series prediction
"""

avocado_prophet_df = avocado_df[['Date', 'AveragePrice']]
print(avocado_prophet_df)

# renaming the columns and making predictions
avocado_prophet_df = avocado_prophet_df.rename(columns= {'Date': 'ds', 'AveragePrice': 'y'})
print(avocado_prophet_df)

# Training our model
m = Prophet()
m.fit(avocado_prophet_df)

# forecasting the future
future = m.make_future_dataframe(periods=365) # Simulate the trend using the extrapolated generative model.
forecast = m.predict(future) # predict the future
print(forecast)

figure = m.plot(forecast, xlabel= 'Date', ylabel='Price')
plt.show()

figure = m.plot_components(forecast) # Plot the Prophet forecast components.
# Will plot whichever are available of: trend, holidays, weekly seasonality, and yearly seasonality.
plt.show()










