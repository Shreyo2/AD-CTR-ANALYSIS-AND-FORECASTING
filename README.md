## Data Preparation
import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.seasonal import seasonal_decompose

import matplotlib.pyplot as plt

data = pd.read_csv("/ctr.csv")

print(data.head())

data['date'].head()


data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')

data.set_index('date', inplace=True)

## Visualizing Clicks and Impressions
fig = go.Figure()

fig.add_trace(go.Scatter(x=data.index, y=data['Clicks'], mode='lines', name='Clicks'))

fig.add_trace(go.Scatter(x=data.index, y=data['Impressions'], mode='lines', name='Impressions'))

fig.update_layout(title='Clicks and Impressions Over Time')

fig.show()
## Creating a scatter plot to visualize the relationship between Clicks and Impressions
fig = px.scatter(data, x='Clicks', y='Impressions', title='Relationship Between Clicks and Impressions',
                 labels={'Clicks': 'Clicks', 'Impressions': 'Impressions'})

## Customizing the layout
fig.update_layout(xaxis_title='Clicks', yaxis_title='Impressions')

fig.show()

## Calculating and visualizing CTR
data['CTR'] = (data['Clicks'] / data['Impressions']) * 100

fig = px.line(data, x=data.index, y='CTR', title='Click-Through Rate (CTR) Over Time')

fig.show()

data['DayOfWeek'] = data.index.dayofweek

## EDA based on DayOfWeek
day_of_week_ctr = data.groupby('DayOfWeek')['CTR'].mean().reset_index()

day_of_week_ctr['DayOfWeek'] = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

fig = px.bar(day_of_week_ctr, x='DayOfWeek', y='CTR', title='Average CTR by Day of the Week')

fig.show()

## Creating a new column 'DayCategory' to categorize weekdays and weekends
data['DayCategory'] = data['DayOfWeek'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

## Calculating average CTR for weekdays and weekends
ctr_by_day_category = data.groupby('DayCategory')['CTR'].mean().reset_index()

## Creating a bar plot to compare CTR on weekdays vs. weekends
fig = px.bar(ctr_by_day_category, x='DayCategory', y='CTR', title='Comparison of CTR on Weekdays vs. Weekends',
             labels={'CTR': 'Average CTR'})

## Customizing the layout
fig.update_layout(yaxis_title='Average CTR')

fig.show()
## Grouping the data by 'DayCategory' and calculate the sum of Clicks and Impressions for each category
grouped_data = data.groupby('DayCategory')[['Clicks', 'Impressions']].sum().reset_index()

## Creating a grouped bar chart to visualize Clicks and Impressions on weekdays vs. weekends
fig = px.bar(grouped_data, x='DayCategory', y=['Clicks', 'Impressions'],
             title='Impressions and Clicks on Weekdays vs. Weekends',
             labels={'value': 'Count', 'variable': 'Metric'},
             color_discrete_sequence=['blue', 'green'])

## Customizing the layout
fig.update_layout(yaxis_title='Count')

fig.update_xaxes(title_text='Day Category')

fig.show()

data.reset_index(inplace=True)

from statsmodels.tsa.arima.model import ARIMA

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
## resetting index
time_series = data.set_index('date')['CTR']

## Differencing
differenced_series = time_series.diff().dropna()

## Plotting ACF and PACF of differenced time series
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(differenced_series, ax=axes[0])

plot_pacf(differenced_series, ax=axes[1])

plt.show()

from statsmodels.tsa.statespace.sarimax import SARIMAX

p, d, q, s = 1, 1, 1, 12

model = SARIMAX(time_series, order=(p, d, q), seasonal_order=(p, d, q, s))

results = model.fit()

print(results.summary())
## Predicting future values
future_steps = 100

predictions = results.predict(len(time_series), len(time_series) + future_steps - 1)

print(predictions)
## Creating a DataFrame with the original data and predictions
forecast = pd.DataFrame({'Original': time_series, 'Predictions': predictions})

## Plotting the original data and predictions
fig = go.Figure()

fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Predictions'],
                         mode='lines', name='Predictions'))

fig.add_trace(go.Scatter(x=forecast.index, y=forecast['Original'],
                         mode='lines', name='Original Data'))

fig.update_layout(title='CTR Forecasting',
                  xaxis_title='Time Period',
                  yaxis_title='Impressions',
                  legend=dict(x=0.1, y=0.9),
                  showlegend=True)

fig.show()
