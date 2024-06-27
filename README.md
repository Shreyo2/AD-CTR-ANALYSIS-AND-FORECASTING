# Objective:
Ad CTR, or Click-Through Rate, is a crucial metric in online advertising, measuring the percentage of users who click on an ad after viewing it. It assesses ad effectiveness, indicating user engagement. A higher CTR often signifies a more compelling ad. For my project, understanding and optimizing Ad CTR will be vital for enhancing advertising strategies and maximizing campaign success in the digital landscape.

# Data Collection:
The dataset obtained contains information related to Ads CTR for a specific online advertising campaign. It consists of the following columns:

1.Date: The date on which the data was recorded.

2.Clicks: The number of times users clicked on the ads.

3.Impressions: The total number of times the ads were displayed to users.

4.CTR= (Clicks/Impressions)

# Analysis:

Upon scrutinizing clicks and impressions, we observe a linear relationship heightened impressions correlate with increased clicks, emphasizing the positive impact of ad visibility on user engagement. Our analysis shows:

- Visualizing the clicks and impressions over time we obtain greater fluctuations in impressions than that of in clicks over time.

- Visualizing the Click-Through Rate (CTR) over time yielded valuable insights showing a peak in Dec 2023 followed by a steep decrease in the CTR till July 2023 which was followed by a gradual yet slight increase till Oct 2023.

- Furthermore, we delve into examining the average CTR based on weekdays and weekends, aiming to uncover potential variations in user engagement patterns throughout the week.

- Additionally, contrasting impressions and clicks specifically on weekdays and weekends provides further insight into the effectiveness of ad campaigns across different timeframes, serving as a basis for strategic decision-making in advertising.

Visualizing the trend of the data, we can see that this data is not stationary, and it’s not appropriate to use the ARIMA model on such data. On such data, we can use the SARIMA model considering the seasonal nature of CTR, we determine the p, d, and q values for the Seasonal Autoregressive Integrated Moving Average (SARIMA) model. These parameters collectively define the non-seasonal part of the SARIMA model. This analytical approach enables us to anticipate CTR trends and contribute valuable insights for optimizing advertising strategies based on anticipated seasonal variations.

- The autoregressive (AR) order, denoted by p signifies the number of lagged values of the series that are used to predict the current value. The PACF plot cuts off at lag 1, indicating p=1.

- The moving average (MA) order, denoted by q captures the relationship between the current observation and the residual errors from previous forecasts. The ACF plot also cuts off at lag 1, indicating q=1.

- The value of d signifies the minimum number of differencing operations required to make the series stationary. Differencing involves subtracting the current observation from the previous one to remove trends or seasonal patterns. So d=1, because the data is non-stationary.

- The value of s in SARIMA represents the seasonal period or the number of time steps in each seasonal cycle. In our SARIMA model, the value of s is set to 12, indicating to a seasonal cycle of 12 months, suggesting that the data has yearly seasonality.

Forecasting As CTR is dependent on impressions and impressions change over time, we used Time Series forecasting techniques to forecast CTR. Upon training the forecasting model using SARIMA we obtain the following from the summary:

- Goodness of Fit: Measures like AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion) that assess the model's

- Goodness of fit. Lower values of AIC and BIC that is 152.730 and 172.048 respectively low values indicate very good fit.

- Diagnostic Tests: Measures like L-Jung Box test, Jarque-Bera test for white noise specifically checks for autocorrelation in the residuals of the model is 5.64 and 1.20 respectively low values indicate independent residuals of the time series model.

Then we predicted the future CTR values for the next 100 days which is a little more than 3 months and visualized it to view the forecasted trend of CTR .

- By plotting the original data and predictions we can expect a smooth gradual decrease with far lesser fluctuations in the number of impressions over Nov 2024 to Feb 2024 which ranges from 0.4 lakh to 0.35 lakh.

- A drastic decrease is expected in the number of impressions over time in Nov 2024 which is a little more than half of the number in the year 2023.

As a result of which CTR will also suffer a huge downfall.
In this way SARIMA extends ARIMA by including seasonal components to account for seasonality in the data as a result of which seasonal patterns were often observed in our time series data.

# Conclusion:
The analysis and forecasting of Click-Through Rate (CTR) using SARIMA modeling present a pivotal tool for businesses to gauge the effectiveness of their advertising endeavors and make informed decisions to enhance ad performance. Our implementation in Python demonstrates the capability to analyze historical CTR trends, develop SARIMA models, and forecast future CTR values.

By refining key components such as ar.L1, ma.L1, and ar.S.L12, we strive to improve the predictive accuracy of the SARIMA model, thereby enabling more precise forecasts and strategic planning. Ultimately, this project underscores the significance of CTR analysis in assessing the return on investment (ROI) of advertising campaigns and empowers businesses to optimize their marketing strategies through data-driven insights.
