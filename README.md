# Commodity-Forecasting---Global-Corn-Prices
Compares accuracy of statistical forecasting methods on global corn prices.


Uses models: Random walk with drift, ARIMA, ARIMAX, SARIMA and GARCH.


Findings:
The results indicate that, on average, a seasonal ARIMA is the best model for forecasting corn prices, followed by ARIMA-GARCH and ARIMA models. However, none of the models demonstrated exceptional performance with the best average RMSE for an 8-month forecast being $31.48. At this stage the models offer limited practical value for stakeholders given their performance, though there is potential for improvement through analysis of other models and variables.
No variables were found that consistently had a relationship with corn prices, but adding them showed potential. During in sample tests when using statically significant variables in an ARIMAX they achieved the lowest in sample error. However, there effectiveness was restricted as they performed poorly out of sample, partly due to compounding errors. To forecast of corn prices, the explanatory variables themselves need to be forecast, meaning that the RMSE of the corn price forecast is influenced by the RMSE of the variable forecast. 
For future research, it is recommended to prioritise identifying relationships with lagged variables. This approach reduces the reliance on forecasting of the explanatory variables, potentially yielding better results. Additionally, focusing on macro-scale variables may enhance the likelihood of finding statistically significant relationships. Companies such as LSEG offer agricultural commodities data packages that contain high quality and relevant data, though a subscription is required (LSEG, 2024). These packages include weather and crop forecasting data which result in better models as they capture a larger group than in this study. Given corns many uses it is a good candidate for a ridge machine learning model. This method could effectively handle a large number of correlated variables, which may lead to more accurate forecasts
