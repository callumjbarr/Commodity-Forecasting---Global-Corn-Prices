#install.packages("fredr")
#install.packages("fpp2")
#install.packages("forecast")
#install.packages("TSstudio")
#install.packages("caret")
#install.packages("patchwork")
install.packages("prophet")

# Load the library
library(patchwork)
library(lubridate)
library(fredr)
library(fpp2)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(forecast)
library(CADFtest)
library(quantmod)
library(zoo)
library(rugarch)
library(lmtest)
library(broom)
library(TSstudio)
library(caret)
library(scales)
library(prophet)

################################################################################
# Data Import
################################################################################


wDir <- "C:/Users/cb241/OneDrive/Documents/1 - University/MAppFin/Semester B/ECONS543 - Applied Econometrics/Research/Data"
setwd(wDir)
dataDf <- read.csv("ECONS543 - Dataset.csv")
dataDf$date <- as.Date(dataDf$date)

testDF <- dataDf %>%
  select(ds = date, y = cornprice)

prophetmod <- prophet(testDF)

yHatdf = make_future_dataframe(prophetmod, periods = 6, freq = 'month')

yHat <- predict(prophetmod,yHatdf)

################################################################################
# Defining Dates
################################################################################

str(dataDf$date)

#date boundaries
startDate = c(1993,1)
startDateStr = as.Date("1993-01-01")
endDate = as.Date("2024-08-01")

#random date generator
#fdate4 <- sample(dataDf$date,1)

#different forecast start dates
fdate1 <- c(2003,9) #2003-09-01
fdate1Str <- as.Date("2003-09-01")
plotStart1 <- c(2003,10)
fdate1end <- c(2004,5)
fdate1endStr <- as.Date("2004-05-01")


fdate2 <- c(2014,1) #2014-01-01
fdate2Str <- as.Date("2014-01-01")
plotStart2 <- c(2014,2)
fdate2end <- c(2014,9)
fdate2endStr <- as.Date("2014-09-01")


fdate3 <- c(2021,10)#2021-10-01
fdate3Str <- as.Date("2021-10-01")
plotStart3 <- c(2021,11)
fdate3end <- c(2022,6)
fdate3endStr <- as.Date("2022-06-01")


fdate4 <- c(2023,8)#2023-08-01
fdate4Str <- as.Date("2023-08-01")
plotStart4 <- c(2023,9)
fdate4end <- c(2024,4)
fdate4endStr <- as.Date("2024-04-01")

#time series for autoplot
cornPriceTS <- ts(dataDf$cornprice, start = startDate, frequency = 12)
geoRiskTS <- ts(dataDf$GPRH, start = startDate, frequency = 12)
cornPriceVol <- ts(dataDf$monthlyreturn, start = startDate, frequency = 12)

#plot windows for autoplot graphs
plotWindow1 =  window(cornPriceTS, start = plotStart1, end = fdate1end)
plotWindow2 =  window(cornPriceTS, start = plotStart2, end = fdate2end)
plotWindow3 =  window(cornPriceTS, start = plotStart3, end = fdate3end)
plotWindow4 =  window(cornPriceTS, start = plotStart4, end = fdate4end)


################################################################################
# Time series data creation
################################################################################


#creating time series and different training periods
trainData1 <- window(cornPriceTS, start = startDate, end = fdate1)
trainData2 <- window(cornPriceTS, start = startDate, end = fdate2)
trainData3 <- window(cornPriceTS, start = startDate, end = fdate3)
trainData4 <- window(cornPriceTS, start = startDate, end = fdate4)

geoRiskData1 <- window(geoRiskTS, start = startDate, end = fdate1)
geoRiskData2 <- window(geoRiskTS, start = startDate, end = fdate2)
geoRiskData3 <- window(geoRiskTS, start = startDate, end = fdate3)
geoRiskData4 <- window(geoRiskTS, start = startDate, end = fdate4)


################################################################################
# Test if corn price is stationary
################################################################################


#Test for level stationarity
adfTest1 <- CADFtest(cornPriceTS, type = "drift", max.lag.y = 6)
print(adfTest1 $p.value)
#pvalue 0.2174 > 0.05, fail to reject null, data is non-stationary

#Test for trend stationarity
adfTest2 <- CADFtest(cornPriceTS, type = "trend", max.lag.y = 6)
print(adfTest2 $p.value)
#pvalue 0.2839 > 0.05, fail to reject null, data is non-stationary


################################################################################
# Test if geopolitical risk is stationary
################################################################################


#Test for level stationarity
GeoadfTest1 <- CADFtest(geoRiskData4, type = "drift", max.lag.y = 6)
print(GeoadfTest1 $p.value)
#pvalues for geopolitical risk < 0.05, reject null, data is stationary


################################################################################
# Forecast models
################################################################################


#random walk with drift
rwD1 <- rwf(trainData1, drift = TRUE, h = 8)
rwD2 <- rwf(trainData2, drift = TRUE, h = 8)
rwD3 <- rwf(trainData3, drift = TRUE, h = 8)
rwD4 <- rwf(trainData4, drift = TRUE, h = 8)

testWindow1 <- window(cornPriceTS, start = fdate1)
testWindow2 <- window(cornPriceTS, start = fdate2)
testWindow3 <- window(cornPriceTS, start = fdate3)
testWindow4 <- window(cornPriceTS, start = fdate4)

#Accuracy test
accuracy(rwD1, testWindow1)
accuracy(rwD2, testWindow2)
accuracy(rwD3, testWindow3)
accuracy(rwD4, testWindow4)

summary(rwD1)

#ARIMA models
arima1 <- auto.arima(trainData1, seasonal = FALSE, stepwise = FALSE, approximation = FALSE)
arimaf1 <- forecast(arima1, h = 8)

arima2 <- auto.arima(trainData2, seasonal = FALSE, stepwise = FALSE, approximation = FALSE)
arimaf2 <- forecast(arima2, h = 8)

arima3 <- auto.arima(trainData3, seasonal = FALSE, stepwise = FALSE, approximation = FALSE)
arimaf3 <- forecast(arima3, h = 8)

arima4 <- auto.arima(trainData4, seasonal = FALSE, stepwise = FALSE, approximation = FALSE)
arimaf4 <- forecast(arima4, h = 8)

#ARIMA test
#H0: Residuals have no autocorrelation, HA: Residuals have autocorrelation
checkresiduals(arima1)
#pvalue > 0.05, accept null

checkresiduals(arima2)
#pvalue < 0.05, reject null, however acf plot seems ok with only a few later lags outside threshold

checkresiduals(arima3)
#pvalue > 0.05, accept null

checkresiduals(arima4)
#pvalue > 0.05, accept null


#Accuracy test
accuracy(arimaf1, testWindow1)
accuracy(arimaf2, testWindow2)
accuracy(arimaf3, testWindow3)
accuracy(arimaf4, testWindow4)
summary(arimaf2)


##seasonal ARIMA models
arimaS1 <- auto.arima(trainData1, seasonal = TRUE, stepwise = FALSE, approximation = FALSE)
arimaSf1 <- forecast(arimaS1, h = 8)

arimaS2 <- auto.arima(trainData2, seasonal = TRUE, stepwise = FALSE, approximation = FALSE)
arimaSf2 <- forecast(arimaS2, h = 8)

arimaS3 <- auto.arima(trainData3, seasonal = TRUE, stepwise = FALSE, approximation = FALSE)
arimaSf3 <- forecast(arimaS3, h = 8)

arimaS4 <- auto.arima(trainData4, seasonal = TRUE, stepwise = FALSE, approximation = FALSE)
arimaSf4 <- forecast(arimaS4, h = 8)

#ARIMA test
#H0: Residuals have no autocorrelation, HA: Residuals have autocorrelation
checkresiduals(arimaS1)
#pvalue > 0.05, accept null

checkresiduals(arimaS2)
#pvalue > 0.05, accept null

checkresiduals(arimaS3)
#pvalue < 0.05, reject null, however acf plot seems ok with only a few later lags outside threshold

checkresiduals(arimaS4)
#pvalue > 0.05, accept null

#Accuracy test
accuracy(arimaSf1, testWindow1)
accuracy(arimaSf2, testWindow2)
accuracy(arimaSf3, testWindow3)
accuracy(arimaSf4, testWindow4)
summary(arimaSf4)

################################################################################
# Adding X Variables
################################################################################
XDf <- as.matrix(dataDf %>% 
                    select(ChinaEI, ChinaEI_L1, ChinaEI_L2, ChinaEI_L3, ChinaEI_L4, 
                           GPRH, GPRH_L1,	GPRH_L2,	GPRH_L3,	GPRH_L4, Npricertn,	
                           NpricertnL1,	NpricertnL2,	NpricertnL3,	NpricertnL4,	
                           AveTempDev,	AveRainDev) %>%
                    filter(dataDf$date >= startDateStr & dataDf$date <= endDate))

#check China Import/Export is stationary
XadfTest1 <- CADFtest(XDf[,1], type = "drift", max.lag.y = 6)
print(XadfTest1$p.value)
plot(XDf[,1])
#p-value 0.0045 < 0.05, reject H0, It is staionary

#check GPRH is stationary
XadfTest2 <- CADFtest(XDf[,6], type = "drift", max.lag.y = 6)
print(XadfTest2$p.value)
plot(XDf[,6])
#p-value 0.00047 < 0.05, reject H0, It is staionary

#Nitrogen price rtn
XadfTest3 <- CADFtest(XDf[,11], type = "drift", max.lag.y = 6)
print(XadfTest3$p.value)
plot(XDf[,11])
#p-value < 0.05, reject H0, is stationary

#check TempDev is stationary
XadfTest4 <- CADFtest(XDf[,16], type = "drift", max.lag.y = 6)
print(XadfTest4$p.value)
plot(XDf[,16])
#p-value < 0.05, reject H0, It is staionary

#check RainDev is stationary
XadfTest5 <- CADFtest(XDf[,17], type = "drift", max.lag.y = 6)
print(XadfTest5$p.value)
plot(XDf[,17])
#p-value < 0.05, reject H0, It is staionary

#final model for training 1
XDf1 <- as.matrix(dataDf %>% 
                    select(ChinaEI, ChinaEI_L1, ChinaEI_L2, ChinaEI_L3, ChinaEI_L4) %>%
                    filter(dataDf$date >= startDateStr & dataDf$date <= fdate1Str))


#final model for training 2
XDf2 <- as.matrix(dataDf %>% 
                    select(Npricertn, NpricertnL1, NpricertnL2) %>%
                    filter(dataDf$date >= startDateStr & dataDf$date <= fdate2Str))

#Model does not improve arima, therefore ignore
XDf3 <- as.matrix(dataDf %>% 
                    select(NpricertnL1) %>%
                    filter(dataDf$date >= startDateStr & dataDf$date <= fdate3Str))

#Model does not improve arima, therefore ignore
XDf4 <- as.matrix(dataDf %>% 
                    select(Npricertn, NpricertnL1) %>%
                    filter(dataDf$date >= startDateStr & dataDf$date <= fdate4Str))

#fitting ARIMA with X
arimaXs1 <- auto.arima(trainData1, seasonal = TRUE, stepwise = FALSE, approximation = FALSE,
                        xreg = XDf1)
arimaXs2 <- auto.arima(trainData2, seasonal = TRUE, stepwise = FALSE, approximation = FALSE,
                        xreg = XDf2)
arimaXs3 <- auto.arima(trainData3, seasonal = TRUE, stepwise = FALSE, approximation = FALSE,
                        xreg = XDf3)
arimaXs4 <- auto.arima(trainData4, seasonal = TRUE, stepwise = FALSE, approximation = FALSE,
                        xreg = XDf4)

summary(arimaXs1)
coeftest(arimaXs1)
checkresiduals(arimaXs1)

#ARIMA test
#H0: Residuals have no autocorrelation, HA: Residuals have autocorrelation
checkresiduals(arimaXs1)
#pvalue > 0.05, accept null

checkresiduals(arimaXs2)
#pvalue > 0.05, accept null

checkresiduals(arimaXs3)
#pvalue < 0.05, reject null, however acf plot seems ok with only a few later lags outside threshold

checkresiduals(arimaXs4)
#pvalue > 0.05, accept null

#create forecasts of X variables required
#forecast for China export:import
ChinaEImodel <- auto.arima(XDf1[,1], seasonal = TRUE, stepwise = FALSE, approximation = FALSE)
ChinaEIforecast <- data.frame(forecast(ChinaEImodel, h = 8))
checkresiduals(ChinaEImodel)

#first lag value is equal to last actual value then the point forecast
Lag1ChinaEI <- lag(ChinaEIforecast$Point.Forecast)
Lag1ChinaEI[1] <- XDf1[nrow(XDf1), 1]
#similar but inludes 2 know values etc for lags 3 and 4
Lag2ChinaEI <- lag(ChinaEIforecast$Point.Forecast, n = 2)
Lag2ChinaEI[1] <- XDf1[nrow(XDf1)-1, 1]
Lag2ChinaEI[2] <- XDf1[nrow(XDf1), 1]

Lag3ChinaEI <- lag(ChinaEIforecast$Point.Forecast, n = 3)
Lag3ChinaEI[1] <- XDf1[nrow(XDf1)-2, 1]
Lag3ChinaEI[2] <- XDf1[nrow(XDf1)-1, 1]
Lag3ChinaEI[3] <- XDf1[nrow(XDf1), 1]

Lag4ChinaEI <- lag(ChinaEIforecast$Point.Forecast, n = 4)
Lag4ChinaEI[1] <- XDf1[nrow(XDf1)-3, 1]
Lag4ChinaEI[2] <- XDf1[nrow(XDf1)-2, 1]
Lag4ChinaEI[3] <- XDf1[nrow(XDf1)-1, 1]
Lag4ChinaEI[4] <- XDf1[nrow(XDf1), 1]

#china Export:import and lags df
Xs1forecast <- data.frame(ChinaEIf = ChinaEIforecast$Point.Forecast, 
                          ChinaEIfL1 = Lag1ChinaEI, ChinaEIfL2 = Lag2ChinaEI, ChinaEIfL3 = Lag3ChinaEI,
                          ChinaEIfL4 = Lag4ChinaEI)

#Forecast of nitrogen price
Npricertnmodel <- auto.arima(XDf2[,1], seasonal = TRUE, stepwise = FALSE, approximation = FALSE)
Npricertnforecast <- data.frame(forecast(Npricertnmodel, h = 8))
checkresiduals(Npricertnmodel)

#first lag value is equal to last actual value then the point forecast
Lag1Npricertn <- lag(Npricertnforecast$Point.Forecast)
Lag1Npricertn[1] <- XDf2[nrow(XDf2), 1]

#similar but inludes 2 known values
Lag2Npricertn <- lag(Npricertnforecast$Point.Forecast, n = 2)
Lag2Npricertn[1] <- XDf2[nrow(XDf2)-1, 1]
Lag2Npricertn[2] <- XDf2[nrow(XDf2), 1]

#nprice return and lags df
Xs2forecast <- data.frame(Npricertn = Npricertnforecast$Point.Forecast, 
                          NpricertnfL1 = Lag1Npricertn, NpricertnfL2 = Lag2Npricertn)


#Test 1 forecast with x variables
test1arimaxf <- forecast(arimaXs1, h = 8, xreg = as.matrix(Xs1forecast))
summary(test1arimaxf)
accuracy(test1arimaxf, testWindow1)
checkresiduals(test1arimaxf)


#Test 2 forecast with x variables
test2arimaxf <- forecast(arimaXs2, h = 8, xreg = as.matrix(Xs2forecast))
summary(test2arimaxf)
accuracy(test2arimaxf, testWindow2)
checkresiduals(test2arimaxf)


################################################################################
# GARCH return model - apply to last price for forecast
################################################################################


#corn price monthly return
monthlyReturnTS <- ts(dataDf$monthlyreturn, start = startDate, frequency = 12)
monthlyReturnTS <- na.omit(monthlyReturnTS)

#creating training data sets
RtrainData1 <- window(monthlyReturnTS, start = startDate, end = fdate1)
RtrainData2 <- window(monthlyReturnTS, start = startDate, end = fdate2)
RtrainData3 <- window(monthlyReturnTS, start = startDate, end = fdate3)
RtrainData4 <- window(monthlyReturnTS, start = startDate, end = fdate4)

#getting arima orders
GarchArima1 <- auto.arima(RtrainData1, seasonal = FALSE, stepwise = FALSE, approximation = FALSE)
summary(GarchArima1)
#GarchArima1(0,0,1)

GarchArima2 <- auto.arima(RtrainData2, seasonal = FALSE, stepwise = FALSE, approximation = FALSE)
summary(GarchArima2)
#GarchArima2(3,0,2)

GarchArima3 <- auto.arima(RtrainData3, seasonal = FALSE, stepwise = FALSE, approximation = FALSE)
summary(GarchArima3)
#GarchArima3(3,0,2)

GarchArima4 <- auto.arima(RtrainData4, seasonal = FALSE, stepwise = FALSE, approximation = FALSE)
summary(GarchArima4)
#GarchArima4(3,0,2)

#garch model spec
gspec1 = ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
                    mean.model = list(armaOrder = c(0,1)))

gspec2 = ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
                    mean.model = list(armaOrder = c(3,2)))

gspec3 = ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
                    mean.model = list(armaOrder = c(3,2)))

gspec4 = ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
                    mean.model = list(armaOrder = c(3,2)))

#garch model
gmod1 = ugarchfit(spec = gspec1, data = RtrainData1)
gmod2 = ugarchfit(spec = gspec2, data = RtrainData2)
gmod3 = ugarchfit(spec = gspec3, data = RtrainData3)
gmod4 = ugarchfit(spec = gspec4, data = RtrainData4)
#model checks
show(gmod1)
show(gmod2)
show(gmod3)
show(gmod4)

#garch forecast - returns
gmodf1 = ugarchforecast(gmod1, n.ahead = 8)
gmodf2 = ugarchforecast(gmod2, n.ahead = 8)
gmodf3 = ugarchforecast(gmod3, n.ahead = 8)
gmodf4 = ugarchforecast(gmod4, n.ahead = 8)

#Garch forecast to df
gmodreturns1 <- as.data.frame(gmodf1@forecast$seriesFor)
colnames(gmodreturns1) <- "Gforecast1"

gmodreturns2 <- as.data.frame(gmodf2@forecast$seriesFor)
colnames(gmodreturns2) <- "Gforecast2"

gmodreturns3 <- as.data.frame(gmodf3@forecast$seriesFor)
colnames(gmodreturns3) <- "Gforecast3"

gmodreturns4 <- as.data.frame(gmodf4@forecast$seriesFor)
colnames(gmodreturns4) <- "Gforecast4"

#last price in training data
lastpricetrain1 = as.numeric(tail(trainData1, n=1))
lastpricetrain2 = as.numeric(tail(trainData2, n=1))
lastpricetrain3 = as.numeric(tail(trainData3, n=1))
lastpricetrain4 = as.numeric(tail(trainData4, n=1))

#define vector
gmodPrice1 <- numeric(nrow(gmodreturns1))
gmodPrice1[1] <- lastpricetrain1 * (1+gmodreturns1$Gforecast1[1])

gmodPrice2 <- numeric(nrow(gmodreturns2))
gmodPrice2[1] <- lastpricetrain2 * (1+gmodreturns2$Gforecast2[1])

gmodPrice3 <- numeric(nrow(gmodreturns3))
gmodPrice3[1] <- lastpricetrain3 * (1+gmodreturns3$Gforecast3[1])

gmodPrice4 <- numeric(nrow(gmodreturns3))
gmodPrice4[1] <- lastpricetrain4 * (1+gmodreturns4$Gforecast4[1])

#add to vector
for (i in 2:nrow(gmodreturns1)){
  gmodPrice1[i] <- gmodPrice1[i-1]*(1 + gmodreturns1$Gforecast1[i])
}

for (i in 2:nrow(gmodreturns2)){
  gmodPrice2[i] <- gmodPrice2[i-1]*(1 + gmodreturns2$Gforecast2[i])
}

for (i in 2:nrow(gmodreturns3)){
  gmodPrice3[i] <- gmodPrice3[i-1]*(1 + gmodreturns3$Gforecast3[i])
}

for (i in 2:nrow(gmodreturns4)){
  gmodPrice4[i] <- gmodPrice4[i-1]*(1 + gmodreturns4$Gforecast4[i])
}

head(gmodPrice1)
tail(gmodPrice1)

#setting forecast start date
fdate1mod <- fdate1
if (fdate1[2] == 12) {
  fdate1mod[2] <- 1  # Reset month to January
  fdate1mod[1] <- fdate1[1] + 1  # Increment year
} else {
  fdate1mod[2] <- fdate1[2] + 1  # Just add 1 to the month
}

fdate2mod <- fdate2
if (fdate2[2] == 12) {
  fdate2mod[2] <- 1  # Reset month to January
  fdate2mod[1] <- fdate2[1] + 1  # Increment year
} else {
  fdate2mod[2] <- fdate2[2] + 1  # Just add 1 to the month
}

fdate3mod <- fdate3
if (fdate3[2] == 12) {
  fdate3mod[2] <- 1  # Reset month to January
  fdate3mod[1] <- fdate3[1] + 1  # Increment year
} else {
  fdate3mod[2] <- fdate3[2] + 1  # Just add 1 to the month
}

fdate4mod <- fdate4
if (fdate4[2] == 12) {
  fdate4mod[2] <- 1  # Reset month to January
  fdate4mod[1] <- fdate4[1] + 1  # Increment year
} else {
  fdate4mod[2] <- fdate4[2] + 1  # Just add 1 to the month
}

#Making ts of garch forecasts
gmodPriceF1 <- ts(gmodPrice1, start = fdate1mod, frequency = 12)
gmodPriceF2 <- ts(gmodPrice2, start = fdate2mod, frequency = 12)
gmodPriceF3 <- ts(gmodPrice3, start = fdate3mod, frequency = 12)
gmodPriceF4 <- ts(gmodPrice4, start = fdate4mod, frequency = 12)


#Garch test 1 forecast error
garcherror1 <- dataDf %>% 
  select(cornprice) %>%
  filter(dataDf$date >= "2003-10-01" & dataDf$date <= fdate1endStr) %>%
  mutate(garchfprice = gmodPrice1)

GarRMSEtest1 <- RMSE(pred = garcherror1$garchfprice, obs = garcherror1$cornprice)
GarMAEtest1 <- MAE(pred = garcherror1$garchfprice, obs = garcherror1$cornprice)
print(GarRMSEtest1)
print(GarMAEtest1)

#Garch test 2 forecast error
garcherror2 <- dataDf %>% 
  select(cornprice) %>%
  filter(dataDf$date >= "2014-02-01" & dataDf$date <= fdate2endStr) %>%
  mutate(garchfprice = gmodPrice2)

GarRMSEtest2 <- RMSE(pred = garcherror2$garchfprice, obs = garcherror2$cornprice)
GarMAEtest2 <- MAE(pred = garcherror2$garchfprice, obs = garcherror2$cornprice)
print(GarRMSEtest2)
print(GarMAEtest2)

#Garch test 3 forecast error
garcherror3 <- dataDf %>% 
  select(cornprice) %>%
  filter(dataDf$date >= "2021-11-01" & dataDf$date <= fdate3endStr) %>%
  mutate(garchfprice = gmodPrice3)

GarRMSEtest3 <- RMSE(pred = garcherror3$garchfprice, obs = garcherror3$cornprice)
GarMAEtest3 <- MAE(pred = garcherror3$garchfprice, obs = garcherror3$cornprice)
print(GarRMSEtest3)
print(GarMAEtest3)

#Garch test 4 forecast error
garcherror4 <- dataDf %>% 
  select(cornprice) %>%
  filter(dataDf$date >= "2023-09-01" & dataDf$date <= fdate4endStr) %>%
  mutate(garchfprice = gmodPrice4)

GarRMSEtest4 <- RMSE(pred = garcherror4$garchfprice, obs = garcherror4$cornprice)
GarMAEtest4 <- MAE(pred = garcherror4$garchfprice, obs = garcherror4$cornprice)
print(GarRMSEtest4)
print(GarMAEtest4)


################################################################################
# Plotting forecasts
################################################################################

#Find volatility thresholds for 10% highs and lowss
highvolThres <- quantile(dataDf$monthlyreturn, 0.9, na.rm = TRUE)
lowvolThres <- quantile(dataDf$monthlyreturn, 0.1, na.rm = TRUE)

#Plot of forecast date 1


plot1 <- dataDf %>% 
  select(date, cornprice, monthlyreturn) %>%
  filter(dataDf$date >= "2003-07-01" & dataDf$date <= fdate1endStr) %>%
  mutate(randomwalkF = c(rep(NA,3), head(rwD1$mean, n = n()-3))) %>%
  mutate(arimaF = c(rep(NA,3), head(arimaf1$mean, n = n()-3))) %>%
  mutate(arimaSF = c(rep(NA,3), head(arimaSf1$mean, n = n()-3))) %>%
  mutate(garchF = c(rep(NA,3), head(garcherror1$garchfprice, n = n()-3))) %>% 
  mutate(arimaxSF = c(rep(NA,3), head(test1arimaxf$mean, n = n()-3))) %>%
  mutate(vol = ifelse(monthlyreturn > highvolThres | monthlyreturn < lowvolThres, 1, 0))
 
volPlot1 <- plot1 %>%
  filter(vol == 1) %>%             
  select(x1 = date) %>%         
  mutate(min = -Inf, max = Inf)     
  
volPlot1 <- volPlot1 %>%
  mutate(x2 = x1 + months(1))
  
ggplot() +
  theme_minimal() +
  theme(panel.grid.minor = element_blank(),
        axis.line = element_line(color = "gray"),   # Adds borders to x and y axes
        axis.ticks = element_line(color = "gray")) +
  
  # Lines for the different forecast models
  geom_line(data = plot1, aes(x = date, y = cornprice, color = "Observed")) +  
  geom_line(data = plot1 %>% filter(!is.na(randomwalkF)), aes(x = date, y = randomwalkF, color = "Random Walk With Drift")) +
  geom_line(data = plot1 %>% filter(!is.na(arimaF)), aes(x = date, y = arimaF, color = "ARIMA")) +
  geom_line(data = plot1 %>% filter(!is.na(arimaSF)), aes(x = date, y = arimaSF, color = "SARIMA")) +
  geom_line(data = plot1 %>% filter(!is.na(garchF)), aes(x = date, y = garchF, color = "GARCH-ARIMA")) +
  geom_line(data = plot1 %>% filter(!is.na(arimaxSF)), aes(x = date, y = arimaxSF, color = "SARIMAX")) +
  
  # Shaded area 
  geom_rect(data = volPlot1, aes(xmin = x1, xmax = x2, ymin = -Inf, ymax = Inf, fill = "High Volatility"),
            alpha = 0.2) +
  
  # Manual color scale for the lines
  scale_color_manual(name = "Forecast Models", 
                     values = c("Observed" = "black", 
                                "Random Walk With Drift" = "#688ae8", 
                                "ARIMA" = "#c33d69", 
                                "SARIMA" = "#2ea597", 
                                "GARCH-ARIMA" = "#8456ce", 
                                "SARIMAX" = "#e07941")) +
  
  # Manual fill scale for the shaded area
  scale_fill_manual(values = c("High Volatility" = "grey"), guide = guide_legend(title = NULL)) +
  
  labs(title = "Corn Price Forecast - Test 1",
       x = NULL,
       y = "Corn Price") +
  scale_x_date(date_breaks = "1 month", date_labels = "%b %y", limits = c(min(plot1$date), max(plot1$date)), expand = c(0,0)) +
  scale_y_continuous(labels = label_dollar()) +
  
  # Guides for legends
  guides(color = guide_legend(title = NULL))
  
  #Plot of forecast date 2

plot2 <- dataDf %>% 
  select(date, cornprice, monthlyreturn) %>%
  filter(dataDf$date >= "2013-11-01" & dataDf$date <= fdate2endStr) %>%
  mutate(randomwalkF = c(rep(NA,3), head(rwD2$mean, n = n()-3))) %>%
  mutate(arimaF = c(rep(NA,3), head(arimaf2$mean, n = n()-3))) %>%
  mutate(arimaSF = c(rep(NA,3), head(arimaSf2$mean, n = n()-3))) %>%
  mutate(garchF = c(rep(NA,3), head(garcherror2$garchfprice, n = n()-3))) %>% 
  mutate(arimaxSF = c(rep(NA,3), head(test2arimaxf$mean, n = n()-3))) %>%
  mutate(vol = ifelse(monthlyreturn > highvolThres | monthlyreturn < lowvolThres, 1, 0))

volPlot2 <- plot2 %>%
  filter(vol == 1) %>%             
  select(x1 = date) %>%         
  mutate(min = -Inf, max = Inf)     

volPlot2 <- volPlot2 %>%
  mutate(x2 = x1 + months(1))

volPlot2 <- volPlot2 %>%
  filter(!(row_number() == n() & x2 > tail(plot2$date,1)))


ggplot() +
  theme_minimal() +
  theme(panel.grid.minor = element_blank(),
        axis.line = element_line(color = "gray"),   # Adds borders to x and y axes
        axis.ticks = element_line(color = "gray")) +
  
  # Lines for the different forecast models
  geom_line(data = plot2, aes(x = date, y = cornprice, color = "Observed")) +  
  geom_line(data = plot2 %>% filter(!is.na(randomwalkF)), aes(x = date, y = randomwalkF, color = "Random Walk With Drift")) +
  geom_line(data = plot2 %>% filter(!is.na(arimaF)), aes(x = date, y = arimaF, color = "ARIMA")) +
  geom_line(data = plot2 %>% filter(!is.na(arimaSF)), aes(x = date, y = arimaSF, color = "SARIMA")) +
  geom_line(data = plot2 %>% filter(!is.na(garchF)), aes(x = date, y = garchF, color = "GARCH-ARIMA")) +
  geom_line(data = plot2 %>% filter(!is.na(arimaxSF)), aes(x = date, y = arimaxSF, color = "SARIMAX")) +
  
  # Shaded area 
  geom_rect(data = volPlot2, aes(xmin = x1, xmax = x2, ymin = -Inf, ymax = Inf, fill = "High Volatility"),
            alpha = 0.2) +
  
  # Manual color scale for the lines
  scale_color_manual(name = "Forecast Models", 
                     values = c("Observed" = "black", 
                                "Random Walk With Drift" = "#688ae8", 
                                "ARIMA" = "#c33d69", 
                                "SARIMA" = "#2ea597", 
                                "GARCH-ARIMA" = "#8456ce", 
                                "SARIMAX" = "#e07941")) +
  
  # Manual fill scale for the shaded area
  scale_fill_manual(values = c("High Volatility" = "grey"), guide = guide_legend(title = NULL)) +
  
  labs(title = "Corn Price Forecast - Test 2",
       x = NULL,
       y = "Corn Price") +
  scale_x_date(date_breaks = "1 month", date_labels = "%b %y", limits = c(min(plot2$date), max(plot2$date)), expand = c(0,0)) +
  scale_y_continuous(labels = label_dollar()) +
  
  # Guides for legends
  guides(color = guide_legend(title = NULL))



#Plot of forecast date 3

plot3 <- dataDf %>% 
  select(date, cornprice, monthlyreturn) %>%
  filter(dataDf$date >= "2021-08-01" & dataDf$date <= fdate3endStr) %>%
  mutate(randomwalkF = c(rep(NA,3), head(rwD3$mean, n = n()-3))) %>%
  mutate(arimaF = c(rep(NA,3), head(arimaf3$mean, n = n()-3))) %>%
  mutate(arimaSF = c(rep(NA,3), head(arimaSf3$mean, n = n()-3))) %>%
  mutate(garchF = c(rep(NA,3), head(garcherror3$garchfprice, n = n()-3))) %>% 
  mutate(vol = ifelse(monthlyreturn > highvolThres | monthlyreturn < lowvolThres, 1, 0))

volPlot3 <- plot3 %>%
  filter(vol == 1) %>%             
  select(x1 = date) %>%         
  mutate(min = -Inf, max = Inf)     

volPlot3 <- volPlot3 %>%
  mutate(x2 = x1 + months(1))

volPlot3 <- volPlot3 %>%
  filter(!(row_number() == n() & x2 > tail(plot3$date,1)))


ggplot() +
  theme_minimal() +
  theme(panel.grid.minor = element_blank(),
        axis.line = element_line(color = "gray"),   # Adds borders to x and y axes
        axis.ticks = element_line(color = "gray")) +
  
  # Lines for the different forecast models
  geom_line(data = plot3, aes(x = date, y = cornprice, color = "Observed")) +  
  geom_line(data = plot3 %>% filter(!is.na(randomwalkF)), aes(x = date, y = randomwalkF, color = "Random Walk With Drift")) +
  geom_line(data = plot3 %>% filter(!is.na(arimaF)), aes(x = date, y = arimaF, color = "ARIMA")) +
  geom_line(data = plot3 %>% filter(!is.na(arimaSF)), aes(x = date, y = arimaSF, color = "SARIMA")) +
  geom_line(data = plot3 %>% filter(!is.na(garchF)), aes(x = date, y = garchF, color = "GARCH-ARIMA")) +
 
  # Shaded area 
  geom_rect(data = volPlot3, aes(xmin = x1, xmax = x2, ymin = -Inf, ymax = Inf, fill = "High Volatility"),
            alpha = 0.2) +
  
  # Manual color scale for the lines
  scale_color_manual(name = "Forecast Models", 
                     values = c("Observed" = "black", 
                                "Random Walk With Drift" = "#688ae8", 
                                "ARIMA" = "#c33d69", 
                                "SARIMA" = "#2ea597", 
                                "GARCH-ARIMA" = "#8456ce")) +
  
  # Manual fill scale for the shaded area
  scale_fill_manual(values = c("High Volatility" = "grey"), guide = guide_legend(title = NULL)) +
  
  labs(title = "Corn Price Forecast - Test 3",
       x = NULL,
       y = "Corn Price") +
  scale_x_date(date_breaks = "1 month", date_labels = "%b %y", limits = c(min(plot3$date), max(plot3$date)), expand = c(0,0)) +
  scale_y_continuous(labels = label_dollar()) +
  
  # Guides for legends
  guides(color = guide_legend(title = NULL))



#Plot of forecast date 4

plot4 <- dataDf %>% 
  select(date, cornprice, monthlyreturn) %>%
  filter(dataDf$date >= "2023-06-01" & dataDf$date <= fdate4endStr) %>%
  mutate(randomwalkF = c(rep(NA,3), head(rwD4$mean, n = n()-3))) %>%
  mutate(arimaF = c(rep(NA,3), head(arimaf4$mean, n = n()-3))) %>%
  mutate(arimaSF = c(rep(NA,3), head(arimaSf4$mean, n = n()-3))) %>%
  mutate(garchF = c(rep(NA,3), head(garcherror4$garchfprice, n = n()-3))) %>% 
  mutate(vol = ifelse(monthlyreturn > highvolThres | monthlyreturn < lowvolThres, 1, 0))

volPlot4 <- plot4 %>%
  filter(vol == 1) %>%             
  select(x1 = date) %>%         
  mutate(min = -Inf, max = Inf)     

volPlot4 <- volPlot4 %>%
  mutate(x2 = x1 + months(1))

volPlot4 <- volPlot4 %>%
  filter(!(row_number() == n() & x2 > tail(plot4$date,1)))

head(rwD4$mean)
tail(rwD4$mean)

ggplot() +
  theme_minimal() +
  theme(panel.grid.minor = element_blank(),
        axis.line = element_line(color = "gray"),   # Adds borders to x and y axes
        axis.ticks = element_line(color = "gray")) +
  
  # Lines for the different forecast models
  geom_line(data = plot4, aes(x = date, y = cornprice, color = "Observed")) +  
  geom_line(data = plot4 %>% filter(!is.na(randomwalkF)), aes(x = date, y = randomwalkF, color = "Random Walk With Drift")) +
  geom_line(data = plot4 %>% filter(!is.na(arimaF)), aes(x = date, y = arimaF, color = "ARIMA")) +
  geom_line(data = plot4 %>% filter(!is.na(arimaSF)), aes(x = date, y = arimaSF, color = "SARIMA")) +
  geom_line(data = plot4 %>% filter(!is.na(garchF)), aes(x = date, y = garchF, color = "GARCH-ARIMA")) +
  
  # Shaded area 
  geom_rect(data = volPlot4, aes(xmin = x1, xmax = x2, ymin = -Inf, ymax = Inf, fill = "High Volatility"),
            alpha = 0.2) +
  
  # Manual color scale for the lines
  scale_color_manual(name = "Forecast Models", 
                     values = c("Observed" = "black", 
                                "Random Walk With Drift" = "#688ae8", 
                                "ARIMA" = "#c33d69", 
                                "SARIMA" = "#2ea597", 
                                "GARCH-ARIMA" = "#8456ce")) +
  
  # Manual fill scale for the shaded area
  scale_fill_manual(values = c("High Volatility" = "grey"), guide = guide_legend(title = NULL)) +
  
  labs(title = "Corn Price Forecast - Test 4",
       x = NULL,
       y = "Corn Price") +
  scale_x_date(date_breaks = "1 month", date_labels = "%b %y", limits = c(min(plot4$date), max(plot4$date)), expand = c(0,0)) +
  scale_y_continuous(labels = label_dollar()) +
  
  #Guides for legends
  guides(color = guide_legend(title = NULL))







#plots used during analysis

autoplot(cornPriceTS) +
  autolayer(rwD3, series = "Random walk with drift", PI = FALSE, colour = "green") +
  autolayer(arimaf3, series = "Arima", PI = FALSE, colour = "blue") +
  autolayer(arimaSf3, series = "Seasonal Arima", PI = FALSE, colour = "red") +
  autolayer(gmodPriceF3, series = "Arima - Garch", color = "purple")

#Plot of forecasts
autoplot(plotWindow1) +
  autolayer(rwD1, series = "Random walk with drift", PI = FALSE, colour = "green") +
  autolayer(arimaf1, series = "Arima", PI = FALSE, colour = "blue") +
  autolayer(arimaSf1, series = "Seasonal Arima", PI = FALSE, colour = "red") +
  autolayer(test1arimaxf, series = "Arimax", PI = FALSE, colour = "orange") +
  autolayer(gmodPriceF1, series = "Arima - Garch", color = "purple")
  
autoplot(plotWindow2) +
  autolayer(rwD2, series = "Random walk with drift", PI = FALSE, colour = "green") +
  autolayer(arimaf2, series = "Arima", PI = FALSE, colour = "blue") +
  autolayer(arimaSf2, series = "Seasonal Arima", PI = FALSE, colour = "red") +
  autolayer(test2arimaxf, series = "Arimax", PI = FALSE, colour = "orange") +
  autolayer(gmodPriceF2, series = "Arima - Garch", color = "purple")
  
autoplot(plotWindow3) +
  autolayer(rwD3, series = "Random walk with drift", PI = FALSE, colour = "green") +
  autolayer(arimaf3, series = "Arima", PI = FALSE, colour = "blue") +
  autolayer(arimaSf3, series = "Seasonal Arima", PI = FALSE, colour = "red") +
  autolayer(gmodPriceF3, series = "Arima - Garch", color = "purple")
  
autoplot(plotWindow4) +
  autolayer(rwD4, series = "Random walk with drift", PI = FALSE, colour = "green") +
  autolayer(arimaf4, series = "Arima", PI = FALSE, colour = "blue") +
  autolayer(arimaSf4, series = "Seasonal Arima", PI = FALSE, colour = "red") +
  autolayer(gmodPriceF4, series = "Arima - Garch", color = "purple") 

head(rwD4$mean)


# Nbeats
targetDf_train <- targetDf_train %>%
  mutate(id = as.factor(1))


pandas <- import("pandas")
for (i in nrow(targetDf_train)) {
  targetDf_train$date[i] <- pandas$to_datetime(targetDf_train$date[i])
  
}

head(targetDf_train)

xx <- pandas$to_datetime(targetDf_train$date[1])
xx
#Nbeats_recipe <- recipe(target ~ date + id, data = targetDf_train) 

targetDf_train %>% to_gluon_list_dataset(date_var = date, value_var = target, freq = "M")

nbeats_model <- nbeats(
  id = "id",
  freq = "MS",
  prediction_length = h,
  lookback_length = 27,
  bagging_size = 5,
  scale = TRUE,
  epochs = 5,
  num_batches_per_epoch = 16,
  loss_function = 'mae') %>%
  set_engine("gluonts_nbeats_ensemble")

nbeats_fit <- nbeats_model %>%
  fit(target ~ date + id, data = targetDf_train)

#nbeats_fit <- workflow() %>%
#add_model(nbeats_model) %>%
#add_recipe(Nbeats_recipe) %>%
#fit(targetDf_train
