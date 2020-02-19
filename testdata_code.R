library(readr)
library(tidyverse)
library(dplyr)
library(imputeTS)
library(foreach)
library("readxl")

#reading the files and skipping redundant rows
test<- read_excel("test_dataset.xlsx", skip = 12)

names(test)[1] <- "date"
#to seperate date and time into seperate columns
test <- tidyr::separate(test, date, c("date", "time"), sep = " ")

#checking NA

#checking total number of missing entries
table(is.na(test))

#removing to date column, pm10 and no as these are not required
test$`To Date`<- NULL
test$PM10 <- NULL
test$NO <- NULL

sum(is.na(test$date))

#Converting to numeric
test$SO2 <- as.numeric(test$SO2)
test$PM2.5 <- as.numeric(test$PM2.5)
test$CO <- as.numeric(test$CO)
test$Ozone <- as.numeric(test$Ozone)
test$NO2 <- as.numeric(test$NO2)

#as we have '0' and 'None' as missing valurs in our data, first we would convert them into NAs so that they can be replaced
test[test == 0] <- NA
test[test == "None"] <- NA

################################################################################
#trying to handle missing values using seasonal adjustment+linear interpolation


test2 <- na.seadec(test, algorithm = "interpolation", find_frequency=TRUE) # Seasonal Adjustment then Linear Interpolation

#checking the number of null
sum(is.na(test2)) #0 null values remaining

#saving interpolated data for future use
write.csv(test2, "testset1.csv", row.names = FALSE)
