#reading the data

library("readxl")
library("imputeTS")
#setting working directory
setwd("~/Final year thesis/Dataset")

#reading the files and skipping redundant rows
air<- read_excel("pollution.xlsx", skip = 15)

#changing the column names using the first row
colnames(air) = air[1, ] # the first row will be the header
air = air[-1, ] 


names(air)[1] <- "date"
#to seperate date and time into seperate columns
air <- tidyr::separate(air, date, c("date", "time"), sep = " ")

#checking NA

#as we have '0' and 'None' as missing valurs in our data, first we would convert them into NAs so that they can be replaced
air[air == 0] <- NA
air[air == "None"] <- NA

#checking total number of missing entries
table(is.na(air))

#removing to date column 
air$`To Date`<- NULL
sum(is.na(air$date))

#Converting to numeric
air$SO2 <- as.numeric(air$SO2)
air$PM2.5 <- as.numeric(air$PM2.5)
air$CO <- as.numeric(air$CO)
air$Ozone <- as.numeric(air$Ozone)
air$NO2 <- as.numeric(air$NO2)
 
air2 <- air

################################################################################
#trying to handle missing values using seasonal adjustment+linear interpolation

air$SO2 <- na.seadec(air$SO2, algorithm = "interpolation") # Seasonal Adjustment then Linear Interpolation
air$PM2.5 <- na.seadec(air$PM2.5, algorithm = "interpolation") 
air$CO <- na.seadec(air$CO, algorithm = "interpolation") 
air$Ozone <- na.seadec(air$Ozone, algorithm = "interpolation") 
air$NO2 <- na.seadec(air$NO2, algorithm = "interpolation") 

air2 <- na.seadec(air, algorithm = "interpolation", find_frequency=TRUE) # Seasonal Adjustment then Linear Interpolation

#saving interpolated data for future use
write.csv(air2, "using_interpolation.csv", row.names = FALSE)

###############################################







