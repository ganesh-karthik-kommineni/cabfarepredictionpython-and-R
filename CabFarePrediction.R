rm(list = ls())

getwd()
setwd("C:/Users/ganes/Desktop/Data Science/Rr/CabFarePredictionProject")

# #loading Libraries
x = c("ggplot2", "corrgram", "DMwR", "usdm", "caret", "randomForest", "e1071",
      "DataCombine", "doSNOW", "inTrees", "rpart.plot", "rpart",'MASS','xgboost','stats')

#load Packages
install.packages('rpart.plot')
install.packages('doSNOW')
lapply(x, require, character.only = TRUE)
rm(x)

# The details of data attributes in the dataset are as follows:
# pickup_datetime - timestamp value indicating when the cab ride started.
# pickup_longitude - float for longitude coordinate of where the cab ride started.
# pickup_latitude - float for latitude coordinate of where the cab ride started.
# dropoff_longitude - float for longitude coordinate of where the cab ride ended.
# dropoff_latitude - float for latitude coordinate of where the cab ride ended.
# passenger_count - an integer indicating the number of passengers in the cab ride.

# loading datasets
train_data = read.csv("train_cab.csv", header = T, na.strings = c(" ", "", "NA"))
test_data = read.csv("test.csv")
test_pickup_datetime = test_data["pickup_datetime"]

# Structure of data
str(train_data)
str(test_data)
summary(train_data)
summary(test_data)
head(train_data,5)
head(test_data,5)


#############        Exploratory Data Analysis         #######################
# Changing the data types of variables
train_data$fare_amount = as.numeric(as.character(train_data$fare_amount))
train_data$passenger_count=round(train_data$passenger_count)

### Removing values which are not within desired range(outlier) depending upon basic understanding of dataset.

# 1.Fare amount has a negative value, which doesn't make sense. 
# The price amount cannot be -ve and also cannot be 0. So we will remove these fields.
train_data[which(train_data$fare_amount < 1 ),]
nrow(train_data[which(train_data$fare_amount < 1 ),])
train_data = train_data[-which(train_data$fare_amount < 1 ),]

#2.Passenger_count variable
for (i in seq(4,11,by=1)){
  print(paste('passenger_count above ' ,i,nrow(train_data[which(train_data$passenger_count > i ),])))
}
# so 20 observations of passenger_count are consistenly above 6,7,8,9,10 passenger_counts, let's check them.
train_data[which(train_data$passenger_count > 6 ),]
# Also let's see if there are any passenger_count==0
train_data[which(train_data$passenger_count <1 ),]
nrow(train_data[which(train_data$passenger_count <1 ),])
# Let's remove these 58 observations and 20 observation which are above 6 value because a cab cannot hold these number of passengers.
train_data = train_data[-which(train_data$passenger_count < 1 ),]
train_data = train_data[-which(train_data$passenger_count > 6),]


# 3.Latitudes range from -90 to 90.Longitudes range from -180 to 180.Let's remove which does not satisfy these ranges
print(paste('pickup_longitude above 180=',nrow(train_data[which(train_data$pickup_longitude >180 ),])))
print(paste('pickup_longitude above -180=',nrow(train_data[which(train_data$pickup_longitude < -180 ),])))
print(paste('pickup_latitude above 90=',nrow(train_data[which(train_data$pickup_latitude > 90 ),])))
print(paste('pickup_latitude above -90=',nrow(train_data[which(train_data$pickup_latitude < -90 ),])))
print(paste('dropoff_longitude above 180=',nrow(train_data[which(train_data$dropoff_longitude > 180 ),])))
print(paste('dropoff_longitude above -180=',nrow(train_data[which(train_data$dropoff_longitude < -180 ),])))
print(paste('dropoff_latitude above -90=',nrow(train_data[which(train_data$dropoff_latitude < -90 ),])))
print(paste('dropoff_latitude above 90=',nrow(train_data[which(train_data$dropoff_latitude > 90 ),])))


# There's only one outlier which is in variable pickup_latitude.So let's remove it with nan.
# Also let's see if there are any values equal to 0.
nrow(train_data[which(train_data$pickup_longitude == 0 ),])
nrow(train_data[which(train_data$pickup_latitude == 0 ),])
nrow(train_data[which(train_data$dropoff_longitude == 0 ),])
nrow(train_data[which(train_data$pickup_latitude == 0 ),])
# there are values which are equal to 0. we will remove them.
train_data = train_data[-which(train_data$pickup_latitude > 90),]
train_data = train_data[-which(train_data$pickup_longitude == 0),]
train_data = train_data[-which(train_data$dropoff_longitude == 0),]

# Make a copy
df=train_data
# train_data=df

#############            Missing Value Analysis            #############
missing_val = data.frame(apply(train_data,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(train_data)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL

missing_val = missing_val[,c(2,1)]
missing_val

unique(train_data$passenger_count)
unique(test_data$passenger_count)
train_data[,'passenger_count'] = factor(train_data[,'passenger_count'], labels=(1:6))
test_data[,'passenger_count'] = factor(test_data[,'passenger_count'], labels=(1:6))
# 1.For Passenger_count:
# Actual value = 1
# Mode = 1
# KNN = 1
train_data$passenger_count[1000]
train_data$passenger_count[1000] = NA
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Mode Method
getmode(train_data$passenger_count)
# We can't use mode method because data will be more biased towards passenger_count=1

# 2.For fare_amount:
# Actual value = 18.1,
# Mean = 15.117,
# Median = 8.5,
# KNN = 18.28
sapply(train_data, sd, na.rm = TRUE)
# fare_amount   pickup_datetime  pickup_longitude 
# 435.968236       4635.700531          2.659050 
# pickup_latitude dropoff_longitude  dropoff_latitude 
# 2.613305          2.710835          2.632400 
# passenger_count 
# 1.266104
train_data$fare_amount[1000]
train_data$fare_amount[1000]= NA

# Mean Method
mean(train_data$fare_amount, na.rm = T)

#Median Method
median(train_data$fare_amount, na.rm = T)

# kNN Imputation
train_data = knnImputation(train_data, k = 181)
train_data$fare_amount[1000]
train_data$passenger_count[1000]
sapply(train_data, sd, na.rm = TRUE)
# fare_amount   pickup_datetime  pickup_longitude 
# 435.661952       4635.700531          2.659050 
# pickup_latitude dropoff_longitude  dropoff_latitude 
# 2.613305          2.710835          2.632400 
# passenger_count 
# 1.263859 
sum(is.na(train_data))
str(train_data)
summary(train_data)

df1=train_data
# train_data=df1

#####################       Outlier Analysis         ##################

# Let's do Outlier Analysis only on Fare_amount just for now and then we will do outlier analysis after feature engineering latitudes and longitudes.
# Boxplot for fare_amount
pl1 = ggplot(train_data,aes(x = factor(passenger_count),y = fare_amount))
pl1 + geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,outlier.size=1, notch=FALSE)+ylim(0,100)

# Replace all outliers with NA and impute
vals = train_data[,"fare_amount"] %in% boxplot.stats(train_data[,"fare_amount"])$out
train_data[which(vals),"fare_amount"] = NA

#lets check the NA's
sum(is.na(train_data$fare_amount))

#Imputing with KNN
train_data = knnImputation(train_data,k=3)

# lets check the missing values
sum(is.na(train_data$fare_amount))
str(train_data)

df2=train_data
# train_data=df2
##################          Feature Engineering              ##########################
# 1.Feature Engineering for timestamp variable
# Let's derive new features from pickup_datetime variable
# new features will be year,month,day_of_week,hour
#Convert pickup_datetime from factor to date time
train_data$pickup_date = as.Date(as.character(train_data$pickup_datetime))
train_data$pickup_weekday = as.factor(format(train_data$pickup_date,"%u"))# Monday = 1
train_data$pickup_mnth = as.factor(format(train_data$pickup_date,"%m"))
train_data$pickup_yr = as.factor(format(train_data$pickup_date,"%Y"))
pickup_time = strptime(train_data$pickup_datetime,"%Y-%m-%d %H:%M:%S")
train_data$pickup_hour = as.factor(format(pickup_time,"%H"))

#Add same features to test set
test_data$pickup_date = as.Date(as.character(test_data$pickup_datetime))
test_data$pickup_weekday = as.factor(format(test_data$pickup_date,"%u"))# Monday = 1
test_data$pickup_mnth = as.factor(format(test_data$pickup_date,"%m"))
test_data$pickup_yr = as.factor(format(test_data$pickup_date,"%Y"))
pickup_time = strptime(test_data$pickup_datetime,"%Y-%m-%d %H:%M:%S")
test_data$pickup_hour = as.factor(format(pickup_time,"%H"))

sum(is.na(train_data))# there was 1 'na' in pickup_datetime which created na's in above feature engineered variables.
train_data = na.omit(train_data) # let's remove that 1 row of na's

train_data = subset(train_data,select = -c(pickup_datetime,pickup_date))
test_data = subset(test_data,select = -c(pickup_datetime,pickup_date))
# Now we will use month,weekday,hour to derive new features like sessions in a day,seasons in a year,week:weekend/weekday
f = function(x){
if ((x >=5)& (x <= 11)){
    return ('morning')
}
if ((x >=12) & (x <= 16)){
    return ('afternoon')
}
if ((x >=17) & (x <= 20)){
    return ('evening')
}
if ((x >=21) & (x <= 23)){
    return ('night (PM)')
}
if ((x >=0) & (x <= 4)){
    return ('night (AM)')
}
}


# 2.Calculate the distance travelled using longitude and latitude
deg_to_rad = function(deg){
  (deg * pi) / 180
}
haversine = function(long1,lat1,long2,lat2){
  #long1rad = deg_to_rad(long1)
  phi1 = deg_to_rad(lat1)
  #long2rad = deg_to_rad(long2)
  phi2 = deg_to_rad(lat2)
  delphi = deg_to_rad(lat2 - lat1)
  dellamda = deg_to_rad(long2 - long1)
  
  a = sin(delphi/2) * sin(delphi/2) + cos(phi1) * cos(phi2) * 
    sin(dellamda/2) * sin(dellamda/2)
  
  c = 2 * atan2(sqrt(a),sqrt(1-a))
  R = 6371e3
  R * c / 1000 #1000 is used to convert to meters
}


# Using haversine formula to calculate distance fr both train and test
train_data$dist = haversine(train_data$pickup_longitude,train_data$pickup_latitude,train_data$dropoff_longitude,train_data$dropoff_latitude)

test_data$dist = haversine(test_data$pickup_longitude,test_data$pickup_latitude,test_data$dropoff_longitude,test_data$dropoff_latitude)

# let's remove the variables which were used to feature engineer new variables
train_data = subset(train_data,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))
test_data = subset(test_data,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))

str(train_data)
summary(train_data)

################         Feature selection           ###################
numeric_index = sapply(train_data,is.numeric) #selecting only numeric

numeric_data = train_data[,numeric_index]

cnames = colnames(numeric_data)

#Correlation analysis for numeric variables
corrgram(train_data[,numeric_index],upper.panel=panel.pie, main = "Correlation Plot")

#ANOVA for categorical variables with target numeric variable

#aov_results = aov(fare_amount ~ passenger_count * pickup_hour * pickup_weekday,data = train_data)
aov_results = aov(fare_amount ~ passenger_count + pickup_hour + pickup_weekday + pickup_mnth + pickup_yr,data = train_data)

summary(aov_results)

# pickup_weekdat has p value greater than 0.05 
train_data = subset(train_data,select=-pickup_weekday)

#remove from test set
test_data = subset(test_data,select=-pickup_weekday)


##################################             Feature Scaling         ################################################
#Normality check
#qqnorm(train_data$fare_amount)
#histogram(train_data$fare_amount)
library(car)
#dev.off()
par(mfrow=c(1,2))
qqPlot(train_data$fare_amount)                             # qqPlot, it has a x values derived from gaussian distribution, if data is distributed normally then the sorted data points should lie very close to the solid reference line 
truehist(train_data$fare_amount)                           # truehist() scales the counts to give an estimate of the probability density.
lines(density(train_data$fare_amount))  # Right skewed      # lines() and density() functions to overlay a density plot on histogram

#Normalisation

print('dist')
train_data[,'dist'] = (train_data[,'dist'] - min(train_data[,'dist']))/
  (max(train_data[,'dist'] - min(train_data[,'dist'])))

#check multicollearity
#library(usdm)
#vif(train_data[,-1])
 
#vifcor(train_data[,-1], th = 0.9)

######## Splitting train into train and validation subsets ##########
set.seed(1000)
tr.idx = createDataPartition(train_data$fare_amount,p=0.75,list = FALSE) # 75% in trainin and 25% in Validation Datasets
train_data1 = train_data[tr.idx,]
test_data1 = train_data[-tr.idx,]

rmExcept(c("test_data","train_data","df",'df1','df2','df3','test_data1','train_data1','test_pickup_datetime'))
###################Model Selection################
#Error metric used to select model is RMSE


#############            Linear regression               #################
lm_model = lm(fare_amount ~.,data=train_data1)

summary(lm_model)
str(train_data1)
plot(lm_model$fitted.values,rstandard(lm_model),main = "Residual plot",
     xlab = "Predicted values of fare_amount",
     ylab = "standardized residuals")


lm_predictions = predict(lm_model,test_data1[,2:6])

qplot(x = test_data1[,1], y = lm_predictions, data = test_data1, color = I("blue"), geom = "point")

regr.eval(test_data1[,1],lm_predictions)
# mae        mse       rmse       mape 
# 3.5303114 19.3079726  4.3940838  0.4510407  

#############           Decision Tree            #####################

Dt_model = rpart(fare_amount ~ ., data = train_data1, method = "anova")

summary(Dt_model)
#Predict for new test cases
predictions_DT = predict(Dt_model, test_data1[,2:6])

qplot(x = test_data1[,1], y = predictions_DT, data = test_data1, color = I("blue"), geom = "point")

regr.eval(test_data1[,1],predictions_DT)
# mae       mse      rmse      mape 
# 1.8981592 6.057651 2.5891063 0.2241461 


#############                             Random forest            #####################
rf_model = randomForest(fare_amount ~.,data=train_data1)

summary(rf_model)

rf_predictions = predict(rf_model,test_data1[,2:6])

qplot(x = test_data1[,1], y = rf_predictions, data = test_data1, color = I("blue"), geom = "point")

regr.eval(test_data1[,1],rf_predictions)
# mae         mse      rmse       mape 
# 1.9103915 6.4214597 2.5340599  0.2342449


############   Improving Accuracy by using Ensemble technique ---- XGBOOST     ###########################
train_data1_matrix = as.matrix(sapply(train_data1[-1],as.numeric))
test_data1_data_matrix = as.matrix(sapply(test_data1[-1],as.numeric))

xgboost_model = xgboost(data = train_data1_matrix,label = train_data1$fare_amount,nrounds = 15,verbose = FALSE)

summary(xgboost_model)
xgb_predictions = predict(xgboost_model,test_data1_data_matrix)

qplot(x = test_data1[,1], y = xgb_predictions, data = test_data1, color = I("blue"), geom = "point")

regr.eval(test_data1[,1],xgb_predictions)
#       mae       mse      rmse      mape 
# 1.6193378   5.2082001   2.2821481   0.1836901  


#############                         Finalizing and Saving Model for later use                         ####################
# In this step we will train our model on whole training Dataset and save that model for later use
train_data1_matrix2 = as.matrix(sapply(train_data[-1],as.numeric))
test_data1_matrix2 = as.matrix(sapply(test_data,as.numeric))

xgboost_model2 = xgboost(data = train_data1_matrix2,label = train_data$fare_amount,nrounds = 15,verbose = FALSE)

# Saving the trained model
saveRDS(xgboost_model2, "./final_Xgboost_model_using_R.rds")

# loading the saved model
super_model <- readRDS("./final_Xgboost_model_using_R.rds")
print(super_model)

# Lets now predict on test dataset
xgb = predict(super_model,test_data1_matrix2)

xgb_pred = data.frame(test_pickup_datetime,"predictions" = xgb)

# Now lets write(save) the predicted fare_amount in disk as .csv format 
write.csv(xgb_pred,"xgb_predictions_R.csv",row.names = FALSE)


