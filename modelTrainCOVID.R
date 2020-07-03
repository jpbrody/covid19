library(h2o)
library(ukbtools)
library(tidyverse)
library(ggplot2)
library(ggpubr)
library(rstatix)

## Create training and validation frames
condensed <- read.csv("/data/ukbiobank/ukb_l2r_ids_allchr_condensed_4splits.txt", sep = " ")

# COVID data
#covid_results <- read.csv("/data/ukbiobank/covid-19/covid19_result.txt", sep = "\t")

#this is the latest results.  I think about 2x as much data.
covid_results <- read.csv("/data/ukbiobank/covid-19/covid19_result_June19-2020.txt", sep = "\t")

# Puts the date in an R data type column
covid_results$betterdate<-as.Date(covid_results$specdate,format="%d/%m/%y" )


# Get age related information
my_ukb_data_cancer <- ukb_df("ukb29274", path = "/data/ukbiobank/cancer")
my_data_age <- select(my_ukb_data_cancer, eid, yearBorn = year_of_birth_f34_0_0)



#Get some statistics by repeating:
# keep the results in aucframe

repeatlength<-100
aucframe<-data.frame(repeatnum=integer(), birthyear=integer(),  auc=double())

for( loweryear in c(1930,1940,1950,1960))
{
  for (repeatnumber in 1:repeatlength)
{
repeatnumber
# Merge with CNV data
all_data <- merge(condensed, my_data_age, by.x = "ids", by.y = "eid")

# Get COVID patients
covid_data <- merge(all_data, covid_results, by.x = "ids", by.y = "eid")
ids
#  Replaced this line with the filter /distinct below
# covid <- covid_data[covid_data$result == 1,]
#
#   added this line to filter based on results, date, and only keep one per patient.
covid <- covid_data %>% filter(result == 1, betterdate < "2020-04-27", yearBorn > loweryear) %>% distinct(ids, .keep_all=TRUE)

# Get breakdown of COVID patients' age
covid_age <- table(covid$yearBorn)

# As an age control we will only look at individuals born within the same time as the COVID patients for our non AD patients
all_data <- subset(all_data, all_data$yearBorn <= max(covid$yearBorn))
all_data <- subset(all_data, all_data$yearBorn >= min(covid$yearBorn))

#Get non COVID patients
no_covid_initial <- all_data[!all_data$ids %in% covid$ids,]

# Randomly get non COVID patients for controls so that there is an equal amount based on age
# This will ensure that the controls are age-matched to the COVID Disease sample
# For example there are 5 patients born 1937 who have covid so we will randomly grab 5 other 
# patients born 1937 who do not have covid
no_covid <- data.frame(matrix(ncol = ncol(no_covid_initial), nrow = 0))
colnames(no_covid) <- colnames(no_covid_initial)
for (i in 1:length(covid_age)) {
  temp <- covid_age[i]
  age_check <- as.integer(names(temp))
  number_cases <- as.integer(unname(temp))
#  possible_controls <- no_covid_initial[no_covid_initial$yearBorn == age_check,]
  possible_controls <- no_covid_initial %>% filter(yearBorn == age_check)
    no_covid <- rbind(no_covid, possible_controls[sample(nrow(possible_controls), number_cases, replace = FALSE), ])
}

# make sure we aren't including two copies of anyone here:
covid <- covid %>% distinct(ids,.keep_all=TRUE)

ind <- sample(c(TRUE, FALSE), nrow(covid), replace=TRUE, prob=c(0.7, 0.3)) # Random split

train <- covid[ind, ]
validate <- covid[!ind, ]

#Remove unnecessary columns
train <- train[,!names(train) %in% c("datereported", "specdate", "spectype", "laboratory", "origin","betterdate")]
validate <- validate[,!names(validate) %in% c( "datereported", "specdate", "spectype", "laboratory", "origin","betterdate")]


controls <- no_covid[sample(nrow(no_covid), nrow(covid), replace = FALSE), ] # Randomly get controls
controls['result'] = 0
controls <- controls %>% distinct(ids,.keep_all = TRUE)


train_controls <- controls[ind, ]
validate_controls <- controls[!ind, ]

# Combine controls with samples

train <- rbind(train, train_controls)
validate <- rbind(validate, validate_controls)

# Set response column to boolean
train$result <- as.logical(train$result)
validate$result <- as.logical(validate$result)
train$result <- as.factor(train$result)
validate$result <- as.factor(validate$result)

#Remove unnecessary columns
train <- train[,!names(train) %in% c("ids")]
validate <- validate[,!names(validate) %in% c( "ids")]


# this is for the random control
train$result<-sample(train$result)


# Free up data 
# rm(no_covid, covid, controls, train_controls, validate_controls)
# rm(covid_results, condensed, my_ukb_data_cancer, my_data_age, all_data)

# Load h2o
h2o.init(nthreads=5,strict_version_check = FALSE)

# Load data into h2o
train.hex <- as.h2o(train, destination_frame = "train.hex")  
validate.hex <- as.h2o(validate, destination_frame = "validate.hex")

#
#  I usually stop here and goto http://localhost:54321/flow/index.html 
#  h2o runs a local webserver on port 54321, it offers a nice little interface.
#  you can run the AutoML from the web browser there and it has some nice features so that 
# you can monitor the progress of the training.

#Response column
response <- "result"
#Get Predictors
predictors <- colnames(train)
predictors <- predictors[! predictors %in% response] #Response cannot be a predictor
predictors <- predictors[! predictors %in% "yearBorn"] #Remove yearBorn as predictor
#model <- h2o.automl(x = predictors,
#                    y = response,
#                    training_frame = train.hex,
#                    validation_frame = validate.hex,
#                    nfolds=5,
#                    max_runtime_secs = 3600)

model <- h2o.xgboost(x = predictors,
                           y = response,
                           training_frame = train.hex,
                           validation_frame = ,
                           nfolds = 5,
                           booster = "dart",
                           normalize_type = "tree",
                           seed = 1234)


#  Got rid of the automl and just went with xgboost, since it was always the best
#record the Leading model AUC in the dataset
# leader <- model@leader
auc=h2o.auc(model, train=FALSE, xval=TRUE)

# plot out the ROC.  We type out the tissue and AUC at the top of the ROC.
plot(h2o.performance(model,train=FALSE, xval=TRUE),type='roc',main=paste("COVID Cross-Validated 4 Splits",auc))

h2o.shutdown(prompt = FALSE)
aucframe<-  aucframe %>% add_row(repeatnum=repeatnumber, birthyear=loweryear,  auc=auc)

  }
}


#  June 27, 2020
#  Notes.  I had some problems getting this to run to completion because of h2o issues.
#  Short version.  I ran the above code to generate dataframe. Saved the dataframe. 
#  This dataframe should contain 100 repetions with three cohorts (1930,1940,1950) 
#  each cohort contains everyone born after that year. I think I had a 1960 cohort, but h2o kept crashing.
#  
#  Then I added line 112 to randomize and I ran it again.  Saved the dataframe.
#  
#  I don't use the validation frame, just throwing away 20% of the data.  I should fix that.
#
# Once I have the two dataframes I do some plots/statistics below.
#
# I ran these lines in the console:
# > testframe<- aucframerando %>% mutate(dataorigin=paste(birthyear," random"))
# > testrealframe <- aucframegood %>% mutate(dataorigin=paste(birthyear," data"))
# > finaldata<-rbind(testframe,testrealframe)
#
#
#  Then, this plots the graph I use:
# > p <- ggboxplot(finaldata, x = "birthyear", y = "auc", add="jitter", color="dataorigin")
# > p


# Table to compare data/randomdata
#
stat.test <- finaldata %>%
  group_by(birthyear) %>%
  t_test(auc ~ dataorigin) %>%
  adjust_pvalue(method = "BH") %>%
  add_significance()
stat.test

#
# make vectors for t.tests
#
rand30<-finaldata %>% filter(dataorigin == "1930  random") 
rand40<-finaldata %>% filter(dataorigin == "1940  random") 
rand50<-finaldata %>% filter(dataorigin == "1950  random") 
data30<-finaldata %>% filter(dataorigin == "1930  data") 
data40<-finaldata %>% filter(dataorigin == "1940  data") 
data50<-finaldata %>% filter(dataorigin == "1950  data") 
t.test(rand30$auc,mu=0.5)
t.test(rand40$auc,mu=0.5)
t.test(rand50$auc,mu=0.5)
t.test(data30$auc,mu=0.5)
t.test(data40$auc,mu=0.5)
t.test(data50$auc,mu=0.5)


#
# Get a nice dataframe with patient numbers for statistics.
#
covidpaper <- covid_data %>% filter(result == 1, betterdate < "2020-04-27") %>% distinct(ids, .keep_all=TRUE) %>% select(ids,yearBorn,specdate,spectype,laboratory,origin,result,betterdate)



#saveRDS(aucframe,file="aucframerando6-26")
aucframerando<-readRDS(file="aucframerando6-26")
aucframegood<-readRDS(file="aucframe.RDS")

aucframe %>% 
  group_by(birthyear) %>% 
  summarise(meanauc=mean(auc),sdauc=sd(auc))

p <- ggboxplot(aucframegood, x = "birthyear", y = "auc", add="jitter")
p + stat_compare_means(method = "t.test")

p <- ggboxplot(aucframerando, x = "birthyear", y = "auc", add="jitter")
p + stat_compare_means(method = "t.test")


p <- ggboxplot(finaldata, x = "birthyear", y = "auc", add="jitter", color="dataorigin")
p

#done
