library(h2o)
library(ukbtools)
library(tidyverse)

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


# Free up data 
# rm(no_covid, covid, controls, train_controls, validate_controls)
# rm(covid_results, condensed, my_ukb_data_cancer, my_data_age, all_data)

# Load h2o
h2o.init(nthreads=5)

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
