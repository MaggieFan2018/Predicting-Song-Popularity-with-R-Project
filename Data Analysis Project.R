setwd("~/Desktop/Predicting-Song-Popularity-in-R")
#get the whole data
wholedata <- read.csv("raw.csv", header = T)
#look at the data type
str(wholedata)
dim(wholedata)

#sample data
set.seed(1)
data <- wholedata[sample(nrow(wholedata), 10000), ]
training <- data[1:5000, ]
testing <- data[5001:10000, ]

#Look at training data
View(training)

#Drop track_id
training <- training[ , -2]
testing <- testing[ , -2]
View(training)

#check NAs
sum(is.na(training))
sum(is.na(testing))

#statistical summary
summary(training)
summary(testing)

#feature distribution
#histograms
names(training)
library(ggplot2)
library(dplyr)
library(ggthemes)
#training data
training %>% select(acousticness,danceability,energy,instrumentalness,liveness,loudness,mode,speechiness,tempo,time_signature,valence,popularity, key, duration_ms) %>% 
  tidyr::gather() %>% 
  ggplot(aes(x=value)) + geom_histogram() + 
  facet_wrap(~key, scales='free', ncol=3) + 
  theme_fivethirtyeight()
#testing data
testing %>% select(acousticness,danceability,energy,instrumentalness,liveness,loudness,mode,speechiness,tempo,time_signature,valence, key, duration_ms) %>% 
  tidyr::gather() %>% 
  ggplot(aes(x=value)) + geom_histogram() + 
  facet_wrap(~key, scales='free', ncol=3) + 
  theme_fivethirtyeight()

#multivariate EDA, check for correlations in predictor variables
library(corrplot)
songs_cor <- training %>%
  select(-c(artist_name, track_name)) %>%
  cor() %>%
  corrplot() 
songs_cor

#drop energy variable
training <- training[ , -6]
testing <- testing[ , -6]
#create a new dataset with variables:artist_name, track_name, popularity
names(data)
textdata <- data[ , c(1, 3, 17)]
View(textdata)
#drop the artist_name and track_name variables in training and testing datasets
training <- training[ , -c(1, 2)]
testing <- testing[ , -c(1, 2)]
View(training)

#convert duration to secondes
training$duration_ms <- training$duration_ms / 1000
testing$duration_ms <- testing$duration_ms / 1000

#transfer data type
training$time_signature <- as.factor(training$time_signature)
training$key <- as.factor(training$key)
testing$time_signature <- as.factor(training$time_signature)
testing$key <- as.factor(training$key)

#mode
training$mode <- ifelse(training$mode == 1, "major", "minor")
testing$mode <- ifelse(training$mode == 1, "major", "minor")

#popularity_new
training$popularity_new <- ifelse(training$popularity <= 20, "dislike", ifelse((training$popularity > 20) & (training$popularity <= 50), "neutral", "like"))
testing$popularity_new <- ifelse(testing$popularity <= 20, "dislike", ifelse((testing$popularity > 20) & (testing$popularity <= 50), "neutral", "like"))

str(training)

#outliers
#detect outliers 1
outlier_values <- boxplot.stats(training$loudness)$out  # outlier values.
boxplot(training$loudness, main="loudness", boxwex=0.1)
mtext(paste("Outliers: ", paste(outlier_values, collapse=", ")), cex=0.6)
outlier_values
#remove outliers
training <- training[-which(training$loudness %in% outlier_values),]

#detect outliers 2
outlier_values2 <- boxplot.stats(training$instrumentalness)$out  # outlier values.
boxplot(training$instrumentalness, main="instrumentalness", boxwex=0.1)
mtext(paste("Outliers: ", paste(outlier_values2, collapse=", ")), cex=0.6)
outlier_values2
#remove outliers
training <- training[-which(training$instrumentalness %in% outlier_values2),]

#detect outliers 3
outlier_values3 <- boxplot.stats(training$danceability)$out  # outlier values.
boxplot(training$danceability, main="danceability", boxwex=0.1)
mtext(paste("Outliers: ", paste(outlier_values3, collapse=", ")), cex=0.6)
outlier_values3
#remove outliers
training <- training[-which(training$danceability %in% outlier_values3),]

#detect outliers 4
outlier_values4 <- boxplot.stats(training$acousticness)$out  # outlier values.
boxplot(training$acousticness, main="acousticness", boxwex=0.1)
mtext(paste("Outliers: ", paste(outlier_values4, collapse=", ")), cex=0.6)
outlier_values4

#feature transformation
#scaling
training_num <- data.frame(training %>% select_if(is.numeric))
names(training_num)
training_num <- scale(training_num %>% select(-popularity))
View(training_num)

testing_num <- data.frame(testing %>% select_if(is.numeric))
names(testing_num)
testing_num <- scale(testing_num %>% select(-popularity))
View(testing_num)
#get dummies
library(fastDummies)
training <- dummy_cols(training, select_columns = c("mode", "key", "time_signature"))
testing <- dummy_cols(testing, select_columns = c("mode", "key", "time_signature"))
View(testing)
View(training)

#create the final datasets
names(training)
names(training_num)
training <-cbind(training_num, training[ , 14:32])
head(training)

names(testing)
names(testing_num)
testing <-cbind(testing_num, testing[ , 14:32])
head(testing)

#train test split
## 75% of the sample size
smp_size <- floor(0.75 * nrow(training))
## set the seed to make partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(training)), size = smp_size)
train <- training[train_ind, ]
test <- training[-train_ind, ]

str(train)

# Run algorithms using 10-fold cross validation
library(caret)
control <- trainControl(method="cv", number=10, repeats=3)
metric <- "Accuracy"
# a) linear algorithms
set.seed(7)
fit.lda <- train(popularity_new~., data=train, method="lda", metric=metric, trControl=control, tuneLength=5)
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(popularity_new~., data=train, method="rpart", metric=metric, trControl=control, tuneLength=5)
# kNN
set.seed(7)
fit.knn <- train(popularity_new~., data=train, method="knn", metric=metric, trControl=control, tuneLength=5)
# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(popularity_new~., data=train, method="svmRadial", metric=metric, trControl=control, tuneLength=5)
# Random Forest
set.seed(7)
fit.rf <- train(popularity_new~., data=train, method="rf", metric=metric, trControl=control, tuneLength=5)

# summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)


# compare accuracy of models
dotplot(results)




#model optimization
names(training)
training2 <- training[ , c(1, 2, 4, 6, 10, 25, 26, 27, 28)]
#train test split
## 75% of the sample size
smp_size2 <- floor(0.75 * nrow(training2))
## set the seed to make partition reproducible
set.seed(123)
train_ind2 <- sample(seq_len(nrow(training2)), size = smp_size2)
train2 <- training2[train_ind2, ]
test2 <- training2[-train_ind2, ]
# a) linear algorithms
set.seed(7)
fit.lda2 <- train(popularity_new~., data=train2, method="lda", metric=metric, trControl=control, tuneLength=5)
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart2 <- train(popularity_new~., data=train2, method="rpart", metric=metric, trControl=control, tuneLength=5)
# kNN
set.seed(7)
fit.knn2 <- train(popularity_new~., data=train2, method="knn", metric=metric, trControl=control, tuneLength=5)
# c) advanced algorithms
# SVM
set.seed(7)
fit.svm2 <- train(popularity_new~., data=train2, method="svmRadial", metric=metric, trControl=control, tuneLength=5)
# Random Forest
set.seed(7)
fit.rf2 <- train(popularity_new~., data=train2, method="rf", metric=metric, trControl=control, tuneLength=5)

# summarize accuracy of models
results2 <- resamples(list(lda=fit.lda2, cart=fit.cart2, knn=fit.knn2, svm=fit.svm2, rf=fit.rf2))
summary(results2)
# compare accuracy of models
dotplot(results2)

# estimate skill of LDA on the validation dataset
test$predictions <- predict(fit.lda, test)
test$popularity_new <- as.factor(test$popularity_new)
confusionMatrix(test$predictions, test$popularity_new)

# estimate skill of other 4 models on the validation dataset
test$predictions2 <- predict(fit.cart, test)
confusionMatrix(test$predictions2, test$popularity_new)

test$predictions3 <- predict(fit.knn, test)
confusionMatrix(test$predictions3, test$popularity_new)

test$predictions4 <- predict(fit.svm, test)
confusionMatrix(test$predictions4, test$popularity_new)

test$predictions5 <- predict(fit.rf, test)
confusionMatrix(test$predictions5, test$popularity_new)

#prediction in testing data
testing$mode_major <- 0
testing$predictions <- predict(fit.lda, testing)
testing$popularity_new <- as.factor(testing$popularity_new)
confusionMatrix(testing$predictions, testing$popularity_new)


#analysis on text data
View(textdata)
Popular_Song <- textdata[textdata$popularity > 50, ]
View(Popular_Song)
library(wordcloud2)

artist_words_counts <- Popular_Song %>%
  count(artist_name, sort = TRUE) 
wordcloud2(artist_words_counts, size = .5)

songs_words_counts <- Popular_Song %>%
  count(track_name, sort = TRUE) 
wordcloud2(songs_words_counts, size = .5)


