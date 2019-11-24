library(dplyr)# Load the data
cd <- read.csv(file ="/home/ptuttosi/Stat350/NNChurn/oneHotBalance.csv", header = TRUE, sep = ",", stringsAsFactors = T)

print("ORIGIN DATA")

summary(cd)

smpsize <- floor(0.70*nrow(cd))

set.seed(6969)
train_index <- sample(seq_len(nrow(cd)), size = smpsize)

train <- cd[train_index, ]
test <- cd[-train_index, ]


## Fitting the model

library(neuralnet)
print("Running")
r <- neuralnet(churn ~ gender + seniorcitizen + partner + dependents + tenure + phoneservice + multiplelines + internetservicefibreoptic + internetserviceno +internetserviceDSL + onlinesecurity + onlinebackup + deviceprotection + techsupport + streamingtv + streamingmovies + contractoneyear + contracttwoyear +contractmtm + paperlessbilling + paymentmethodcreditcard + paymentmethodbanktransfer + paymentmethodmailedcheck + paymentmethodelectroniccheck + monthlycharges + totalcharges,
  data=cd, hidden=1, threshold=0.01, rep = 1, stepmax = 1e+100
)
print("Done")

plot(r)



Predict <- compute(r, test)

probs <- Predict$net.result

preds <- rep("No", length(probs))

preds[probs > 0.5] <- "Yes"

tests<- test$churn

tests <- rep("No", length(tests))

tests[test$churn == 1] <- "Yes"

cd1 <- data.frame(tests = tests, predicted = preds)

xtabs(~ predicted + tests, data = cd1)
