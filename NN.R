---
title: "Churn analysis with neuralnet"
author: "Paige, Vicky, and Jackie"
date: '2019-11-10'
output: 
  prettydoc::html_pretty:
  theme : cayman
  highlight : github
  math: katex
---

setwd("C:\\Users\\Correy\\Documents\\Stat 350\\finalProject\\NNChurn")

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
r <- neuralnet(churn ~ gender + seniorcitizen + partner + dependents + tenure + phoneservice + multiplelines + internetservicefibreoptic + internetserviceno +internetserviceDSL + onlinesecurity + onlinebackup + deviceprotection + techsupport + streamingtv + streamingmovies + contractoneyear + contracttwoyear +contractmtm + paperlessbilling + paymentmethodcreditcard + paymentmethodbanktransfer + paymentmethodmailedcheck + paymentmethodelectroniccheck + monthlycharges + totalcharges,
  data=cd, hidden=20, threshold=0.01, rep = 10
)

plot(r)



Predict <- compute(r, test)

preds <- Predict$net.result

preds <- rep("No", length(preds))

preds[preds > 0.5] <- "Yes"

cd1 <- data.frame(Churn = cd$churn, predicted = preds)

xtabs(~ predicted + Churn, data = cd1)
