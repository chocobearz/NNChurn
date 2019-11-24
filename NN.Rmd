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

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

Info:
```{r}
# Link to the article that informed this code
# https://www.datacamp.com/community/tutorials/neural-network-models-r
```

**Trying neural networks with churn data**

```{r}
library(dplyr)# Load the data
cd <- read.csv(file = "oneHotBalance.csv", header = TRUE, sep = ",", 
                stringsAsFactors = T)

print("ORIGIN DATA")

summary(cd)
```

```{r}
smpsize <- floor(0.70*nrow(cd))

set.seed(6969)
train_index <- sample(seq_len(nrow(cd)), size = smpsize)

train <- cd[train_index, ]
test <- cd[-train_index, ]
```

## Fitting the model
```{r}
library(neuralnet)
r <- neuralnet(churn ~ gender + seniorcitizen + partner + dependents + tenure + phoneservice + multiplelines + internetservicefibreoptic + internetserviceno +internetserviceDSL + onlinesecurity + onlinebackup + deviceprotection + techsupport + streamingtv + streamingmovies + contractoneyear + contracttwoyear +contractmtm + paperlessbilling + paymentmethodcreditcard + paymentmethodbanktransfer + paymentmethodmailedcheck + paymentmethodelectroniccheck + monthlycharges + totalcharges,
  data=cd, hidden=20, threshold=0.01, rep = 10
)

plot(r)
```

```{r}
Predict <- compute(r, test)

preds <- Predict$net.result

preds <- rep("No", length(preds))

preds[preds > 0.5] <- "Yes"

cd1 <- data.frame(Churn = cd$churn, predicted = preds)

xtabs(~ predicted + Churn, data = cd1)



```

## Using effect plots to detemine which regressors to use for logistic reg.
```{r, fig.width = 10, fig.height = 8}
# Making effect plots and omitting certain variables
# Splitting into three plots so that it is more readable
# Create the three formulas
formula_1 <- as.formula(paste("churn ~ ", paste(colnames(cd1)[c(2:5, 7:11)], collapse = " + ")))
formula_2 <- as.formula(paste("churn ~ ", paste(colnames(cd1)[c(12:17, 19, 22, 23)], collapse = " + ")))
formula_3 <- as.formula(paste("churn ~ ", paste(colnames(cd1)[c(24, 25)], collapse = " + ")))
# Creating the logistic regression models
lrfit1 <- glm(formula_1, cd1, family = binomial(link = "logit"))
lrfit2 <- glm(formula_2, cd1, family = binomial(link = "logit"))
lrfit3 <- glm(formula_3, cd1, family = binomial(link = "logit"))
# Making the plots; third plot is in the next code chunk
plot(allEffects(lrfit1), type = "response")
plot(allEffects(lrfit2), type = "response")
#plot(allEffects(lrfit3), type = "response")
```

```{r, fig.width = 8, fig.height = 4}
plot(allEffects(lrfit3), type = "response")
```






