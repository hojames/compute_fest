########### final test only
library(readr)
#X_train_updt <- read_csv("~/ownCloud/Scripts/computefest/data/X_train_updt.csv")
X_test_updt <- read_csv("~/ownCloud/Scripts/computefest/data/X_test_updt.csv")

X_complete <- X_test_updt

X_complete$X1 <- NULL

ids <- X_test_updt$Doctor.Identifier

X_complete$Specialty.Category <- as.factor(X_complete$Specialty.Category)
X_complete$Provider.Type <- NULL
X_complete$Freq <- NULL

X_complete$Doctor.Identifier <- NULL

# remove percentages
X_complete$Percent.Alzheimer.s.Disease.or.Dementia <- NULL
X_complete$Percent.Atrial.Fibrillation <- NULL
X_complete$Percent.Asthma <- NULL
X_complete$Percent.Cancer <- NULL
X_complete$Percent.Heart.Failure <- NULL
X_complete$Percent.Chronic.Kidney.Disease <- NULL
X_complete$Percent.Chronic.Obstructive.Pulmonary.Disease <- NULL
X_complete$Percent.Depression <- NULL
X_complete$Percent.Diabetes <-NULL
X_complete$Percent.Hyperlipidemia <- NULL
X_complete$Percent.Hypertension <- NULL
X_complete$Percent.Ischemic.Heart.Disease <- NULL
X_complete$Percent.Osteoporosis <- NULL
X_complete$Percent.Rheumatoid.Arthritis.or.Osteoarthritis <- NULL
X_complete$Percent.Schizophrenia.or.Other.Psychotic.Disorders <- NULL
X_complete$Percent.Stroke <- NULL

# generate new variables
X_complete$AllowPayFrac <- X_complete$Total.Allowed.Amount / X_complete$Total.Payment.Amount
X_complete$PayFrac <- X_complete$Total.Payment.Amount / X_complete$Total.Standardized.Payment.Amount

library(h2o)
h2o.init()
# generate deep features
h2o.df <- as.h2o(X_complete)
feature_names <- colnames(h2o.df)
model_nn <- h2o.deeplearning(x = feature_names,
                             training_frame = h2o.df,
                             model_id = "model_nn3",
                             autoencoder = TRUE,
                             reproducible = FALSE, #slow - turn off for real problems
                             ignore_const_cols = FALSE,
                             seed = 43,
                             hidden = c(50, 10, 2, 50), 
                             epochs = 100,
                             l1 = 0.0001,
                             activation = "TanhWithDropout")

feats <- h2o.deepfeatures(model_nn, h2o.df, layer = 3)

X_iso <- cbind(X_complete, as.data.frame(feats))

library(isofor)
mod <- iForest(X = X_iso, 100, 40)

# generate deep features

p <- predict(mod, X_iso)

sub8 <- data.frame(`Doctor.Identifier` = ids,
                   Risk = as.vector(p))
write.csv(sub8, "~/ownCloud/Scripts/computefest/data/subtest.csv", row.names=F)

