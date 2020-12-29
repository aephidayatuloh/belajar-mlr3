# https://github.com/aephidayatuloh/belajar-mlr3

library("data.table")
library("mlr3")
library("mlr3learners")
library("mlr3viz")
library("ggplot2")

data("german", package = "rchallenge")

dim(german)
str(german)

skimr::skim(german)

task <- TaskClassif$new("GermanCredit", german, target = "credit_risk")

mlr_learners

library("mlr3learners")
learner_logreg <- lrn("classif.log_reg")
print(learner_logreg)

learner_logreg$train(task)

train_set = sample(task$row_ids, 0.8 * task$nrow)
test_set = setdiff(task$row_ids, train_set)

head(train_set)

learner_logreg$train(task, row_ids = train_set)
learner_logreg$model

class(learner_logreg$model)
summary(learner_logreg$model)

learner_rf <- lrn("classif.ranger", importance = "permutation")
learner_rf$train(task, row_ids = train_set)

learner_rf$importance()
importance <- as.data.table(learner_rf$importance(), keep.rownames = TRUE)

colnames(importance) = c("Feature", "Importance")

ggplot(importance, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col() + 
  coord_flip() + 
  xlab("")

pred_logreg <- learner_logreg$predict(task, row_ids = test_set)
pred_rf <- learner_rf$predict(task, row_ids = test_set)

pred_logreg
pred_rf

pred_logreg$confusion
pred_rf$confusion

learner_logreg$predict_type = "prob"
learner_logreg$predict(task, row_ids = test_set)

resampling <- rsmp("holdout", ratio = 2/3)
print(resampling)
res <- resample(task, learner = learner_logreg, resampling = resampling)
res
res$aggregate()

resampling <- rsmp("subsampling", repeats = 10)
rr <- resample(task, learner = learner_logreg, resampling = resampling)
rr$aggregate()

resampling <- rsmp("cv", folds = 10)
rr <- resample(task, learner = learner_logreg, resampling = resampling)
rr$aggregate()

# false positive rate
rr$aggregate(msr("classif.fpr"))

measures <- msrs(c("classif.fpr", "classif.fnr"))
rr$aggregate(measures)

mlr_resamplings

learners <- lrns(c("classif.log_reg", "classif.ranger"), predict_type = "prob")
bm_design <- benchmark_grid(
  tasks = task,
  learners = learners,
  resamplings = rsmp("cv", folds = 10)
)
bmr <- benchmark(bm_design)

measures <- msrs(c("classif.ce", "classif.auc"))
performances <- bmr$aggregate(measures)
performances[, c("learner_id", "classif.ce", "classif.auc")]

learner_rf$param_set

rf_med <- lrn("classif.ranger", id = "med", predict_type = "prob")

rf_low <- lrn("classif.ranger", id = "low", predict_type = "prob",
             num.trees = 5, mtry = 2)

rf_high <- lrn("classif.ranger", id = "high", predict_type = "prob",
              num.trees = 1000, mtry = 11)

learners <- list(rf_low, rf_med, rf_high)
bm_design <- benchmark_grid(
  tasks = task,
  learners = learners,
  resamplings = rsmp("cv", folds = 10)
)
bmr <- benchmark(bm_design)
print(bmr)

measures <- msrs(c("classif.ce", "classif.auc"))
performances <- bmr$aggregate(measures)
performances[, .(learner_id, classif.ce, classif.auc)]
autoplot(bmr)
