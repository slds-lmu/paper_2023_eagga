
# EA GGA

``` r
library(mlr3)
library(mlr3learners)
library(mlr3pipelines)
library(mlr3tuning)
library(eagga)

task = tsk("spam")

resampling = rsmp("cv", folds = 3)

learner = as_learner(po("colapply") %>>% po("select") %>>% po("sortfeatures") %>>% lrn("classif.xgboost"))
learner$param_set$values$classif.xgboost.booster = "gbtree"
learner$param_set$values$classif.xgboost.tree_method = "exact"
learner$param_set$values$colapply.applicator = function(x) - x

measures = list(msr("classif.ce"),
                msr("selected_features_proxy"),
                msr("selected_interactions_proxy"),
                msr("selected_non_monotone_proxy"))

terminator = trm("evals", n_evals = 220)

search_space = ps(
  classif.xgboost.nrounds = p_dbl(lower = log(1), upper = log(5000), tags = c("int", "log"),
                                  trafo = function(x) as.integer(round(exp(x))), default = log(500)),
  classif.xgboost.eta = p_dbl(lower = log(1e-4), upper = log(1), tags = "log",
                              trafo = function(x) exp(x), default = log(0.3)),
  classif.xgboost.gamma = p_dbl(lower = log(1e-4), upper = log(7), tags = "log",
                                trafo = function(x) exp(x), default = log(1e-4)),
  classif.xgboost.lambda = p_dbl(lower = log(1e-4), upper = log(1000), tags = "log",
                                 trafo = function(x) exp(x), default = log(1)),
  classif.xgboost.alpha = p_dbl(lower = log(1e-4), upper = log(1000), tags = "log",
                                trafo = function(x) exp(x), default = log(1e-4)),
  classif.xgboost.subsample = p_dbl(lower = 0.1, upper = 1, default = 1),
  classif.xgboost.max_depth = p_int(lower = 1L, upper = 20L, default = 6L),
  classif.xgboost.min_child_weight = p_dbl(lower = log(1), upper = log(150), tags = "log",
                                           trafo = function(x) exp(x), default = log(exp(1))),
  classif.xgboost.colsample_bytree = p_dbl(lower = 0.01, upper = 1, default = 1),
  classif.xgboost.colsample_bylevel = p_dbl(lower = 0.01, upper = 1, default = 1),
  select.selector = p_uty(),  # must be part of the search space
  classif.xgboost.interaction_constraints = p_uty(),  # must be part of the search space
  classif.xgboost.monotone_constraints = p_uty()  # must be part of the search space
)

instance = TuningInstanceMultiCrit$new(
  task = task,
  learner = learner,
  resampling = resampling, 
  measures = measures,
  terminator = terminator,
  search_space = search_space,
  store_models = TRUE
)

tuner = tnr("eagga", mu = 20, lambda = 10)
tuner$optimize(instance)
```
