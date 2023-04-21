#' @importFrom R6 R6Class
#' @import checkmate
#' @import data.table
#' @import paradox
#' @import mlr3misc
#' @import bbotk
#' @import lgr
#' @import mlr3
#' @import mlr3learners
#' @import mlr3pipelines
#' @import mlr3tuning
#' @import relations
#' @import mlr3filters
#' @import xgboost
#' @import progress
#' @importFrom stats setNames runif dnorm pnorm rnorm rgeom var
#' @importFrom utils stack
#' @importFrom R.utils withTimeout

.onLoad = function(libname, pkgname) { # nolint
  # nocov start
  backports::import(pkgname)

  # add eagga to tuner dictionary
  x = utils::getFromNamespace("mlr_tuners", ns = "mlr3tuning")
  x$add("eagga", TunerEAGGA)
  x$add("eagga_ablation", TunerEAGGAX)


  # add measures dictionary
  x = utils::getFromNamespace("mlr_measures", ns = "mlr3")
  x$add("selected_features_proxy", function() MeasureSelectedFeaturesProxy$new())
  x$add("selected_interactions_proxy", function() MeasureSelectedInteractionsProxy$new())
  x$add("selected_non_monotone_proxy", function() MeasureSelectedNonMonotoneProxy$new())

  # add sortfeatures to pipelines dictionary
  x = utils::getFromNamespace("mlr_pipeops", ns = "mlr3pipelines")
  x$add("sortfeatures", PipeOpSortFeatures)

  # setup logger
  assign("lg", lgr::get_logger("bbotk"), envir = parent.env(environment()))

  if (Sys.getenv("IN_PKGDOWN") == "true") {
    lg$set_threshold("warn")
  }
} # nocov end

# static code checks should not complain about commonly used data.table columns
utils::globalVariables(c("id", "Parent", "Child", "var", "stack", "Child", "Tree", "Feature", "Node", "Quality", "Yes", "No", "Missing", "ID", "isLeaf", "feature"))
