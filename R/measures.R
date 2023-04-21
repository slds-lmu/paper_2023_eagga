#' @title Selected Features Proxy
#'
#' @name mlr_measures_selected_features_proxy
#'
#' @description
#' Proxy measure that simply returns 0.
#' Can be used as a proxy measure which is updated within an optimizer.
#' Only to be used internally.
#'
#' @templateVar id selected_features_proxy
#' @template measure
#'
#' @export
MeasureSelectedFeaturesProxy = R6Class("MeasureSelectedFeaturesProxy",
  inherit = mlr3::Measure,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {

      super$initialize(
        id = "selected_features_proxy",
        param_set = ps(),
        task_type = NA_character_,
        properties = NA_character_,
        predict_type = "response",
        range = c(0, Inf),
        minimize = TRUE,
        label = "Proxy",
        man = "eagga::mlr_measures_selected_features_proxy"
      )
    }
  ),

  private = list(
    .score = function(...) {
      0
    }
  )
)

#' @title Selected Interactions Proxy
#'
#' @name mlr_measures_selected_interactions_proxy
#'
#' @description
#' Proxy measure that simply returns 0.
#' Can be used as a proxy measure which is updated within an optimizer.
#' Only to be used internally.
#'
#' @templateVar id selected_interactions_proxy
#' @template measure
#'
#' @export
MeasureSelectedInteractionsProxy = R6Class("MeasureSelectedInteractionsProxy",
  inherit = mlr3::Measure,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {

      super$initialize(
        id = "selected_interactions_proxy",
        param_set = ps(),
        task_type = NA_character_,
        properties = NA_character_,
        predict_type = "response",
        range = c(0, Inf),
        minimize = TRUE,
        label = "Proxy",
        man = "eagga::mlr_measures_selected_interactions_proxy"
      )
    }
  ),

  private = list(
    .score = function(...) {
      0
    }
  )
)

#' @title Selected Non Monotone Proxy
#'
#' @name mlr_measures_selected_non_monotone_proxy
#'
#' @description
#' Proxy measure that simply returns 0.
#' Can be used as a proxy measure which is updated within an optimizer.
#' Only to be used internally.
#'
#' @templateVar id selected_non_monotone_proxy
#' @template measure
#'
#' @export
MeasureSelectedNonMonotoneProxy = R6Class("MeasureSelectedNonMonotoneProxy",
  inherit = mlr3::Measure,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {

      super$initialize(
        id = "selected_non_monotone_proxy",
        param_set = ps(),
        task_type = NA_character_,
        properties = NA_character_,
        predict_type = "response",
        range = c(0, Inf),
        minimize = TRUE,
        label = "Proxy",
        man = "eagga::mlr_measures_selected_non_monotone_proxy"
      )
    }
  ),

  private = list(
    .score = function(...) {
      0
    }
  )
)

