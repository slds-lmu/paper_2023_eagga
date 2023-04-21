MonotonicityDetector = R6Class("MonotonicityDetectorDetector",
  public = list(
    initialize = function(task) {
      assert_task(task, feature_types = c("integer", "numeric"))
      feature_names = task$feature_names
      feature_types = task$feature_types
      y = task$data(cols = task$target_names)[[1L]]  # regardless of regr or classif
      n_features = length(feature_names)
      self$task = task
      self$classification = task$task_type == "classif"
      self$data = task$data()
      self$n_features = n_features
      self$feature_names = feature_names
      self$feature_types = feature_types
      self$y_name = task$target_names
      self$rho_table = data.table(feature_name = feature_names, rho = numeric(n_features))
      self$unconstrained_weight_table = data.table(feature_name = feature_names, unconstrained_weight = numeric(n_features))
    },

    task = NULL,
    classification = NULL,
    data = NULL,
    n_features = NULL,
    feature_names = NULL,
    feature_types = NULL,
    y_name = NULL,
    rho_table = NULL,
    unconstrained_weight_table = NULL,

    compute = function() {
      pb = progress_bar$new(format = "Detecting monotonicity [:bar] :percent eta: :eta", total = length(self$feature_names))
      for (x_name in self$feature_names) {
        pb$tick()
        private$.compute_rho(x_name)
      }
      self$rho_table[is.na(rho), rho := 0]
    },

    get_sign = function(feature_name) {
      x_name = assert_choice(feature_name, choices = self$feature_names)
      as.integer(sign(self$rho_table[feature_name == x_name, rho]))
    },

    compute_unconstrained_weights = function() {
      self$unconstrained_weight_table = self$rho_table
      colnames(self$unconstrained_weight_table) = c("feature_name", "unconstrained_weight")
      self$unconstrained_weight_table[, unconstrained_weight := 1 - abs(unconstrained_weight)]
      self$unconstrained_weight_table[, unconstrained_weight := ((unconstrained_weight - 0) / (1 - 0)) * (0.8 - 0.2) +  0.2]  # bound by [0.2, 0.8]
    }
  ),

  private = list(
    .compute_rho = function(x_name, repls = 10L) {
    task = self$task$clone(deep = TRUE)
    if (task$task_type == "classif") {
      learner = as_learner(po("subsample", frac = 0.9) %>>% lrn("classif.rpart"))
      learner$predict_type = "prob"
    } else if (task$task_type == "regr") {
      learner = as_learner(po("subsample", frac = 0.9) %>>% lrn("regr.rpart"))
    }
    task$col_roles$feature = x_name
    rho_repls = map_dbl(seq_len(repls), function(repl) {
      learner$train(task)
      pred = learner$predict(task)
      if (task$task_type == "classif") {
        y = pred$prob[, 1L]
      } else if (task$task_type == "regr") {
        y = pred$response
      }
      cor(y, task$data(cols = x_name)[[1L]], method = "spearman")
    })

      self$rho_table[feature_name == x_name, rho := mean(rho_repls)]
    }
  )
)

### test
if (FALSE) {
  task = tsk("spam")
  detector = MonotonicityDetector$new(task)
  detector$compute()
  detector$rho_table
  detector$get_sign("address")
  detector$compute_unconstrained_weights()
  detector$unconstrained_weight_table
}
