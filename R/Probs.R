Probs = R6Class("Probs",
  public = list(
    p_overall_mutate = NULL,              # overall prob to apply mutation to param and groupstructure (at all)
    p_overall_crossover = NULL,           # overall prob to apply crossover to param and groupstructure (at all)
    p_param_mutate = NULL,                # individual prob for each param to mutate
    p_param_crossover = NULL,             # individual prob for each param to crossover
    p_groupstructure_mutate = NULL,       # individual prob for each feature in groupstructure to mutate
    p_groupstructure_crossover = NULL,    # same as p_overall_crossover because groupstructure crossover works global

    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function(p_overall_mutate = 0.3, p_overall_crossover = 0.7, p_param_mutate = 0.2, p_param_crossover = 0.5, p_groupstructure_mutate = 0.2) {
      self$p_overall_mutate = assert_number(p_overall_mutate, lower = 0, upper = 1)
      self$p_overall_crossover = assert_number(p_overall_crossover, lower = 0, upper = 1)
      self$p_param_mutate = assert_number(p_param_mutate, lower = 0, upper = 1)
      self$p_param_crossover = assert_number(p_param_crossover, lower = 0, upper = 1)
      self$p_groupstructure_mutate = assert_number(p_groupstructure_mutate, lower = 0, upper = 1)
      self$p_groupstructure_crossover = p_overall_crossover
    },

    update = function() {
    }
  )
)

