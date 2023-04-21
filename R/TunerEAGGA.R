#' @title Multi-objective Hyperparameter Optimization, Feature Selection and Interaction and Monotonicity Constraints
#'
#' @name mlr_tuner_eagga
#'
#' @description
#' Performs joint multi-objective optimization of hyperparameters, feature selection and interaction and monotonicity
#' constraints of a suitable [mlr3::Learner].
#'
#' This requires an appropriate [mlr3::Learner], that allows for selecting features, and setting interaction and
#' monotonicity constraints, e.g., xgboost.
#'
#' @templateVar id eagga
#' @template section_dictionary_tuners
#'
#' @section Parameters:
#' \describe{
#' \item{`select_id`}{`character(1)`\cr
#' ID of param in Learner that selects features.}
#' \item{`interaction_id`}{`character(1)`\cr
#' ID of param in Learner that sets interaction constraints.}
#' \item{`monotone_id`}{`character(1)`\cr
#' ID of param in Learner that sets monotonicity constraints.}
#' \item{`mu`}{`integer(1)`\cr
#' Population size.}
#' \item{`lambda`}{`integer(1)`\cr
#' Offspring size of each generation.}}
#'
#' @template section_progress_bars
#' @template section_logging
#'
#' @family Tuner
#'
#' @export
TunerEAGGA = R6Class("TunerEAGGA",
  inherit = mlr3tuning::Tuner,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      param_set = ps(
        select_id = p_uty(tags = "required"),
        interaction_id = p_uty(tags = "required"),
        monotone_id = p_uty(tags = "required"),
        mu = p_int(tags = "required"),
        lambda = p_int(tags = "required")
      )
      param_set$values = list(select_id = "select.selector", interaction_id = "classif.xgboost.interaction_constraints", monotone_id = "classif.xgboost.monotone_constraints")
      super$initialize(
        param_set = param_set,
        param_classes = c("ParamDbl", "ParamFct", "ParamInt", "ParamLgl", "ParamUty"),
        properties = "multi-crit",
        packages = "eagga",
        label = "Joint HPO and Optimization of Feature Selection, Interaction Constraints and Monotonicity Constraints",
        man = "eagga::mlr_tuners_eagga"
      )
    }
  ),

  private = list(
    .optimize = function(inst) {
      # FIXME: assert that select_id, interaction_id, monotone_id are part of the search space
      select_id = self$param_set$values$select_id
      interaction_id = self$param_set$values$interaction_id
      monotone_id = self$param_set$values$monotone_id
      mu = self$param_set$values$mu
      lambda = self$param_set$values$lambda

      # probs for mutation and crossover
      probs = Probs$new()
      private$.probs = probs

      # split param space from sIm space
      param_ids = setdiff(inst$search_space$ids(), c(select_id, interaction_id, monotone_id))
      param_space = ParamSet$new(inst$search_space$params[param_ids])
      param_space$trafo = inst$search_space$trafo
      param_space$deps = inst$search_space$deps
      task = inst$objective$task

      # initial population
      population = map_dtr(seq_len(mu), function(i) param_space$default)  # NOTE: for now, we use the initial design not random but defaults
      # we then mutate the initial design like during the GA but with p = 1 for each param (except the mu-th one)
      for (j in seq_len(nrow(population) - 1L)) {
        for (param_id in param_ids) {
          population[j, ][[param_id]] = mutate(population[j, ][[param_id]], param = param_space$params[[param_id]], p = 1)  # mutate the defaults with probability 1
        }
      }

      n_selected_prob = 1 / get_n_selected_rpart(task)
      n_selected = replicate(mu, sample_from_truncated_geom(n_selected_prob, lower = 1L, upper = length(task$feature_names)))  # number of selected features sampled from truncated geometric distribution
      filter = FilterInformationGain$new()  # NOTE: can use any other Filter or use a custom FilterEnsemble
      scores = as.data.table(filter$calculate(task))  # filter scores are used to weight probabilities of inclusion in GroupStructure
      scores[, score := score / sum(score)]
      scores[score < .Machine$double.eps, score := .Machine$double.eps]
      interaction_detector = InteractionDetector$new(task)  # interaction detection
      interaction_detector$compute_best_rss()
      monotonicity_detector = MonotonicityDetector$new(task)  # monotonicity detection
      monotonicity_detector$compute()
      monotonicity_detector$compute_unconstrained_weights()
      unconstrained_weight_table = monotonicity_detector$unconstrained_weight_table
      switch_sign_affected = task$feature_names[map_lgl(task$feature_names, function(feature_name) monotonicity_detector$get_sign(feature_name) == -1L)]
      inst$objective$learner$param_set$values$colapply.affect_columns = selector_name(switch_sign_affected)

      private$.n_selected_prob = n_selected_prob
      private$.n_selected = n_selected
      private$.filter = filter
      private$.scores = scores
      private$.interaction_detector = interaction_detector
      private$.monotonicity_detector = monotonicity_detector
      private$.unconstrained_weight_table = unconstrained_weight_table
      private$.switch_sign_affected = switch_sign_affected

      sIm = map_dtr(seq_len(mu - 1L), function(i) {  # sIm space
        groupstructure = GroupStructure$new(task, n_selected = n_selected[i], scores = scores, interaction_detector = interaction_detector, unconstrained_weight_table = unconstrained_weight_table)
        data.table(groupstructure = list(groupstructure),
                   s = list(groupstructure$create_selector()),
                   I = list(groupstructure$create_interaction_constraints()),
                   m = list(groupstructure$create_monotonicity_constraints()))
      })
      # as the mu-th add the unconstrained sIm point (to make sure we also have the most complex sIm)
      groupstructure_unconstrained = GroupStructure$new(task, n_selected = length(task$feature_names), scores = scores, interaction_detector = interaction_detector, unconstrained_weight_table = unconstrained_weight_table, unconstrained = TRUE)
      sIm_unconstrained = data.table(groupstructure = list(groupstructure_unconstrained),
                                     s = list(groupstructure_unconstrained$create_selector()),
                                     I = list(groupstructure_unconstrained$create_interaction_constraints()),
                                     m = list(groupstructure_unconstrained$create_monotonicity_constraints()))
      sIm = rbind(sIm, sIm_unconstrained)
      colnames(sIm) = c("groupstructure", select_id, interaction_id, monotone_id)

      population = cbind(population, sIm)
      gen = 0
      population[, generation := gen]
      population[, status := "alive"]

      # evaluate initial population
      # proxy measures for selected features, interactions and non monotone are evaluated here
      learner_for_measures = inst$objective$learner$clone(deep = TRUE)
      orig_pvs = learner_for_measures$param_set$values
      for (i in seq_len(nrow(population))) {
        inst$eval_batch(population[i, ])
        # NOTE: cannot use i but we can use inst$archive$n_batch (due to synchronous evaluation)
        j = inst$archive$n_batch

        # NOTE: this messes with logging (proxy measures are logged and the updated ones are not logged)
        # actually evaluate the proxy measures
        proxy_measures = calculate_proxy_measures(learner_for_measures, task = task, orig_pvs = orig_pvs, xdt = inst$archive$data[j, inst$archive$cols_x, with = FALSE], search_space = inst$search_space)
        inst$archive$data[j, selected_features_proxy := proxy_measures$n_selected]
        inst$archive$data[j, selected_interactions_proxy := proxy_measures$n_interactions]
        inst$archive$data[j, selected_non_monotone_proxy := proxy_measures$n_non_monotone]

        # update the groupstructure (x space)
        groupstructure = inst$archive$data[j, ][["groupstructure"]][[1L]]$clone(deep = TRUE)
        groupstructure_orig = groupstructure$clone(deep = TRUE)
        set(inst$archive$data, i = j, j = "groupstructure_orig", value = list(groupstructure_orig))
        groupstructure = update_sIm(groupstructure, used = proxy_measures$used, belonging = proxy_measures$belonging)

        inst$archive$data[j, ][["groupstructure"]][[1L]] = groupstructure
        inst$archive$data[j, ][[select_id]][[1L]] = groupstructure$create_selector()
        inst$archive$data[j, ][[interaction_id]][[1L]] = groupstructure$create_interaction_constraints()
        inst$archive$data[j, ][[monotone_id]][[1L]] = groupstructure$create_monotonicity_constraints()
      }

      # groupstructure in the population with zero selection are killed
      zero_selected = map_lgl(inst$archive$data[[select_id]], function(selector) get_number_of_selected_features_from_selector(selector, task = task) == 0L)
      inst$archive$data[zero_selected, status := "dead"]

      repeat {  # iterate until we have an exception from eval_batch

        probs$update()

        zero_selected = map_lgl(inst$archive$data[[select_id]], function(selector) get_number_of_selected_features_from_selector(selector, task = task) == 0L)
        stopifnot(all(inst$archive$data[zero_selected, ][["status"]] == "dead"))

        # new gen, new individuals that are still alive
        gen = gen + 1
        data = inst$archive$data[, inst$archive$cols_y, with = FALSE]
        stopifnot(colnames(data) == inst$objective$codomain$ids())

        ys = t(t(data) * mult_max_to_min(inst$objective$codomain))
        alive_ids = which(inst$archive$data$status == "alive")

        # create children
        # binary tournament selection of parents
        # FIXME: if alive ids is < 1 (due to zero selection being killed) simply generate children at random but this is unlikely to happen
        children = map_dtr(seq_len(ceiling(lambda / 2)), function(i) {
          parent_id1 = binary_tournament(ys, alive_ids)
          parent_id2 = binary_tournament(ys, alive_ids)
          parents = transpose_list(copy(inst$archive$data[c(parent_id1, parent_id2), c("groupstructure", inst$archive$cols_x), with = FALSE]))

          # param_space
          # Uniform crossover for HPs
          if (runif(1L, min = 0, max = 1) <= probs$p_overall_crossover) {
            param_ids_to_cross = param_ids[which(runif(length(param_ids), min = 0, max = 1) <= probs$p_param_crossover)]
            tmp = parents[[1L]]
            for (param_id in param_ids_to_cross) {
              parents[[1L]][[param_id]] = parents[[2L]][[param_id]]
              parents[[2L]][[param_id]] = tmp[[param_id]]
            }
          }
          # Gaussian or uniform discrete mutation for HPs
          for (j in 1:2) {
            if (runif(1L, min = 0, max = 1) <= probs$p_overall_mutate) {
              for (param_id in param_ids) {
                parents[[j]][[param_id]] = mutate(parents[[j]][[param_id]], param = param_space$params[[param_id]], p = probs$p_param_mutate)
              }
            }
          }

          # sIm space
          # crossover and mutation via GGA
          groupstructure1 = parents[[1L]][["groupstructure"]]$clone(deep = TRUE)
          groupstructure2 = parents[[2L]][["groupstructure"]]$clone(deep = TRUE)

          if (runif(1L, min = 0, max = 1) <= probs$p_groupstructure_crossover) {  # this is the same as probs$p_overall_crossover
            crossing_sections = groupstructure1$get_crossing_sections(groupstructure2)
            tmp = groupstructure1$clone(deep = TRUE)
            groupstructure1$crossover(groupstructure2, crossing_sections = crossing_sections)
            groupstructure2$crossover(tmp, crossing_sections = rev(crossing_sections))
          }

          if (runif(1L, min = 0, max = 1) <= probs$p_overall_mutate) {
            groupstructure1$mutate(p = probs$p_groupstructure_mutate)
          }

          if (runif(1L, min = 0, max = 1) <= probs$p_overall_mutate) {
            groupstructure2$mutate(p = probs$p_groupstructure_mutate)
          }

          parents[[1L]][["groupstructure"]] = groupstructure1
          parents[[1L]][[select_id]] = groupstructure1$create_selector()
          parents[[1L]][[interaction_id]] = groupstructure1$create_interaction_constraints()
          parents[[1L]][[monotone_id]] = groupstructure1$create_monotonicity_constraints()

          parents[[2L]][["groupstructure"]] = groupstructure2
          parents[[2L]][[select_id]] = groupstructure2$create_selector()
          parents[[2L]][[interaction_id]] = groupstructure2$create_interaction_constraints()
          parents[[2L]][[monotone_id]] = groupstructure2$create_monotonicity_constraints()

          parents_table = rbindlist(map(parents, function(parent) parent[param_ids]))
          set(parents_table, j = "groupstructure", value = map(parents, function(parent) parent[["groupstructure"]]))
          parents_table[, eval(select_id) := map(parents, function(parent) parent[[select_id]])]
          parents_table[, eval(interaction_id) := map(parents, function(parent) parent[[interaction_id]])]
          parents_table[, eval(monotone_id) := map(parents, function(parent) parent[[monotone_id]])]

          stopifnot(setequal(colnames(parents_table), names(parents[[1L]])))
          parents_table
        })
        children[, generation := gen]
        children = children[seq_len(lambda), ]  # restrict to lambda children

        # evaluate children 
        for (i in seq_len(nrow(children))) {
          inst$eval_batch(children[i, ])
          # NOTE: cannot use i but we can use inst$archive$n_batch (due to synchronous evaluation)
          j = inst$archive$n_batch

          # actually evaluate the proxy measures
          # learner_for_measures and orig_pvs have been defined above (eval of initial population)
          proxy_measures = calculate_proxy_measures(learner_for_measures, task = task, orig_pvs = orig_pvs, xdt = inst$archive$data[j, inst$archive$cols_x, with = FALSE], search_space = inst$search_space)
          inst$archive$data[j, selected_features_proxy := proxy_measures$n_selected]
          inst$archive$data[j, selected_interactions_proxy := proxy_measures$n_interactions]
          inst$archive$data[j, selected_non_monotone_proxy := proxy_measures$n_non_monotone]

          # update the groupstructure (x space)
          groupstructure = inst$archive$data[j, ][["groupstructure"]][[1L]]$clone(deep = TRUE)
          groupstructure_orig = groupstructure$clone(deep = TRUE)
          set(inst$archive$data, i = j, j = "groupstructure_orig", value = list(groupstructure_orig))
          groupstructure = update_sIm(groupstructure, used = proxy_measures$used, belonging = proxy_measures$belonging)

          inst$archive$data[j, ][["groupstructure"]][[1L]] = groupstructure
          inst$archive$data[j, ][[select_id]][[1L]] = groupstructure$create_selector()
          inst$archive$data[j, ][[interaction_id]][[1L]] = groupstructure$create_interaction_constraints()
          inst$archive$data[j, ][[monotone_id]][[1L]] = groupstructure$create_monotonicity_constraints()
        }

        # NSGA-II stuff for survival
        # mu + lambda
        all_ids = seq_len(nrow(inst$archive$data))
        # groupstructure with zero selection are not allowed to be selected
        zero_selected = map_lgl(inst$archive$data[[select_id]], function(selector) get_number_of_selected_features_from_selector(selector, task = task) == 0L)
        considered_ids = all_ids[!zero_selected]
        if (length(considered_ids) <= mu) {
          inst$archive$data[, status := "dead"]
          inst$archive$data[considered_ids, status := "alive"]
        } else {
          ys = t(t(inst$archive$data[considered_ids, inst$archive$cols_y, with = FALSE]) * mult_max_to_min(inst$objective$codomain))
          rankings = emoa::nds_rank(t(ys))  # non-dominated fronts
          cds = map_dtr(unique(rankings), function(ranking) {  # crowding distances
            ids_rank = which(rankings == ranking)
            data.table(id = ids_rank, cd = emoa::crowding_distance(t(ys[ids_rank, , drop = FALSE])))
          })
          setorderv(cds, "id")
          stopifnot(setequal(cds$id, seq_len(nrow(cds))))
          alive_ids = integer(mu)
          current_front = 0L
          while(sum(alive_ids == 0L) != 0) {
            current_front = current_front + 1L
            candidate_ids = which(rankings == current_front)
            to_insert = sum(alive_ids == 0L)
            if (length(candidate_ids) <= to_insert) {
              alive_ids[alive_ids == 0][seq_along(candidate_ids)] = candidate_ids
            } else {
              alive_ids[alive_ids == 0] = candidate_ids[order(cds[candidate_ids, ]$cd, decreasing = TRUE)][seq_len(to_insert)]
            }
          }
          stopifnot(length(unique(alive_ids)) == length(alive_ids))
          inst$archive$data[, status := "dead"]
          inst$archive$data[considered_ids[alive_ids], status := "alive"]  # alive_ids was determined on the ys that did not carry the considered_ids info
        }
      }

      inst
    },
    .probs = NULL,
    .n_selected_prob = NULL,
    .n_selected = NULL,
    .filter = NULL,
    .scores = NULL,
    .interaction_detector = NULL,
    .monotonicity_detector = NULL,
    .unconstrained_weight_table = NULL,
    .switch_sign_affected = NULL
  )
)

# mutation function for traditional parameters
mutate = function(value, param, p = 0.2, sigma = 0.1) {
  stopifnot(param$class %in% c("ParamDbl", "ParamFct", "ParamInt", "ParamLgl"))
  if (runif(1L, min = 0, max = 1) >= p) {
    return(value)  # early exit
  }
  if (param$class %in% c("ParamDbl", "ParamInt")) {
    value_ = (value - param$lower) / (param$upper - param$lower)
    value_ = max(0, min(stats::rnorm(1L, mean = value_, sd = sigma), 1))
    value = (value_ * (param$upper - param$lower)) + param$lower
    if (param$class == "ParamInt") {
      value = round(value, 0L)
    }
    value = min(max(value, param$lower), param$upper)
  } else if (param$class %in% c("ParamFct", "ParamLgl")) {
    value = sample(setdiff(param$levels, value), size = 1L)
  }
  value
}

# function for binary tournament selection of parents via non-dominated sorting and crowding distance
binary_tournament = function(ys, alive_ids) {
  ids = sample(alive_ids, size = 2L, replace = FALSE)
  rankings = emoa::nds_rank(t(ys))  # non-dominated fronts
  if (rankings[ids][1L] != rankings[ids][2L]) {
    return(ids[order(rankings[ids])[1L]])  # early exit
  }
  cds = map_dtr(unique(rankings), function(ranking) {  # crowding distances
    ids_rank = which(rankings == ranking)
    data.table(id = ids_rank, cd = emoa::crowding_distance(t(ys[ids_rank, , drop = FALSE])))
  })
  ids[order(cds[id %in% ids]$cd, decreasing = TRUE)[1L]]
}

