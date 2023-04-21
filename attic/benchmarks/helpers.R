# for the outer resampling (holdout):
#   1. construct the train task (optim set based on outer resampling) and test task (test set based on outer resampling) and initialize the inner resampling (fixed seed) (train and valid set based on optim set)
#   2. perform tuning using the inner resampling on the train task and obtain pareto set on valid set of train task
#   3. refit pareto set on whole train task and evaluate on test task (based on outer resampling)
#   4. calculate the dominated hypervolume based on the valid set --> dhv_valid
#   5. calculate the dominated hypervolume based on the test set --> dhv_test
nested_resampling_eagga = function(task_train, task_test, resampling_inner, task_id_, repl_, secs = 12L * 3600L) {
  reference_point = t(t(c(minus_classif.auc = 0, selected_features = 1, selected_interactions = 1, selected_non_monotone = 1)))
  mu = 100L
  lambda = 10L

  learner = as_learner(po("colapply") %>>% po("select") %>>% po("sortfeatures") %>>% lrn("classif.xgboost"))
  learner$predict_type = "prob"
  learner$param_set$values$classif.xgboost.booster = "gbtree"
  learner$param_set$values$classif.xgboost.tree_method = "exact"
  learner$param_set$values$colapply.applicator = function(x) - x

  measures = list(msr("classif.auc"),
                  msr("selected_features_proxy"),
                  msr("selected_interactions_proxy"),
                  msr("selected_non_monotone_proxy"))

  terminator = trm("run_time", secs = secs)

  search_space = get_xgboost_search_space_eagga()

  instance = TuningInstanceMultiCrit$new(
    task = task_train,
    learner = learner,
    resampling = resampling_inner, 
    measures = measures,
    terminator = terminator,
    search_space = search_space,
    store_models = TRUE
  )

  tuner = tnr("eagga", mu = mu, lambda = lambda)
  tuner$optimize(instance)

  tuning_data = copy(instance$archive$data)
  tuning_pareto = copy(instance$archive$best())

  pareto = copy(tuning_pareto)
  learner_on_test = instance$objective$learner$clone(deep = TRUE)
  orig_pvs = instance$objective$learner$param_set$values
  for (p in seq_len(NROW(pareto))) {
    groupstructure = pareto[p, ][["groupstructure_orig"]][[1L]]
    xdt = copy(pareto[p, ])
    xdt[["groupstructure"]][[1L]] = groupstructure
    xdt[[tuner$param_set$values$select_id]][[1L]] = groupstructure$create_selector()
    xdt[[tuner$param_set$values$interaction_id]][[1L]] = groupstructure$create_interaction_constraints()
    xdt[[tuner$param_set$values$monotone_id]][[1L]] = groupstructure$create_monotonicity_constraints()
    xss = transform_xdt_to_xss(xdt, search_space = instance$search_space)[[1L]]
    xss = insert_named(orig_pvs, xss)
    learner_on_test$param_set$values = xss
    learner_on_test$train(task_train)
    pareto[p, auc_test := learner_on_test$predict(task_test)$score(msr("classif.auc"))]
    # proxy measures must not be updated because they were already determined on the task_train during tuning
  }

  root = here::here()
  majority_vote = readRDS(file.path(root, "attic", "benchmarks", "results", "eagga_majority_vote.rds"))[task_id == task_id_ & repl == repl_]

  # validation dominated hypervolume
  y_val = tuning_pareto[, c("classif.auc", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]
  y_val = rbind(y_val, majority_vote[, c("classif.auc", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")])
  dhv_val = emoa::dominated_hypervolume(t(y_val) * eagga:::mult_max_to_min(instance$objective$codomain), ref = reference_point)

  # test dominated hypervolume
  y_test = pareto[, c("auc_test", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]
  y_test = rbind(y_test, majority_vote[, c("auc_test", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")])
  colnames(y_test) = c("classif.auc", "selected_features", "selected_interactions", "selected_non_monotone")
  dhv_test = emoa::dominated_hypervolume(t(y_test) * eagga:::mult_max_to_min(instance$objective$codomain), ref = reference_point)

  # validation dominated hypervolume anytime
  dhv_val_anytime = map_dtr(seq_len(NROW(tuning_data)), function(p) {
    start = tuning_data[1L, ]$timestamp
    stop = tuning_data[p, ]$timestamp
    non_dominated = !is_dominated(t(tuning_data[seq_len(p), c("classif.auc", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]) * eagga:::mult_max_to_min(instance$objective$codomain))
    y_val = tuning_data[which(non_dominated), c("classif.auc", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]
    y_val = rbind(y_val, majority_vote[, c("classif.auc", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")])
    dhv_val = emoa::dominated_hypervolume(t(y_val) * eagga:::mult_max_to_min(instance$objective$codomain), ref = reference_point)
    data.table(iteration = p, dhv_val = dhv_val, runtime = as.numeric(stop - start, units = "secs"))
  })

  data.table(tuning_data = list(tuning_data), tuning_pareto = list(tuning_pareto), pareto = list(pareto), dhv_val = dhv_val, dhv_test = dhv_test, dhv_val_anytime = list(dhv_val_anytime), best_val = max(tuning_pareto$classif.auc), best_test = max(pareto$auc_test))
}

nested_resampling_eagga_md2 = function(task_train, task_test, resampling_inner, task_id_, repl_, secs = 12L * 3600L) {
  reference_point = t(t(c(minus_classif.auc = 0, selected_features = 1, selected_interactions = 1, selected_non_monotone = 1)))
  mu = 100L
  lambda = 10L

  learner = as_learner(po("colapply") %>>% po("select") %>>% po("sortfeatures") %>>% lrn("classif.xgboost"))
  learner$predict_type = "prob"
  learner$param_set$values$classif.xgboost.booster = "gbtree"
  learner$param_set$values$classif.xgboost.tree_method = "exact"
  learner$param_set$values$colapply.applicator = function(x) - x
  learner$param_set$values$classif.xgboost.max_depth = 2L

  measures = list(msr("classif.auc"),
                  msr("selected_features_proxy"),
                  msr("selected_interactions_proxy"),
                  msr("selected_non_monotone_proxy"))

  terminator = trm("run_time", secs = secs)

  search_space = get_xgboost_search_space_eagga_md2()

  instance = TuningInstanceMultiCrit$new(
    task = task_train,
    learner = learner,
    resampling = resampling_inner, 
    measures = measures,
    terminator = terminator,
    search_space = search_space,
    store_models = TRUE
  )

  tuner = tnr("eagga", mu = mu, lambda = lambda)
  tuner$optimize(instance)

  tuning_data = copy(instance$archive$data)
  tuning_pareto = copy(instance$archive$best())

  pareto = copy(tuning_pareto)
  learner_on_test = instance$objective$learner$clone(deep = TRUE)
  orig_pvs = instance$objective$learner$param_set$values
  for (p in seq_len(NROW(pareto))) {
    groupstructure = pareto[p, ][["groupstructure_orig"]][[1L]]
    xdt = copy(pareto[p, ])
    xdt[["groupstructure"]][[1L]] = groupstructure
    xdt[[tuner$param_set$values$select_id]][[1L]] = groupstructure$create_selector()
    xdt[[tuner$param_set$values$interaction_id]][[1L]] = groupstructure$create_interaction_constraints()
    xdt[[tuner$param_set$values$monotone_id]][[1L]] = groupstructure$create_monotonicity_constraints()
    xss = transform_xdt_to_xss(xdt, search_space = instance$search_space)[[1L]]
    xss = insert_named(orig_pvs, xss)
    learner_on_test$param_set$values = xss
    learner_on_test$train(task_train)
    pareto[p, auc_test := learner_on_test$predict(task_test)$score(msr("classif.auc"))]
    # proxy measures must not be updated because they were already determined on the task_train during tuning
  }

  root = here::here()
  majority_vote = readRDS(file.path(root, "attic", "benchmarks", "results", "eagga_majority_vote.rds"))[task_id == task_id_ & repl == repl_]

  # validation dominated hypervolume
  y_val = tuning_pareto[, c("classif.auc", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]
  y_val = rbind(y_val, majority_vote[, c("classif.auc", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")])
  dhv_val = emoa::dominated_hypervolume(t(y_val) * eagga:::mult_max_to_min(instance$objective$codomain), ref = reference_point)

  # test dominated hypervolume
  y_test = pareto[, c("auc_test", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]
  y_test = rbind(y_test, majority_vote[, c("auc_test", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")])
  colnames(y_test) = c("classif.auc", "selected_features", "selected_interactions", "selected_non_monotone")
  dhv_test = emoa::dominated_hypervolume(t(y_test) * eagga:::mult_max_to_min(instance$objective$codomain), ref = reference_point)

  # validation dominated hypervolume anytime
  dhv_val_anytime = map_dtr(seq_len(NROW(tuning_data)), function(p) {
    start = tuning_data[1L, ]$timestamp
    stop = tuning_data[p, ]$timestamp
    non_dominated = !is_dominated(t(tuning_data[seq_len(p), c("classif.auc", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]) * eagga:::mult_max_to_min(instance$objective$codomain))
    y_val = tuning_data[which(non_dominated), c("classif.auc", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]
    y_val = rbind(y_val, majority_vote[, c("classif.auc", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")])
    dhv_val = emoa::dominated_hypervolume(t(y_val) * eagga:::mult_max_to_min(instance$objective$codomain), ref = reference_point)
    data.table(iteration = p, dhv_val = dhv_val, runtime = as.numeric(stop - start, units = "secs"))
  })

  data.table(tuning_data = list(tuning_data), tuning_pareto = list(tuning_pareto), pareto = list(pareto), dhv_val = dhv_val, dhv_test = dhv_test, dhv_val_anytime = list(dhv_val_anytime), best_val = max(tuning_pareto$classif.auc), best_test = max(pareto$auc_test))
}

nested_resampling_xgboost = function(task_train, task_test, resampling_inner, ..., secs = 12L * 3600L) {
  learner = as_learner(po("sortfeatures") %>>% lrn("classif.xgboost"))
  learner$predict_type = "prob"
  learner$param_set$values$classif.xgboost.booster = "gbtree"
  learner$param_set$values$classif.xgboost.tree_method = "exact"

  measure = msr("classif.auc")

  terminator = trm("run_time", secs = secs)

  search_space = get_xgboost_search_space()

  instance = TuningInstanceSingleCrit$new(
    task = task_train,
    learner = learner,
    resampling = resampling_inner, 
    measure = measure,
    terminator = terminator,
    search_space = search_space,
    store_models = TRUE
  )

  design = generate_design_random(instance$search_space, n = 4L * length(instance$search_space$params))$data
  for (i in seq_len(nrow(design))) {
    instance$eval_batch(design[i, ])
  }

  surrogate = SurrogateLearner$new(lrn("regr.ranger", num.trees = 1000L, keep.inbag = TRUE))
  acq_function = AcqFunctionEI$new()
  acq_optimizer = AcqOptimizer$new(opt("random_search", batch_size = 10000L), terminator = trm("evals", n_evals = 10000L))
  tuner = tnr("mbo", surrogate = surrogate, acq_function = acq_function, acq_optimizer = acq_optimizer)
  tuner$optimize(instance)

  tuning_data = copy(instance$archive$data)
  learner_on_train = instance$objective$learner$clone(deep = TRUE)
  orig_pvs = instance$objective$learner$param_set$values

  best = instance$archive$best()
  learner_on_train$param_set$values = insert_named(orig_pvs, transform_xdt_to_xss(best, search_space = instance$search_space)[[1L]])

  learner_on_train$train(task_train)
  auc = learner_on_train$predict(task_test)$score(measure)

  model = learner_on_train$model$classif.xgboost$model
  features = model$feature_names
  stopifnot(all(features == sort(features)))  # if internal xgboost feature representation does not match the alphabetically ordered one something is really messed up
  n_selected_total = length(task_train$feature_names)  # all
  tmp = tryCatch(eagga:::xgb_model_dt_tree(features, model = model), error = function(ec) {
    NULL
  })
  used = if (is.null(tmp)) {
    features
  } else {
    sort(unique(tmp$Feature[tmp$Feature != "Leaf"]))  # alphabetical order
  }
  n_selected = length(used)
  n_selected = n_selected / n_selected_total  # normalize

  n_interactions_total = (n_selected_total * (n_selected_total - 1L)) / 2L
  pairs = tryCatch(eagga:::interactions(model, option = "pairs"), error = function(ec) {
    NULL
  })
  if (is.null(pairs)) {
    n_interactions = n_interactions_total
  } else {
    tmp = eagga:::get_actual_interactions(used, pairs)
    n_interactions = tmp$n_interactions
  }
  n_interactions = n_interactions / n_interactions_total
  if (n_interactions_total == 0) {
    n_interactions = 0L
  }

  n_non_monotone = n_selected

  data.table(tuning_data = list(tuning_data), best = list(best), auc_test = auc, selected_features_proxy = n_selected, selected_interactions_proxy = n_interactions, selected_non_monotone_proxy = n_non_monotone)
}

nested_resampling_ebm = function(task_train, task_test, resampling_inner, ..., secs = 12L * 3600L) {
  n_features = length(task_train$col_roles$feature)

  learner = lrn("classif.ebm")
  learner$predict_type = "prob"

  measure = msr("classif.auc")

  terminator = trm("run_time", secs = secs)

  search_space = get_ebm_search_space(n_features)

  instance = TuningInstanceSingleCrit$new(
    task = task_train,
    learner = learner,
    resampling = resampling_inner, 
    measure = measure,
    terminator = terminator,
    search_space = search_space,
    store_models = TRUE
  )

  # we generate the design as in mbo but also include EBM's defaults
  design = generate_design_random(instance$search_space, n = 4L * length(instance$search_space$params))$data
  design[1L, ] = as.data.table(instance$search_space$default)
  for (i in seq_len(nrow(design))) {
    instance$eval_batch(design[i, ])
  }

  surrogate = SurrogateLearner$new(lrn("regr.ranger", num.trees = 1000L, keep.inbag = TRUE))
  acq_function = AcqFunctionEI$new()
  acq_optimizer = AcqOptimizer$new(opt("random_search", batch_size = 10000L), terminator = trm("evals", n_evals = 10000L))
  tuner = tnr("mbo", surrogate = surrogate, acq_function = acq_function, acq_optimizer = acq_optimizer)
  tuner$optimize(instance)

  tuning_data = copy(instance$archive$data)
  learner_on_train = instance$objective$learner$clone(deep = TRUE)
  orig_pvs = instance$objective$learner$param_set$values

  best = instance$archive$best()
  learner_on_train$param_set$values = insert_named(orig_pvs, transform_xdt_to_xss(best, search_space = instance$search_space)[[1L]])

  learner_on_train$train(task_train)
  auc = learner_on_train$predict(task_test)$score(measure)

  data.table(tuning_data = list(tuning_data), best = list(best), auc_test = auc, selected_features_proxy = 1, selected_interactions_proxy = min(c(best$interactions, n_features * (n_features - 1L) / 2L)) / (n_features * (n_features - 1L) / 2L), selected_non_monotone_proxy = 1)
}

nested_resampling_glmnet = function(task_train, task_test, resampling_inner, ..., secs = 12L * 3600L) {
  learner = lrn("classif.glmnet")
  learner$predict_type = "prob"

  measure = msr("classif.auc")

  terminator = trm("run_time", secs = secs)

  search_space = get_glmnet_search_space()

  instance = TuningInstanceSingleCrit$new(
    task = task_train,
    learner = learner,
    resampling = resampling_inner, 
    measure = measure,
    terminator = terminator,
    search_space = search_space,
    store_models = TRUE
  )

  design = generate_design_random(instance$search_space, n = 4L * length(instance$search_space$params))$data
  for (i in seq_len(nrow(design))) {
    instance$eval_batch(design[i, ])
  }

  surrogate = SurrogateLearner$new(lrn("regr.ranger", num.trees = 1000L, keep.inbag = TRUE))
  acq_function = AcqFunctionEI$new()
  acq_optimizer = AcqOptimizer$new(opt("random_search", batch_size = 10000L), terminator = trm("evals", n_evals = 10000L))
  tuner = tnr("mbo", surrogate = surrogate, acq_function = acq_function, acq_optimizer = acq_optimizer)
  tuner$optimize(instance)

  tuning_data = copy(instance$archive$data)
  learner_on_train = instance$objective$learner$clone(deep = TRUE)
  orig_pvs = instance$objective$learner$param_set$values

  best = instance$archive$best()
  xss = transform_xdt_to_xss(best, search_space = instance$search_space)[[1L]]
  learner_on_train$param_set$values = insert_named(orig_pvs, xss)

  learner_on_train$train(task_train)
  xss$lambda = mlr3learners:::glmnet_get_lambda(learner_on_train, pv = xss)
  learner_on_train$param_set$values = insert_named(orig_pvs, xss)
  learner_on_train$train(task_train)
  auc = learner_on_train$predict(task_test)$score(measure)

  data.table(tuning_data = list(tuning_data), best = list(best), auc_test = auc, selected_features_proxy = get_n_selected_glmnet(learner_on_train, task = task_train), selected_interactions_proxy = 0L, selected_non_monotone_proxy = 0L)
}

random_forest = function(task_train, task_test, ...) {
  learner = as_learner(po("sortfeatures") %>>% lrn("classif.xgboost"))
  learner$predict_type = "prob"
  xss = list(classif.xgboost.booster = "gbtree", classif.xgboost.tree_method = "exact", classif.xgboost.subsample = 1 - exp(-1), classif.xgboost.colsample_bynode = 1 - exp(-1), classif.xgboost.num_parallel_tree = 1000L, classif.xgboost.nrounds = 1L, classif.xgboost.eta = 1)
  learner$param_set$values = insert_named(learner$param_set$values, xss)

  measure = msr("classif.auc")

  learner$train(task_train)
  auc = learner$predict(task_test)$score(measure)

  model = learner$model$classif.xgboost$model
  features = model$feature_names
  stopifnot(all(features == sort(features)))  # if internal xgboost feature representation does not match the alphabetically ordered one something is really messed up
  n_selected_total = length(task_train$feature_names)  # all
  tmp = tryCatch(eagga:::xgb_model_dt_tree(features, model = model), error = function(ec) {
    NULL
  })
  used = if (is.null(tmp)) {
    features
  } else {
    sort(unique(tmp$Feature[tmp$Feature != "Leaf"]))  # alphabetical order
  }
  n_selected = length(used)
  n_selected = n_selected / n_selected_total  # normalize

  n_interactions_total = (n_selected_total * (n_selected_total - 1L)) / 2L
  pairs = tryCatch(eagga:::interactions(model, option = "pairs"), error = function(ec) {
    NULL
  })
  if (is.null(pairs)) {
    n_interactions = n_interactions_total
  } else {
    tmp = eagga:::get_actual_interactions(used, pairs)
    n_interactions = tmp$n_interactions
  }
  n_interactions = n_interactions / n_interactions_total
  if (n_interactions_total == 0) {
    n_interactions = 0L
  }

  n_non_monotone = n_selected

  data.table(tuning_data = NULL, best = NULL, auc_test = auc, selected_features_proxy = n_selected, selected_interactions_proxy = n_interactions, selected_non_monotone_proxy = n_non_monotone)
}

get_n_selected_glmnet = function(learner, task, normalize = TRUE) {
  features = task$feature_names
  n_selected_total = length(features)
  n_selected = learner$model$df
  stopifnot(length(n_selected) == 1L)
  if (normalize) {
    n_selected = n_selected / n_selected_total
  }
  n_selected
}

nested_resampling_eagga_ablation = function(task_train, task_test, resampling_inner, crossover, mutation, random, detectors, task_id_, repl_, secs = 12L * 3600L) {
  reference_point = t(t(c(minus_classif.auc = 0, selected_features = 1, selected_interactions = 1, selected_non_monotone = 1)))
  mu = 100L
  lambda = 10L

  learner = as_learner(po("colapply") %>>% po("select") %>>% po("sortfeatures") %>>% lrn("classif.xgboost"))
  learner$predict_type = "prob"
  learner$param_set$values$classif.xgboost.booster = "gbtree"
  learner$param_set$values$classif.xgboost.tree_method = "exact"
  learner$param_set$values$colapply.applicator = function(x) - x

  measures = list(msr("classif.auc"),
                  msr("selected_features_proxy"),
                  msr("selected_interactions_proxy"),
                  msr("selected_non_monotone_proxy"))

  terminator = trm("run_time", secs = secs)

  search_space = get_xgboost_search_space_eagga()

  instance = TuningInstanceMultiCrit$new(
    task = task_train,
    learner = learner,
    resampling = resampling_inner, 
    measures = measures,
    terminator = terminator,
    search_space = search_space,
    store_models = TRUE
  )

  tuner = tnr("eagga_ablation", mu = mu, lambda = lambda, crossover = crossover, mutation = mutation, random = random, detectors = detectors)
  tuner$optimize(instance)

  tuning_data = copy(instance$archive$data)
  tuning_pareto = copy(instance$archive$best())

  pareto = copy(tuning_pareto)
  learner_on_test = instance$objective$learner$clone(deep = TRUE)
  orig_pvs = instance$objective$learner$param_set$values
  for (p in seq_len(NROW(pareto))) {
    groupstructure = pareto[p, ][["groupstructure"]][[1L]]
    # during a random search it can happen that zero selected (after updating) is still Pareto optimal
    if (groupstructure$n_selected > 0L) {
      xdt = copy(pareto[p, ])
      xdt[["groupstructure"]][[1L]] = groupstructure
      xdt[[tuner$param_set$values$select_id]][[1L]] = groupstructure$create_selector()
      xdt[[tuner$param_set$values$interaction_id]][[1L]] = groupstructure$create_interaction_constraints()
      xdt[[tuner$param_set$values$monotone_id]][[1L]] = groupstructure$create_monotonicity_constraints()
      xss = transform_xdt_to_xss(xdt, search_space = instance$search_space)[[1L]]
      xss = insert_named(orig_pvs, xss)
      learner_on_test$param_set$values = xss
      learner_on_test$train(task_train)
      pareto[p, auc_test := learner_on_test$predict(task_test)$score(msr("classif.auc"))]
    } else {
      pareto[p, auc_test := 0.5]  # zero selected, i.e., featureless will be AUC of 0.5
    }
    # proxy measures must not be updated because they were already determined on the task_train during tuning
  }

  root = here::here()
  majority_vote = readRDS(file.path(root, "attic", "benchmarks", "results", "eagga_majority_vote.rds"))[task_id == task_id_ & repl == repl_]

  # validation dominated hypervolume
  y_val = tuning_pareto[, c("classif.auc", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]
  y_val = rbind(y_val, majority_vote[, c("classif.auc", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")])
  dhv_val = emoa::dominated_hypervolume(t(y_val) * eagga:::mult_max_to_min(instance$objective$codomain), ref = reference_point)

  # test dominated hypervolume
  y_test = pareto[, c("auc_test", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]
  y_test = rbind(y_test, majority_vote[, c("auc_test", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")])
  colnames(y_test) = c("classif.auc", "selected_features", "selected_interactions", "selected_non_monotone")
  dhv_test = emoa::dominated_hypervolume(t(y_test) * eagga:::mult_max_to_min(instance$objective$codomain), ref = reference_point)

  # validation dominated hypervolume anytime
  dhv_val_anytime = map_dtr(seq_len(NROW(tuning_data)), function(p) {
    start = tuning_data[1L, ]$timestamp
    stop = tuning_data[p, ]$timestamp
    non_dominated = !is_dominated(t(tuning_data[seq_len(p), c("classif.auc", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]) * eagga:::mult_max_to_min(instance$objective$codomain))
    y_val = tuning_data[which(non_dominated), c("classif.auc", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]
    y_val = rbind(y_val, majority_vote[, c("classif.auc", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")])
    dhv_val = emoa::dominated_hypervolume(t(y_val) * eagga:::mult_max_to_min(instance$objective$codomain), ref = reference_point)
    data.table(iteration = p, dhv_val = dhv_val, runtime = as.numeric(stop - start, units = "secs"))
  })

  data.table(tuning_data = list(tuning_data), tuning_pareto = list(tuning_pareto), pareto = list(pareto), dhv_val = dhv_val, dhv_test = dhv_test, dhv_val_anytime = list(dhv_val_anytime), best_val = max(tuning_pareto$classif.auc), best_test = max(pareto$auc_test))
}

nested_resampling_xgboost_mo = function(task_train, task_test, resampling_inner, task_id_, repl_, secs = 12L * 3600L) {
  reference_point = t(t(c(minus_classif.auc = 0, selected_features = 1, selected_interactions = 1, selected_non_monotone = 1)))

  learner = as_learner(po("sortfeatures") %>>% lrn("classif.xgboost"))
  learner$predict_type = "prob"
  learner$param_set$values$classif.xgboost.booster = "gbtree"
  learner$param_set$values$classif.xgboost.tree_method = "exact"

  terminator = trm("run_time", secs = secs)

  search_space = get_xgboost_search_space()

  surrogate = SurrogateLearner$new(lrn("regr.ranger", num.trees = 1000L, keep.inbag = TRUE))
  acq_function = AcqFunctionEI$new()
  acq_optimizer = AcqOptimizer$new(opt("random_search", batch_size = 10000L), terminator = trm("evals", n_evals = 10000L))
  optimizer = opt("mbo", loop_function = bayesopt_parego, surrogate = surrogate, acq_function = acq_function, acq_optimizer = acq_optimizer)

  learner_on_train = learner$clone(deep = TRUE)
  orig_pvs = learner$param_set$values
  objective = ObjectiveRFunDt$new(
    fun = function(xdt) {
      map_dtr(seq_len(nrow(xdt)), function(i) {
        learner_on_train$param_set$values = insert_named(orig_pvs, as.list(xdt[i, ]))
        rr = resample(task = task_train, learner = learner_on_train, resampling = resampling_inner, store_models = TRUE)
        auc = rr$aggregate(msr("classif.auc"))
        learner_on_train$train(task_train)
        model = learner_on_train$model$classif.xgboost$model
        features = model$feature_names  # pre-selected based on selector
         stopifnot(all(features == sort(features)))  # if internal xgboost feature representation does not match the alphabetically ordered one something is really messed up
        n_selected_total = length(task_train$feature_names)  # all
        tmp = tryCatch(eagga:::xgb_model_dt_tree(features, model = model), error = function(ec) {
          NULL
        })
        used = if (is.null(tmp)) {
          features
        } else {
          sort(unique(tmp$Feature[tmp$Feature != "Leaf"]))  # alphabetical order
        }
        n_selected = length(used)
        n_selected = n_selected / n_selected_total  # normalize

        n_interactions_total = (n_selected_total * (n_selected_total - 1L)) / 2L
        pairs = tryCatch(eagga:::interactions(model, option = "pairs"), error = function(ec) {
          NULL
        })
        if (is.null(pairs)) {
          n_interactions = n_interactions_total
        } else {
          tmp = eagga:::get_actual_interactions(used, pairs)
          n_interactions = tmp$n_interactions
        }
        n_interactions = n_interactions / n_interactions_total
        if (n_interactions_total == 0) {
          n_interactions = 0L
        }

        n_non_monotone = n_selected

        data.table(classif.auc = auc, selected_features_proxy = n_selected, selected_interactions_proxy = n_interactions, selected_non_monotone_proxy = n_non_monotone)
      })
    },
    domain = get_xgboost_domain(),
    codomain = ps(classif.auc = p_dbl(lower = 0, upper = 1, tags = "maximize"),
                  selected_features_proxy = p_dbl(lower = 0, upper = 1, tags = "minimize"),
                  selected_interactions_proxy = p_dbl(lower = 0, upper = 1, tags = "minimize"),
                  selected_non_monotone_proxy = p_dbl(lower = 0, upper = 1, tags = "minimize"))
  )

  instance = OptimInstanceMultiCrit$new(
    objective = objective,
    terminator = terminator,
    search_space = search_space
  )

  design = generate_design_random(instance$search_space, n = 4L * length(instance$search_space$params))$data
  for (i in seq_len(nrow(design))) {
    instance$eval_batch(design[i, ])
  }

  optimizer$optimize(instance)

  tuning_data = copy(instance$archive$data)
  tuning_pareto = copy(instance$archive$best())

  pareto = copy(tuning_pareto)
  learner_on_test = learner$clone(deep = TRUE)
  for (p in seq_len(NROW(pareto))) {
    xdt = copy(pareto[p, ])
    xss = transform_xdt_to_xss(xdt, search_space = instance$search_space)[[1L]]
    xss = insert_named(orig_pvs, xss)
    learner_on_test$param_set$values = xss
    learner_on_test$train(task_train)
    pareto[p, auc_test := learner_on_test$predict(task_test)$score(msr("classif.auc"))]
    # proxy measures must not be updated because they were already determined on the task_train during tuning
  }

  root = here::here()
  majority_vote = readRDS(file.path(root, "attic", "benchmarks", "results", "eagga_majority_vote.rds"))[task_id == task_id_ & repl == repl_]

  # validation dominated hypervolume
  y_val = tuning_pareto[, c("classif.auc", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]
  y_val = rbind(y_val, majority_vote[, c("classif.auc", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")])
  dhv_val = emoa::dominated_hypervolume(t(y_val) * eagga:::mult_max_to_min(instance$objective$codomain), ref = reference_point)

  # test dominated hypervolume
  y_test = pareto[, c("auc_test", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]
  y_test = rbind(y_test, majority_vote[, c("auc_test", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")])
  colnames(y_test) = c("classif.auc", "selected_features", "selected_interactions", "selected_non_monotone")
  dhv_test = emoa::dominated_hypervolume(t(y_test) * eagga:::mult_max_to_min(instance$objective$codomain), ref = reference_point)

  # validation dominated hypervolume anytime
  dhv_val_anytime = map_dtr(seq_len(NROW(tuning_data)), function(p) {
    start = tuning_data[1L, ]$timestamp
    stop = tuning_data[p, ]$timestamp
    non_dominated = !is_dominated(t(tuning_data[seq_len(p), c("classif.auc", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]) * eagga:::mult_max_to_min(instance$objective$codomain))
    y_val = tuning_data[which(non_dominated), c("classif.auc", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]
    y_val = rbind(y_val, majority_vote[, c("classif.auc", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")])
    dhv_val = emoa::dominated_hypervolume(t(y_val) * eagga:::mult_max_to_min(instance$objective$codomain), ref = reference_point)
    data.table(iteration = p, dhv_val = dhv_val, runtime = as.numeric(stop - start, units = "secs"))
  })

  data.table(tuning_data = list(tuning_data), tuning_pareto = list(tuning_pareto), pareto = list(pareto), dhv_val = dhv_val, dhv_test = dhv_test, dhv_val_anytime = list(dhv_val_anytime), best_val = max(tuning_pareto$classif.auc), best_test = max(pareto$auc_test))
}

nested_resampling_ebm_fallback = function(task_train, task_test, resampling_inner, ..., secs = 12L * 3600L) {
  n_features = length(task_train$col_roles$feature)

  learner = lrn("classif.ebm")
  learner$predict_type = "prob"

  measure = msr("classif.auc")

  terminator = trm("run_time", secs = secs)

  search_space = get_ebm_search_space(n_features)

  instance = TuningInstanceSingleCrit$new(
    task = task_train,
    learner = learner,
    resampling = resampling_inner,
    measure = measure,
    terminator = terminator,
    search_space = search_space,
    store_models = TRUE
  )

  # we only evaluate EBM's default as a fallback
  design = as.data.table(instance$search_space$default)
  instance$eval_batch(design)

  tuning_data = copy(instance$archive$data)
  learner_on_train = instance$objective$learner$clone(deep = TRUE)
  orig_pvs = instance$objective$learner$param_set$values

  best = instance$archive$best()
  learner_on_train$param_set$values = insert_named(orig_pvs, transform_xdt_to_xss(best, search_space = instance$search_space)[[1L]])

  learner_on_train$train(task_train)
  auc = learner_on_train$predict(task_test)$score(measure)

  data.table(tuning_data = list(tuning_data), best = list(best), auc_test = auc, selected_features_proxy = 1, selected_interactions_proxy = min(c(best$interactions, n_features * (n_features - 1L) / 2L)) / (n_features * (n_features - 1L) / 2L), selected_non_monotone_proxy = 1)
}

