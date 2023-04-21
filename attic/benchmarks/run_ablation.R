library(data.table)
setDTthreads(1L)
library(mlr3)
library(mlr3learners)
library(mlr3pipelines)
library(mlr3oml)
library(mlr3misc)
library(mlr3tuning)
library(mlr3mbo)
library(paradox)
library(bbotk)
library(eagga)

root = here::here()
source_files = file.path(root, "attic", "benchmarks", c("helpers.R", "search_spaces.R"))

RhpcBLASctl::blas_set_num_threads(1L)
RhpcBLASctl::omp_set_num_threads(1L)

eval_ = function(job, data, instance, ...) {
  library(data.table)
  library(mlr3)
  library(mlr3learners)
  library(mlr3pipelines)
  library(mlr3misc)
  library(mlr3tuning)
  library(mlr3mbo)
  library(paradox)
  library(bbotk)
  library(eagga)

  RhpcBLASctl::blas_set_num_threads(1L)
  RhpcBLASctl::omp_set_num_threads(1L)

  logger = lgr::get_logger("mlr3")
  logger$set_threshold("warn")
  logger = lgr::get_logger("bbotk")
  logger$set_threshold("warn")
  future::plan("sequential")

  task = instance$task
  task_id = instance$id

  repl = job$repl
  set.seed(repl)  # same outer and inner resampling for all methods given a repl on a task
  resampling_outer = rsmp("holdout", ratio = 2/3)$instantiate(task)
  train_set = resampling_outer$train_set(1L)
  test_set = resampling_outer$test_set(1L)
  task_train = task$clone(deep = TRUE)$filter(rows = train_set)
  task_test = task$clone(deep = TRUE)$filter(rows = test_set)
  resampling_inner = rsmp("cv", folds = 5L)$instantiate(task_train)  # changed
  secs = 8L * 3600L
 
  method = job$algo.pars$method
  # EAGGA: (crossover FALSE, mutation FALSE, both FALSE, both TRUE) x use_detectors (TRUE / FALSE); both TRUE + use_detectors TRUE not needed
  #        random (TRUE, FALSE) x use_detectors (TRUE, FALSE); random FALSE + use_detectors TRUE not needed
  crossover = isTRUE(job$algo.pars$crossover)
  mutation = isTRUE(job$algo.pars$mutation)
  random = isTRUE(job$algo.pars$random)
  detectors = isTRUE(job$algo.pars$detectors)

  set.seed(job$seed) 

  results = if (method == "eagga_ablation") {
    nested_resampling_eagga_ablation(task_train, task_test = task_test, resampling_inner = resampling_inner, crossover = crossover, mutation = mutation, random = random, detectors = detectors, task_id_ = task_id, repl_ = repl, secs = secs)
  } else if (method == "xgboost_mo") {
    nested_resampling_xgboost_mo(task_train, task_test = task_test, resampling_inner = resampling_inner, task_id_ = task_id, repl_ = repl, secs = secs)
  }
  results
}

library(batchtools)
reg = makeExperimentRegistry(file.dir = "/gscratch/lschnei8/registry_eagga_ablation", source = source_files)
#reg = makeExperimentRegistry(file.dir = NA)
saveRegistry(reg)

ids = c(37, 43, 3903, 3913, 3904, 3918, 10093, 9946, 146819, 359955, 189922, 359962, 190392, 167120, 190137, 190410, 168350, 359975, 359972, 146820)
tasks = map(ids, function(id) {
  task = tsk("oml", task_id = id)
  if (id == 3904) {
    tmp = task$data()
    tmp = na.omit(tmp)
    task = TaskClassif$new(id = task$id, backend = tmp, target = task$target_names)
  }
  task
})
#checks = map_lgl(tasks, function(task) {
#  all(c("factor", "ordered", "logical", "POSIXct", "character") %nin% unique(task$feature_types)) && sum(task$missings()) == 0L && sum(apply(task$data(cols = task$feature_names), 2, function(x) length(unique(x)) <= 2)) == 0L
#})

instances = data.table(id = ids, task = tasks)
instances[, id_plan := 1:.N]

# add problems
prob_designs = imap(split(instances, instances$id_plan), function(instance, name) {
  addProblem(as.character(instance$id_plan), fun = function(...) list(...), seed = instance$id)
  set_names(list(instance), as.character(instance$id_plan))
})
nn = sapply(prob_designs, names)
prob_designs = unlist(prob_designs, recursive = FALSE, use.names = FALSE)
names(prob_designs) = nn

# add eval_ algorithm (never use `eval` as a function name or have a function named `eval` in .GlobalEnv)
addAlgorithm("eval_", fun = eval_)
ablation1 = as.data.table(expand.grid(method = "eagga_ablation", crossover = c(TRUE, FALSE), mutation = c(TRUE, FALSE), detectors = c(TRUE, FALSE)))[-1L, ]  # TRUE, TRUE, TRUE is default
ablation2 = as.data.table(expand.grid(method = "eagga_ablation", random = c(TRUE, FALSE), detectors = c(TRUE, FALSE)))[-2L, ]  # FALSE, TRUE is default
ablation3 = data.table(method = "xgboost_mo")
# NAs get default
ablation = rbind(ablation1, ablation2, ablation3, fill = TRUE)

for (i in seq_len(nrow(ablation))) {
  ids = addExperiments(
      prob.designs = prob_designs,
      algo.designs = list(eval_ = ablation[i, ]),
      repls = 10L
  )
  addJobTags(ids, as.character(ablation[i, ]$method))
}

# standard resources used to submit jobs to cluster
resources.serial.default = list(max.concurrent.jobs = 9999L, ncpus = 1L)

jobs = getJobTable()
jobs[, memory := 1024L * 48L]
jobs[(problem == 11 | problem == 13 | problem == 14 | problem == 16), memory := 1024L * 64L]
jobs[, walltime := 16L * 3600L]

submitJobs(jobs[tags == "xgboost_mo", ], resources = resources.serial.default)
random = map_lgl(seq_len(nrow(jobs)), function(i) isTRUE(jobs[i, ]$algo.pars[[1L]]$detectors == TRUE) & isTRUE(jobs[i, ]$algo.pars[[1L]]$random == TRUE))
no_detectors = map_lgl(seq_len(nrow(jobs)), function(i) isTRUE(jobs[i, ]$algo.pars[[1L]]$crossover == TRUE) & isTRUE(jobs[i, ]$algo.pars[[1L]]$mutation == TRUE) & isTRUE(jobs[i, ]$algo.pars[[1L]]$detectors == FALSE))
no_crossover = map_lgl(seq_len(nrow(jobs)), function(i) isTRUE(jobs[i, ]$algo.pars[[1L]]$crossover == FALSE) & isTRUE(jobs[i, ]$algo.pars[[1L]]$mutation == TRUE) & isTRUE(jobs[i, ]$algo.pars[[1L]]$detectors == TRUE))
no_mutation = map_lgl(seq_len(nrow(jobs)), function(i) isTRUE(jobs[i, ]$algo.pars[[1L]]$crossover == TRUE) & isTRUE(jobs[i, ]$algo.pars[[1L]]$mutation == FALSE) & isTRUE(jobs[i, ]$algo.pars[[1L]]$detectors == TRUE))
no_crossover_mutation = map_lgl(seq_len(nrow(jobs)), function(i) isTRUE(jobs[i, ]$algo.pars[[1L]]$crossover == FALSE) & isTRUE(jobs[i, ]$algo.pars[[1L]]$mutation == FALSE) & isTRUE(jobs[i, ]$algo.pars[[1L]]$detectors == TRUE))
submitJobs(jobs[random, ], resources = resources.serial.default)
submitJobs(jobs[no_detectors, ], resources = resources.serial.default)
submitJobs(jobs[no_crossover, ], resources = resources.serial.default)
submitJobs(jobs[no_mutation, ], resources = resources.serial.default)
submitJobs(jobs[no_crossover_mutation, ], resources = resources.serial.default)

# for the task 9, 14, 15, xgboost_mo initial design took longer than the already generous 16h at least one time
# therefore we increased to 24 hours to see if it helps
# job.id 2132 got 36 hours ...

#######################################################################################################################################################################################################

tab = getJobTable()
tab = tab[job.id %in% findDone()$job.id & tags == "xgboost_mo"]
sum(tab$time.running, na.rm = TRUE) / 3600L  # roughly 1800 CPU hours
results = reduceResultsDataTable(tab$job.id, fun = function(x, job) {
  data = x
  data[, tuning_data := NULL]
  data[, task_id := job$prob.pars$id]
  data[, method := job$algo.pars$method]
  data[, repl := job$repl]
  data
})
results = rbindlist(results$result, fill = TRUE)
saveRDS(results, "/gscratch/lschnei8/eagga_ablation_xgboost_mo.rds")

tab = getJobTable()
tab = tab[job.id %in% findDone()$job.id & tags == "eagga_ablation"]
sum(tab$time.running, na.rm = TRUE) / 3600L  # roughly 8050 CPU hours
results = reduceResultsDataTable(tab$job.id, fun = function(x, job) {
  data = x
  data[, tuning_data := NULL]
  data[, task_id := job$prob.pars$id]
  data[, method := job$algo.pars$method]
  data[, crossover := job$algo.pars$crossover]
  data[, mutation := job$algo.pars$mutation]
  data[, detectors := job$algo.pars$detectors]
  data[, random := job$algo.pars$random]
  data[, repl := job$repl]
  data
})
results = rbindlist(results$result, fill = TRUE)
saveRDS(results, "/gscratch/lschnei8/eagga_ablation_eagga.rds")

tab = getJobTable()
tab = tab[job.id %in% findDone()$job.id]
info = reduceResultsDataTable(tab$job.id, fun = function(x, job) {
  n = nrow(x$tuning_data[[1L]])
  startt = x$tuning_data[[1L]]$timestamp[1L]
  stopt = x$tuning_data[[1L]]$timestamp[n]
  elapsed = as.numeric(stopt - startt)
  data = data.table(n = n, elapsed = elapsed)
  data[, task_id := job$prob.pars$id]
  data[, method := job$algo.pars$method]
  data[, repl := job$repl]
  data
})
info = rbindlist(info$result, fill = TRUE)
saveRDS(info, "/gscratch/lschnei8/eagga_ablation_info.rds")

