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
  set.seed(job$seed) 

  results = if (method == "eagga") {
    nested_resampling_eagga(task_train, task_test = task_test, resampling_inner = resampling_inner, task_id_ = task_id, repl_ = repl, secs = secs)
  } else if (method == "eagga_md2") {
    nested_resampling_eagga_md2(task_train, task_test = task_test, resampling_inner = resampling_inner, task_id_ = task_id, repl_ = repl, secs = secs)
  } else if (method == "xgboost") {
    nested_resampling_xgboost(task_train, task_test = task_test, resampling_inner = resampling_inner, secs = secs)
  } else if (method == "ebm") {
    reticulate::use_condaenv("EBmlr3", required = TRUE)
    library(EBmlr3)
    #nested_resampling_ebm(task_train, task_test = task_test, resampling_inner = resampling_inner, secs = secs)
    nested_resampling_ebm_fallback(task_train, task_test = task_test, resampling_inner = resampling_inner, secs = secs)
  } else if (method == "glmnet") {
    nested_resampling_glmnet(task_train, task_test = task_test, resampling_inner = resampling_inner, secs = secs)
  } else if (method == "rf") {
    random_forest(task_train, task_test = task_test)
  }
  results
}

library(batchtools)
reg = makeExperimentRegistry(file.dir = "/gscratch/lschnei8/registry_eagga_ours_so", source = source_files)
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

for (method in c("eagga", "eagga_md2", "xgboost", "ebm", "glmnet", "rf")) {
  ids = addExperiments(
      prob.designs = prob_designs,
      algo.designs = list(eval_ = data.table(method = method)),
      repls = 10L
  )
  addJobTags(ids, method)
}

# standard resources used to submit jobs to cluster
resources.serial.default = list(max.concurrent.jobs = 9999L, ncpus = 1L)

jobs = getJobTable()
jobs[, memory := 1024L * 48L]
jobs[(problem == 11 | problem == 13 | problem == 14 | problem == 16), memory := 1024L * 64L]
jobs[, walltime := 16L * 3600L]
jobs[tags == "rf", memory := 1024L * 16L]
jobs[tags == "rf", walltime := 3600L]

submitJobs(jobs, resources = resources.serial.default)

# for the following tasks, ebm initial design took longer than the already generous 16h at least one time: 2, 5, 7, 8, 11, 13, 14, 15, 16, 17, 19, 20
# for 15, 5 and 8 we increase the walltime to 24 hours to see if it helps
# for the other ones we fallback to only evaluating the default

# --> all 24 hours
#    job.id
# 1:    642
# 2:    671
# 3:    673
# 4:    675
# 5:    676
# 6:    678
# 7:    679
# 8:    680
# 9:    742
#10:    743
#11:    748

# --> 24 hours
#    job.id
# 1:    442  --> moran-hugemem 256
# 2:    445

# EBM fallback:
# [1] 442 445 611 612 613 614 615 616 617 618 619 620 661 662 663 664 665 666 667
#[20] 668 669 670 701 702 703 704 705 706 707 708 709 710 721 722 723 724 725 726
#[39] 727 728 729 730 731 732 733 734 735 736 737 738 739 740 751 752 753 754 755
#[58] 756 757 758 759 760 761 762 763 764 765 766 768 769 770 781 782 783 784 785
#[77] 786 787 788 789 790 791 792 793 794 795 797 798 799 800

#######################################################################################################################################################################################################

tab = getJobTable()
tab = tab[job.id %in% findDone()$job.id]
sum(tab$time.running, na.rm = TRUE) / 3600L  # roughly 7850 CPU hours
results = reduceResultsDataTable(tab$job.id, fun = function(x, job) {
  data = x
  data[, tuning_data := NULL]
  data[, task_id := job$prob.pars$id]
  data[, method := job$algo.pars$method]
  data[, repl := job$repl]
  data
})
results = rbindlist(results$result, fill = TRUE)
saveRDS(results, "/gscratch/lschnei8/eagga_ours_so.rds")

info = reduceResultsDataTable(tab$job.id, fun = function(x, job) {
  if (job$algo.pars$method != "rf") {
    n = nrow(x$tuning_data[[1L]])
    startt = x$tuning_data[[1L]]$timestamp[1L]
    stopt = x$tuning_data[[1L]]$timestamp[n]
    elapsed = as.numeric(stopt - startt)
  } else {
    n = 1
    elapsed = 0
  }
  data = data.table(n = n, elapsed = elapsed)
  data[, task_id := job$prob.pars$id]
  data[, method := job$algo.pars$method]
  data[, repl := job$repl]
  data
})
info = rbindlist(info$result, fill = TRUE)
saveRDS(info, "/gscratch/lschnei8/eagga_ours_so_info.rds")

