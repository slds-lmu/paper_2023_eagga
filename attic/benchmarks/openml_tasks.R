# all tasks from AutoML classification suite, binary classification, numeric or integer features, < 100.000 obs, < 1000 features

library(mlr3)
library(mlr3misc)
library(mlr3oml)
library(data.table)

info = map_dtr(c(37, 43, 3903, 3904, 3913, 3918, 10093, 9946, 146819, 359955, 189922, 359962, 190392, 167120, 190137, 190410, 168350, 359975, 359972, 146820), function(id) {
  task = tsk("oml", task_id = id)
  x = strsplit(task$id, " ")[[1L]]
  data.table(id = id, name = x, n = task$nrow, p = task$ncol - 1)
})

setorderv(info, col = "id")

#        id                             name     n   p
# 1:     37                         diabetes   768   8
# 2:     43                         spambase  4601  57
# 3:   3903                              pc3  1563  37
# 4:   3904                              jm1 10885  21  # n - 5!
# 5:   3913                              kc2   522  21
# 6:   3918                              pc1  1109  21
# 7:   9946                             wdbc   569  30
# 8:  10093          banknote-authentication  1372   4
# 9: 146819 climate-model-simulation-crashes   540  20
#10: 146820                             wilt  4839   5
#11: 167120                      numerai28.6 96320  21
#12: 168350                          phoneme  5404   5
#13: 189922                             gina  3153 970
#14: 190137                  ozone-level-8hr  2534  72
#15: 190392                         madeline  3140 259
#16: 190410                       philippine  5832 308
#17: 359955 blood-transfusion-service-center   748   4
#18: 359962                              kc1  2109  21
#19: 359972                          sylvine  5124  20
#20: 359975                        Satellite  5100  36

# 3904 has five missings we excluded
task = tsk("oml", task_id = 3904)
tmp = task$data()
tmp = na.omit(tmp)
task = TaskClassif$new(id = task$id, backend = tmp, target = task$target_names)
task$nrow
task$ncol - 1L

