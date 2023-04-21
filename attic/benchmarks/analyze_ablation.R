library(data.table)
library(mlr3misc)
library(bbotk)
library(ggplot2)
library(scales)
library(emoa)
library(pammtools)
library(scmamp)
library(xtable)

source("openml_tasks.R")
setorderv(info, col = "p")
info$name[1L] = "banknote"
info$name[2L] = "blood"
info$name[6L] = "climate"
info$name[17L] = "ozone"
info[, task := paste0(name, " (", p, ")")]
info = info[, c("id", "task"), with = FALSE]
colnames(info) = c("task_id", "task")

# Section 5.2 and 5.3
eagga = readRDS("results/eagga_ours_so.rds")[method %in% c("eagga")]
eagga_dhv = map_dtr(seq_len(nrow(eagga)), function(i) {
  x = eagga[i, ]
  tmp = x$dhv_val_anytime[[1L]]
  tmp[, task_id := x$task_id]
  tmp[, method := x$method]
  tmp[, repl := x$repl]
  tmp
})

xgboost_mo = readRDS("results/eagga_ablation_xgboost_mo.rds")
xgboost_mo_dhv = map_dtr(seq_len(nrow(xgboost_mo)), function(i) {
  x = xgboost_mo[i, ]
  tmp = x$dhv_val_anytime[[1L]]
  tmp[, task_id := x$task_id]
  tmp[, method := x$method]
  tmp[, repl := x$repl]
  tmp
})

eagga_ablation = readRDS("results/eagga_ablation_eagga.rds")
eagga_ablation_dhv = map_dtr(seq_len(nrow(eagga_ablation)), function(i) {
  x = eagga_ablation[i, ]
  tmp = x$dhv_val_anytime[[1L]]
  tmp[, task_id := x$task_id]
  tmp[, method := x$method]
  tmp[, repl := x$repl]
  tmp[, crossover := x$crossover]
  tmp[, mutation := x$mutation]
  tmp[, detectors := x$detectors]
  tmp[, random := x$random]
  tmp
})
eagga_ablation_dhv[, method := paste0(crossover, "_", mutation, "_", detectors, "_", random)]

anytime_hvs = rbind(eagga_dhv, eagga_ablation_dhv, xgboost_mo_dhv, fill = TRUE)

anytime_hvs_runtime = map_dtr(unique(anytime_hvs$task_id), function(task_id_) {
  seqs = seq(0, 8L * 3600L, length.out = 1001L)
  ref = anytime_hvs[task_id == task_id_]
  min_ = min(ref$dhv_val)
  max_ = max(ref$dhv_val)
  anytime_hvs[task_id == task_id_, dhv_val_normalized := (dhv_val - min_) / (max_ - min_)]
  map_dtr(unique(anytime_hvs$repl), function(repl_) {
    tmp = anytime_hvs[task_id == task_id_ & repl == repl_]
    result = map_dtr(seqs, function(seq_) {
      tmp[runtime <= seq_, .(dhv_val = max(dhv_val), dhv_val_normalized = max(dhv_val_normalized), runtime = seq_), by = .(method)]
    })
    result[, task_id := task_id_]
    result[, repl := repl_]
    result
  })
})

anytime_hvs_runtime = merge(anytime_hvs_runtime, info, by = "task_id")

anytime_hvs_init = map_dtr(unique(anytime_hvs$task_id), function(task_id_) {
  map_dtr(unique(anytime_hvs$repl), function(repl_) {
    tmp_eagga = anytime_hvs[method != "xgboost_mo" & task_id == task_id_ & repl == repl_ & iteration == 101L]  # mu = 100
    tmp_xgboost_mo = anytime_hvs[method == "xgboost_mo" & task_id == task_id_ & repl == repl_ & iteration == 41L]  # xgboost search space d = 10, init 4 * d
    rbind(tmp_eagga, tmp_xgboost_mo)
  })
})

anytime_hvs_init = merge(anytime_hvs_init, info, by = "task_id")

mean_anytime_hvs_runtime = anytime_hvs_runtime[, .(mean_anytime_hv = mean(dhv_val), se_anytime_hv = sd(dhv_val) / sqrt(.N), mean_anytime_hv_normalized = mean(dhv_val_normalized), se_anytime_hv_normalized = sd(dhv_val_normalized) / sqrt(.N)), by = .(runtime, method, task)]
mean_anytime_hvs_runtime$task = factor(mean_anytime_hvs_runtime$task, levels = info$task)
mean_anytime_hvs_runtime[, method := factor(method, labels = c("EAGGA_XGBoost", "No_Crossover", "No_Mutation", "No_Cross_Mut", "No_Detectors", "Random Search", "XGBoost_MO"))]
mean_anytime_hvs_init = anytime_hvs_init[, .(mean_init = mean(runtime), se_init = sd(runtime) / sqrt(.N)), by = .(method, task)]
mean_anytime_hvs_init$task = factor(mean_anytime_hvs_init$task, levels = info$task)
mean_anytime_hvs_init[, method := factor(method, labels = c("EAGGA_XGBoost", "No_Crossover", "No_Mutation", "No_Cross_Mut", "No_Detectors", "Random Search", "XGBoost_MO"))]

# Figure 6 xgboost_mo anytime valid
mean_anytime_hvs_runtime_xgboost = mean_anytime_hvs_runtime[method %in% c("EAGGA_XGBoost", "XGBoost_MO")]
mean_anytime_hvs_init_xgboost = mean_anytime_hvs_init[method %in% c("EAGGA_XGBoost", "XGBoost_MO")]

g = ggplot(aes(x = runtime, y = mean_anytime_hv, colour = method, fill = method), data = mean_anytime_hvs_runtime_xgboost) +
  geom_step() +
  geom_stepribbon(aes(ymin = mean_anytime_hv - se_anytime_hv, ymax = mean_anytime_hv + se_anytime_hv), colour = NA, alpha = 0.1) +
  labs(y = "Mean Dominated Hypervolume", x = "Runtime (s)", colour = "Method", fill = "Method", linetype = "Method") +
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x), labels = trans_format("log10", math_format(10^.x))) +
  facet_wrap(~ task, scales = "free", nrow = 5, ncol = 4) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

ggsave("plots/eagga_ablation_xgboost.pdf", plot = g, device = "pdf", width = 10, height = 8)

# inner optimization xgboost_mo final test
wilcoxonSignedTest(mean_anytime_hvs_runtime_xgboost[method == "EAGGA_XGBoost" & runtime == 8L * 3600L, ]$mean_anytime_hv, mean_anytime_hvs_runtime_xgboost[method == "XGBoost_MO" & runtime == 8L * 3600L, ]$mean_anytime_hv) # T = 30, p = 0.002556

# xgboost_mo final test
all = readRDS("results/eagga_ours_so.rds")
dat = rbind(all, xgboost_mo, fill = TRUE)
majority = readRDS("results/eagga_majority_vote.rds")

ref = t(t(c(minus_classif.auc = 0, selected_features = 1, selected_interactions = 1, selected_non_monotone = 1)))
fct = c(-1, 1, 1, 1)
ys = c("auc_test", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")

hvs = map_dtr(unique(dat$task_id), function(task_id_) {
  map_dtr(unique(dat$repl), function(repl_) {
    tmp = dat[task_id == task_id_ & repl == repl_]
    eagga = tmp[method == "eagga"]$pareto[[1L]][, ..ys]
    eagga = rbind(eagga, majority[task_id == task_id_ & repl == repl_, ..ys])
    eagga_hv = dominated_hypervolume(t(eagga) * fct, ref = ref)
    eagga_md2 = tmp[method == "eagga_md2"]$pareto[[1L]][, ..ys]
    eagga_md2 = rbind(eagga_md2, majority[task_id == task_id_ & repl == repl_, ..ys])
    eagga_md2_hv = dominated_hypervolume(t(eagga_md2) * fct, ref = ref)
    xgboost_mo = tmp[method == "xgboost_mo"]$pareto[[1L]][, ..ys]
    xgboost_mo = rbind(xgboost_mo, majority[task_id == task_id_ & repl == repl_, ..ys])
    xgboost_mo_hv = dominated_hypervolume(t(xgboost_mo) * fct, ref = ref)
    data.table(eagga_hv = eagga_hv, eagga_md2_hv = eagga_md2_hv, xgboost_mo_hv = xgboost_mo_hv, repl = repl_, task_id = task_id_)
  })
})

hvs = merge(hvs, info, by = "task_id")

mean_hvs = hvs[, .(mean_eagga_hv = mean(eagga_hv), se_eagga_hv = sd(eagga_hv) / sqrt(.N), mean_eagga_md2_hv = mean(eagga_md2_hv), se_eagga_md2_hv = sd(eagga_md2_hv) / sqrt(.N), mean_xgboost_mo_hv = mean(xgboost_mo_hv), se_xgboost_mo_hv = sd(xgboost_mo_hv) / sqrt(.N)), by = .(task)]

mean_hvs = rbind(data.table(task = mean_hvs$task, mean_hv = mean_hvs$mean_eagga_hv, se_hv = mean_hvs$se_eagga_hv, method = "EAGGA_XGBoost"),
                 data.table(task = mean_hvs$task, mean_hv = mean_hvs$mean_eagga_md2_hv, se_hv = mean_hvs$se_eagga_md2_hv, method = "EAGGA_XGBoost_md2"),
                 data.table(task = mean_hvs$task, mean_hv = mean_hvs$mean_xgboost_mo_hv, se_hv = mean_hvs$se_xgboost_mo_hv, method = "XGBoost_MO"))
mean_hvs$task = factor(mean_hvs$task, levels = info$task)
mean_hvs[, method := factor(method, levels = c("XGBoost_MO", "EAGGA_XGBoost", "EAGGA_XGBoost_md2"))]

# Figure 3 + tests in text
g = ggplot(aes(x = task, y = mean_hv, colour = method), data = mean_hvs) +
  geom_point(position = position_dodge(width = 0.5)) +
  geom_errorbar(aes(ymin = mean_hv - se_hv, ymax = mean_hv + se_hv), width = 0.5, position = position_dodge(width = 0.5)) +
  labs(y = "Mean Dominated Hypervolume", x = "Task (p)", colour = "Method") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(vjust = 0.5, angle = 60), legend.position = "bottom")

ggsave("plots/mdhv_xgboost.pdf", plot = g, device = "pdf", width = 6, height = 4)

wilcoxonSignedTest(mean_hvs[method == "EAGGA_XGBoost", ]$mean_hv, mean_hvs[method == "XGBoost_MO", ]$mean_hv) # T = 40, p = 0.00762
wilcoxonSignedTest(mean_hvs[method == "EAGGA_XGBoost_md2", ]$mean_hv, mean_hvs[method == "XGBoost_MO", ]$mean_hv) # T = 50, p = 0.02002

# Analysis of best performance
best = map_dtr(unique(dat$task_id), function(task_id_) {
  map_dtr(unique(dat$repl), function(repl_) {
    tmp = dat[task_id == task_id_ & repl == repl_]
      eagga = tmp[method == "eagga"]$pareto[[1L]][, c("auc_test", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]
      eagga = eagga[which.max(auc_test), ]
      eagga[, method := "eagga"]
      eagga_md2 = tmp[method == "eagga_md2"]$pareto[[1L]][, c("auc_test", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]
      eagga_md2 = eagga_md2[which.max(auc_test), ]
      eagga_md2[, method := "eagga_md2"]
      xgboost_mo = tmp[method == "xgboost_mo"]$pareto[[1L]][, c("auc_test", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]
      xgboost_mo = xgboost_mo[which.max(auc_test), ]
      xgboost_mo[, method := "xgboost_mo"]
      result = rbind(eagga, eagga_md2, xgboost_mo)
      result[, repl := repl_]
      result[, task_id := task_id_]
      result
  })
})
best = merge(best, info, by = "task_id")

mean_best = best[, .(mean_auc_test = mean(auc_test), se_auc_test = sd(auc_test) / sqrt(.N),
                     mean_selected_features_proxy = mean(selected_features_proxy), se_selected_features_proxy = sd(selected_features_proxy) / sqrt(.N),
                     mean_selected_interactions_proxy = mean(selected_interactions_proxy), se_selected_interactions_proxy = sd(selected_interactions_proxy) / sqrt(.N),
                     mean_selected_non_monotone_proxy = mean(selected_non_monotone_proxy), se_selected_non_monotone_proxy = sd(selected_non_monotone_proxy) / sqrt(.N)), by = .(method, task)]
mean_best[, mf := as.numeric(as.factor(method)), by = .(task)]
mean_best[, text_space := mf + 0.5]
mean_best[, method := factor(method, levels = c("eagga", "eagga_md2", "xgboost_mo"), labels = c("EAGGA_XGBoost", "EAGGA_XGBoost_md2", "XGBoost_MO"))]
mean_best$task = factor(mean_best$task, levels = info$task)
mean_best[, epsilon := max(mean_auc_test + se_auc_test) - min(mean_auc_test - se_auc_test), by = .(task)]
mean_best[, ymin := min(mean_auc_test - se_auc_test), by = .(task)]
mean_best[, ymin := ymin - 0.3 * epsilon]

format_f = function(x) {
  x = format(round(x, 2), nsmall = 2)
  x[x == "1.00"] = "1."
  x = gsub("0\\.", replacement = "\\.", x = x)
  x
}

g = ggplot(aes(x = method, y = mean_auc_test, colour = method), data = mean_best) +
  geom_point() +
  geom_errorbar(aes(ymin = mean_auc_test - se_auc_test, ymax = mean_auc_test + se_auc_test), width = 0.2) +
  geom_text(aes(x = text_space, y = -Inf, label = paste0(format_f(mean_selected_features_proxy), "/", format_f(mean_selected_interactions_proxy), "/", format_f(mean_selected_non_monotone_proxy))), vjust = -2, size = 2.5, angle = 30) +
  facet_wrap(~ task, scales = "free", nrow = 5, ncol = 4) +
  geom_blank(aes(y = ymin)) +
  labs(y = "Mean AUC", x = "", colour = "Method") +
  scale_x_discrete(breaks = NULL, limits = c("EAGGA_XGBoost", "EAGGA_XGBoost_md2", "XGBoost_MO", "")) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

mean_mean_best = mean_best[, .(mean_auc_test = mean(mean_auc_test), se_auc_test = sd(mean_auc_test) / sqrt(.N),
                               mean_selected_features_proxy = mean(mean_selected_features_proxy), se_selected_features_proxy = sd(mean_selected_features_proxy) / sqrt(.N),
                               mean_selected_interactions_proxy = mean(mean_selected_interactions_proxy), se_selected_interactions_proxy = sd(mean_selected_interactions_proxy) / sqrt(.N),
                               mean_selected_non_monotone_proxy = mean(mean_selected_non_monotone_proxy), se_selected_non_monotone_proxy = sd(mean_selected_non_monotone_proxy) / sqrt(.N)), by = .(method)]

# Mind the Gap - doesn't change much
get_pessimistic_front = function(front) {
  front = fct * t(front)
  ids = map_lgl(seq_len(ncol(front)), function(j) {
    tmp = map_lgl(seq_len(ncol(front)), function(k) {
      is_dominated(front[, c(j, k)])[2L]
    })
    all(!tmp)
  })
  t(fct * front[, ids])
}

hvs_gap = map_dtr(unique(dat$task_id), function(task_id_) {
  map_dtr(unique(dat$repl), function(repl_) {
    tmp = dat[task_id == task_id_ & repl == repl_]
    eagga = tmp[method == "eagga"]$pareto[[1L]][, ..ys]
    eagga = rbind(eagga, majority[task_id == task_id_ & repl == repl_, ..ys])
    eagga_hv = dominated_hypervolume(t(eagga) * fct, ref = ref)

    eagga_pess = get_pessimistic_front(tmp[method == "eagga"]$pareto[[1L]][, ..ys])
    eagga_pess = rbind(eagga_pess, majority[task_id == task_id_ & repl == repl_, ..ys])
    eagga_pess_hv = dominated_hypervolume(t(eagga_pess) * fct, ref = ref)

    eagga_md2 = tmp[method == "eagga_md2"]$pareto[[1L]][, ..ys]
    eagga_md2 = rbind(eagga_md2, majority[task_id == task_id_ & repl == repl_, ..ys])
    eagga_md2_hv = dominated_hypervolume(t(eagga_md2) * fct, ref = ref)

    eagga_md2_pess = get_pessimistic_front(tmp[method == "eagga_md2"]$pareto[[1L]][, ..ys])
    eagga_md2_pess = rbind(eagga_md2_pess, majority[task_id == task_id_ & repl == repl_, ..ys])
    eagga_md2_pess_hv = dominated_hypervolume(t(eagga_md2_pess) * fct, ref = ref)

    xgboost_mo = tmp[method == "xgboost_mo"]$pareto[[1L]][, ..ys]
    xgboost_mo = rbind(xgboost_mo, majority[task_id == task_id_ & repl == repl_, ..ys])
    xgboost_mo_hv = dominated_hypervolume(t(xgboost_mo) * fct, ref = ref)

    xgboost_mo_pess = get_pessimistic_front(tmp[method == "xgboost_mo"]$pareto[[1L]][, ..ys])
    xgboost_mo_pess = rbind(xgboost_mo_pess, majority[task_id == task_id_ & repl == repl_, ..ys])
    xgboost_mo_pess_hv = dominated_hypervolume(t(xgboost_mo_pess) * fct, ref = ref)

    data.table(eagga_hv = eagga_hv,
               eagga_pess_hv = eagga_pess_hv,
               eagga_gap = eagga_hv - eagga_pess_hv,
               eagga_md2_hv = eagga_md2_hv,
               eagga_md2_pess_hv = eagga_md2_pess_hv,
               eagga_md2_gap = eagga_md2_hv - eagga_md2_pess_hv,
               xgboost_mo_hv = xgboost_mo_hv,
               xgboost_mo_pess_hv = xgboost_mo_pess_hv,
               xgboost_mo_gap = xgboost_mo_hv - xgboost_mo_pess_hv,
               repl = repl_,
               task_id = task_id_)
  })
})

hvs_gap = merge(hvs_gap, info, by = "task_id")

mean_hvs_gap = hvs_gap[, .(mean_eagga_gap = mean(eagga_gap), se_eagga_gap = sd(eagga_gap) / sqrt(.N), mean_eagga_md2_gap = mean(eagga_md2_gap), se_eagga_md2_gap = sd(eagga_md2_gap) / sqrt(.N), mean_xgboost_mo_gap = mean(xgboost_mo_gap), se_xgboost_mo_gap = sd(xgboost_mo_gap) / sqrt(.N)), by = .(task)]

mean_hvs_gap = rbind(data.table(task = mean_hvs_gap$task, mean_hv_gap = mean_hvs_gap$mean_eagga_gap, se_hv_gap = mean_hvs_gap$se_eagga_gap, method = "EAGGA_XGBoost"),
                     data.table(task = mean_hvs_gap$task, mean_hv_gap = mean_hvs_gap$mean_eagga_md2_gap, se_hv_gap = mean_hvs_gap$se_eagga_md2_gap, method = "EAGGA_XGBoost_md2"),
                     data.table(task = mean_hvs_gap$task, mean_hv_gap = mean_hvs_gap$mean_xgboost_mo_gap, se_hv_gap = mean_hvs_gap$se_xgboost_mo_gap, method = "XGBoost_MO"))
mean_hvs_gap$task = factor(mean_hvs_gap$task, levels = info$task)
mean_hvs_gap[, method := factor(method, levels = c("EAGGA_XGBoost", "EAGGA_XGBoost_md2", "XGBoost_MO"))]

g = ggplot(aes(x = task, y = mean_hv_gap, colour = method), data = mean_hvs_gap) +
  geom_point(position = position_dodge(width = 0.5)) +
  geom_errorbar(aes(ymin = mean_hv_gap - se_hv_gap, ymax = mean_hv_gap + se_hv_gap), width = 0.5, position = position_dodge(width = 0.5)) +
  labs(y = "Mean Hypervolume Gap", x = "Task (p)", colour = "Method") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(vjust = 0.5, angle = 60), legend.position = "bottom")

gap_winner = hvs_gap[, .(eagga_winner = eagga_pess_hv > xgboost_mo_hv, eagga_md2_winner = eagga_md2_pess_hv > xgboost_mo_hv, xgboost_mo_winner = xgboost_mo_pess_hv > eagga_hv), by = .(repl, task)]
mean_gap_winner = gap_winner[, .(mean_eagga_winner = mean(eagga_winner), se_eagga_winner = sd(eagga_winner) / sqrt(.N),
                                 mean_eagga_md2_winner = mean(eagga_md2_winner), se_eagga_md2_winner = sd(eagga_md2_winner) / sqrt(.N),
                                 mean_xgboost_mo_winner = mean(xgboost_mo_winner), se_xgboost_mo_winner = sd(xgboost_mo_winner) / sqrt(.N)), by = .(task)]

# Figure 7 eagga ablation anytime valid
g = ggplot(aes(x = runtime, y = mean_anytime_hv, colour = method, fill = method), data = mean_anytime_hvs_runtime[method %nin% c("XGBoost_MO")]) +
  geom_step() +
  geom_stepribbon(aes(ymin = mean_anytime_hv - se_anytime_hv, ymax = mean_anytime_hv + se_anytime_hv), colour = NA, alpha = 0.1) +
  labs(y = "Mean Dominated Hypervolume", x = "Runtime (s)", colour = "Method", fill = "Method", linetype = "Method") +
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x), labels = trans_format("log10", math_format(10^.x))) +
  facet_wrap(~ task, scales = "free", nrow = 5, ncol = 4) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

ggsave("plots/eagga_ablation.pdf", plot = g, device = "pdf", width = 10, height = 8)

# Table 7
table_dat = mean_anytime_hvs_runtime[runtime == 8L * 3600L]
table_mean = xtabs(mean_anytime_hv ~ task + method, data = table_dat)
table_se = xtabs(se_anytime_hv ~ task + method, data = table_dat)

table = map_dtr(seq_len(nrow(table_mean)), function(i) {
  tmp_mean = table_mean[i, ]
  tmp_se = table_se[i, ]
  best = which.max(tmp_mean)
  x = paste0(sprintf(round(tmp_mean, 3), fmt = "%#.3f"), " (", sprintf(round(tmp_se, 3), fmt = "%#.3f"), ")")
  x[best] = paste0("\\textbf{", sprintf(round(tmp_mean[best], 3), fmt = "%#.3f"), "}", " (", sprintf(round(tmp_se[best], 3), fmt = "%#.3f"), ")")
  x = as.list(x)
  names(x) = colnames(table_mean)
  x
})
table = cbind(data.table("Task (p)" = rownames(table_mean)), table)
rownames(table) = NULL

table = xtable(table)
print(table, sanitize.text.function = function(x) x, include.rownames = FALSE)

# Friedman test and cd plot Figure 4
tmp = as.matrix(dcast(mean_anytime_hvs_runtime[runtime == 8L * 3600L], task ~ method, value.var = "mean_anytime_hv")[, -1L])
friedmanTest(tmp)  # Friedman's chi-squared = 52.993, df = 6, p-value = 1.177e-09
pdf(file = "plots/eagga_ablation_cd_1.pdf", width = 12, height = 5)
plotCD(tmp, cex = 1.2)
dev.off()

