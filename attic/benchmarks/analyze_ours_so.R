library(data.table)
library(mlr3misc)
library(bbotk)
library(ggplot2)
library(emoa)
library(scmamp)

source("openml_tasks.R")
setorderv(info, col = "p")
info$name[1L] = "banknote"
info$name[2L] = "blood"
info$name[6L] = "climate"
info$name[17L] = "ozone"
info[, task := paste0(name, " (", p, ")")]
info = info[, c("id", "task"), with = FALSE]
colnames(info) = c("task_id", "task")

# Section 5.1
dat = readRDS("results/eagga_ours_so.rds")
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
    rest = tmp[method %nin% c("eagga", "eagga_md2"), ..ys]
    rest = rbind(rest, majority[task_id == task_id_ & repl == repl_, ..ys])
    rest_hv = dominated_hypervolume(t(rest) * fct, ref = ref)
    data.table(eagga_hv = eagga_hv, eagga_md2_hv = eagga_md2_hv, rest_hv = rest_hv, repl = repl_, task_id = task_id_)
  })
})

hvs = merge(hvs, info, by = "task_id")

mean_hvs = hvs[, .(mean_eagga_hv = mean(eagga_hv), se_eagga_hv = sd(eagga_hv) / sqrt(.N), mean_eagga_md2_hv = mean(eagga_md2_hv), se_eagga_md2_hv = sd(eagga_md2_hv) / sqrt(.N), mean_rest_hv = mean(rest_hv), se_rest_hv = sd(rest_hv) / sqrt(.N)), by = .(task)]

mean_hvs = rbind(data.table(task = mean_hvs$task, mean_hv = mean_hvs$mean_eagga_hv, se_hv = mean_hvs$se_eagga_hv, method = "EAGGA_XGBoost"),
                 data.table(task = mean_hvs$task, mean_hv = mean_hvs$mean_eagga_md2_hv, se_hv = mean_hvs$se_eagga_md2_hv, method = "EAGGA_XGBoost_md2"),
                 data.table(task = mean_hvs$task, mean_hv = mean_hvs$mean_rest_hv, se_hv = mean_hvs$se_rest_hv, method = "Competitors"))
mean_hvs$task = factor(mean_hvs$task, levels = info$task)

# Figure 2 + tests in text
g = ggplot(aes(x = task, y = mean_hv, colour = method), data = mean_hvs) +
  geom_point(position = position_dodge(width = 0.5)) +
  geom_errorbar(aes(ymin = mean_hv - se_hv, ymax = mean_hv + se_hv), width = 0.5, position = position_dodge(width = 0.5)) +
  labs(y = "Mean Dominated Hypervolume", x = "Task (p)", colour = "Method") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(vjust = 0.5, angle = 60), legend.position = "bottom")

ggsave("plots/mdhv_all.pdf", plot = g, device = "pdf", width = 6, height = 4)

wilcoxonSignedTest(mean_hvs[method == "EAGGA_XGBoost", ]$mean_hv, mean_hvs[method == "Competitors", ]$mean_hv) # T = 0, p < 0.001
wilcoxonSignedTest(mean_hvs[method == "EAGGA_XGBoost_md2", ]$mean_hv, mean_hvs[method == "Competitors", ]$mean_hv) # T = 0, p < 0.001

get_is_dominated_by_eagga = function(type = c("eagga", "eagga_md2"), dat, ref) {
  is_dominated_by_eagga = map_dtr(unique(dat$task_id), function(task_id_) {
    map_dtr(unique(dat$repl), function(repl_) {
      tmp = dat[task_id == task_id_ & repl == repl_]
      eagga = tmp[method == type]$pareto[[1L]][, ..ys]
      eagga = rbind(eagga, majority[task_id == task_id_ & repl == repl_, ..ys])
      n_eagga = nrow(eagga)
      map_dtr(c("xgboost", "ebm", "glmnet", "rf"), function(method_) {
        other = tmp[method == method_, ..ys]
        data.table(is_dominated = is_dominated(t(rbind(eagga, other)) * fct)[n_eagga + 1L], method = method_, repl = repl_, task_id = task_id_)
      })
    })
  })
  is_dominated_by_eagga
}

is_dominated_by_eagga = get_is_dominated_by_eagga("eagga", dat, ref)
is_dominated_by_eagga_md2 = get_is_dominated_by_eagga("eagga_md2", dat, ref)

# Table 1 a) and b)
mean_is_dominated_by_eagga = is_dominated_by_eagga[, .(mean_is_dominated = mean(is_dominated), se_is_dominated = sd(is_dominated) / sqrt(.N)), by = .(method)]
mean_is_dominated_by_eagga_md2 = is_dominated_by_eagga_md2[, .(mean_is_dominated = mean(is_dominated), se_is_dominated = sd(is_dominated) / sqrt(.N)), by = .(method)]
round(mean_is_dominated_by_eagga[c(2, 3, 4, 1), -1], 2L)
round(mean_is_dominated_by_eagga_md2[c(2, 3, 4, 1), -1], 2L)

get_is_dominated_by_competitors = function(type = c("eagga", "eagga_md2"), dat, ref) {
  is_dominated_by_competitors = map_dtr(unique(dat$task_id), function(task_id_) {
    map_dtr(unique(dat$repl), function(repl_) {
      tmp = dat[task_id == task_id_ & repl == repl_]
      eagga = tmp[method == type]$pareto[[1L]][, ..ys]
      eagga = rbind(eagga, majority[task_id == task_id_ & repl == repl_, ..ys])
      n_eagga = nrow(eagga)
      rest = tmp[method %nin% c("eagga", "eagga_md2"), ..ys]
      rest = rbind(rest, majority[task_id == task_id_ & repl == repl_, ..ys])
  
      data.table(is_dominated_fraction = mean(is_dominated(t(rbind(eagga, rest)) * fct)[1:n_eagga]), all_dominated = all(is_dominated(t(rbind(eagga, rest)) * fct)[1:n_eagga]), repl = repl_, task_id = task_id_)
    })
  })
  is_dominated_by_competitors
}

is_dominated_by_competitors_eagga = get_is_dominated_by_competitors("eagga", dat, ref)
is_dominated_by_competitors_eagga_md2 = get_is_dominated_by_competitors("eagga_md2", dat, ref)

mean_is_dominated_by_competitors_eagga = is_dominated_by_competitors_eagga[, .(mean_is_dominated_fraction = mean(is_dominated_fraction), se_is_dominated_fraction = sd(is_dominated_fraction) / sqrt(.N),
                                                                               mean_all_dominated = mean(all_dominated), se_all_dominated = sd(all_dominated) / sqrt(.N))]
mean_is_dominated_by_competitors_eagga_md2 = is_dominated_by_competitors_eagga_md2[, .(mean_is_dominated_fraction = mean(is_dominated_fraction), se_is_dominated_fraction = sd(is_dominated_fraction) / sqrt(.N),
                                                                                       mean_all_dominated = mean(all_dominated), se_all_dominated = sd(all_dominated) / sqrt(.N))]

# Analysis of best performance, Figure 5 in supplement
best = map_dtr(unique(dat$task_id), function(task_id_) {
  map_dtr(unique(dat$repl), function(repl_) {
    tmp = dat[task_id == task_id_ & repl == repl_]
      eagga = tmp[method == "eagga"]$pareto[[1L]][, c("auc_test", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]
      eagga = eagga[which.max(auc_test), ]
      eagga[, method := "eagga"]
      eagga_md2 = tmp[method == "eagga_md2"]$pareto[[1L]][, c("auc_test", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]
      eagga_md2 = eagga_md2[which.max(auc_test), ]
      eagga_md2[, method := "eagga_md2"]
      rest = tmp[method %nin% c("eagga", "eagga_md2"), c("auc_test", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy", "method")]
      result = rbind(eagga, eagga_md2, rest)
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
mean_best[, method := factor(method, levels = c("eagga", "eagga_md2", "ebm", "glmnet", "rf", "xgboost"), labels = c("EAGGA_XGBoost", "EAGGA_XGBoost_md2", "EBM", "Elastic-Net", "RF", "XGBoost"))]
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
  scale_x_discrete(breaks = NULL, limits = c("EAGGA_XGBoost", "EAGGA_XGBoost_md2", "EBM", "Elastic-Net", "RF", "XGBoost", "")) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

ggsave("plots/best_all.pdf", plot = g, device = "pdf", width = 10, height = 8)

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

    rest = tmp[method %nin% c("eagga", "eagga_md2"), ..ys]
    rest = rbind(rest, majority[task_id == task_id_ & repl == repl_, ..ys])
    rest_hv = dominated_hypervolume(t(rest) * fct, ref = ref)

    rest_pess = get_pessimistic_front(tmp[method %nin% c("eagga", "eagga_md2"), ..ys])
    rest_pess = rbind(rest_pess, majority[task_id == task_id_ & repl == repl_, ..ys])
    rest_pess_hv = dominated_hypervolume(t(rest_pess) * fct, ref = ref)

    data.table(eagga_hv = eagga_hv,
               eagga_pess_hv = eagga_pess_hv,
               eagga_gap = eagga_hv - eagga_pess_hv,
               eagga_md2_hv = eagga_md2_hv,
               eagga_md2_pess_hv = eagga_md2_pess_hv,
               eagga_md2_gap = eagga_md2_hv - eagga_md2_pess_hv,
               rest_hv = rest_hv,
               rest_pess_hv = rest_pess_hv,
               rest_gap = rest_hv - rest_pess_hv,
               repl = repl_,
               task_id = task_id_)
  })
})

hvs_gap = merge(hvs_gap, info, by = "task_id")

mean_hvs_gap = hvs_gap[, .(mean_eagga_gap = mean(eagga_gap), se_eagga_gap = sd(eagga_gap) / sqrt(.N), mean_eagga_md2_gap = mean(eagga_md2_gap), se_eagga_md2_gap = sd(eagga_md2_gap) / sqrt(.N), mean_rest_gap = mean(rest_gap), se_rest_gap = sd(rest_gap) / sqrt(.N)), by = .(task)]

mean_hvs_gap = rbind(data.table(task = mean_hvs_gap$task, mean_hv_gap = mean_hvs_gap$mean_eagga_gap, se_hv_gap = mean_hvs_gap$se_eagga_gap, method = "EAGGA_XGBoost"),
                     data.table(task = mean_hvs_gap$task, mean_hv_gap = mean_hvs_gap$mean_eagga_md2_gap, se_hv_gap = mean_hvs_gap$se_eagga_md2_gap, method = "EAGGA_XGBoost_md2"),
                     data.table(task = mean_hvs_gap$task, mean_hv_gap = mean_hvs_gap$mean_rest_gap, se_hv_gap = mean_hvs_gap$se_rest_gap, method = "Competitors"))
mean_hvs_gap$task = factor(mean_hvs_gap$task, levels = info$task)

g = ggplot(aes(x = task, y = mean_hv_gap, colour = method), data = mean_hvs_gap) +
  geom_point(position = position_dodge(width = 0.5)) +
  geom_errorbar(aes(ymin = mean_hv_gap - se_hv_gap, ymax = mean_hv_gap + se_hv_gap), width = 0.5, position = position_dodge(width = 0.5)) +
  labs(y = "Mean Hypervolume Gap", x = "Task (p)", colour = "Method") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(vjust = 0.5, angle = 60), legend.position = "bottom")

gap_winner = hvs_gap[, .(eagga_winner = eagga_pess_hv > rest_hv, eagga_md2_winner = eagga_md2_pess_hv > rest_hv, rest_winner = rest_pess_hv > eagga_hv), by = .(repl, task)]
mean_gap_winner = gap_winner[, .(mean_eagga_winner = mean(eagga_winner), se_eagga_winner = sd(eagga_winner) / sqrt(.N),
                                 mean_eagga_md2_winner = mean(eagga_md2_winner), se_eagga_md2_winner = sd(eagga_md2_winner) / sqrt(.N),
                                 mean_rest_winner = mean(rest_winner), se_rest_winner = sd(rest_winner) / sqrt(.N)), by = .(task)]

# Illustrational example in supplement Table 2
tmp = dat[task_id == 190137 & repl == 5L]
tmp_eagga = tmp[method == "eagga"]$pareto[[1L]][, c("auc_test", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]
setorderv(tmp_eagga, cols = "auc_test")
round(tmp_eagga, 3L)

tmp_competitors = map_dtr(c("ebm", "glmnet", "rf", "xgboost"), function(method_) {
  tmp[method == method_, c("auc_test", "selected_features_proxy", "selected_interactions_proxy", "selected_non_monotone_proxy")]
})
round(tmp_competitors, 3L)
