library(tidyverse)

billd_gates <- function(df, keys) {
  i <- keys$index
  t <- df$total[[1]]
  n <- nrow(df)
  if (t == 0) {
    tibble()
  } else if (n == 1 & all(is.na(df$issue))) {
    tibble(index = i, start = 1, end = t, issue = NA)
  } else {
    # TODO add padding?
    end <- df$event_index
    # ASSUME here that all issues only cover two points (some discontinuity that
    # has a start point and an end point) in which case we can start and end
    # before/after these points to ignore the issue
    tibble(
      index = i,
      # ASSUME the minimum start index for the diff dataframe is 2 (since we
      # can't differentiate the first point)
      start = c(2, end + 1),
      end = c(end - 1, t),
      issue = c(df$issue, NA)
    )
  }
}

get_ss <- function(x) {
  sum((x - mean(x)) ^ 2)
}

step_r2 <- function(gate, x, sst) {
  ss1 <- get_ss(x[1:gate])
  ss2 <- get_ss(x[(gate + 1):length(x)])
  1 - (ss1 + ss2) / sst
}

get_step_gates <- function(acc, start_index, time_diffs, thresh, min_size) {
  # This function is a basic step function curve fitter. The "gate" is the
  # boundary of the step, and it is placed such that the means of the two
  # partitions on either side of the step result in a maximized R^2, where R^2
  # is 1 - (SS1 + SS2) / SStot where SS1/2 are the sum of squares for each
  # partition and SStot is the sum of squares for the entire vector. This
  # function will continue partitioning the vector until an arbitrary threshold
  # is achieved
  #
  # NOTE thresh should be a positive number
  n <- length(time_diffs)
  sst <- get_ss(time_diffs)
  # set tol to 0.5 so we stop the model when we get to the nearest integer-ish
  res <- optimize(step_r2, c(1, n), time_diffs, sst, maximum = TRUE, tol = 0.5)
  gate <- floor(res$maximum)
  d1 <- time_diffs[1:gate]
  d2 <- time_diffs[(gate+1):n]
  rate_diff <- log10(mean(d1) / mean(d2))
  # if gate is larger than our minimum size and produces two partitions with
  # flow rates that differ beyond our threshold, place the gate, and try to
  # gate the two new partitions, otherwise return whatever gates we have so
  # far, which may be none
  if (abs(rate_diff) > thresh && n - gate > min_size && gate > min_size) {
    acc %>%
      get_step_gates(start_index, d1, thresh, min_size) %>%
      c(start_index + gate) %>%
      get_step_gates(start_index + gate + 1, d2, thresh, min_size)
  } else {
    acc
  }
}

build_step_gates <- function(df, index, thresh, min_size) {
  i0 <- min(df$event_index)
  i1 <- max(df$event_index)
  .thresh <- log10(100 / thresh)
  inner_gates <- get_step_gates(integer(), i0, df$tdiff, .thresh, min_size)
  # minus 1 because the diff vector is N - 1
  realstart <- i0 - 1
  if (length(inner_gates) == 0) {
    tibble(
      index = index,
      start = realstart,
      end = i1
    )
  } else {
    tibble(
      index = index,
      start = c(realstart, inner_gates),
      end = c(inner_gates, i1)
    )
  }
}

THRESH <- snakemake@params[["thresh"]]
MIN_SIZE <- snakemake@params[["min_size"]]

MIN_EVENT_SOP1 <- snakemake@params[["min_event1"]]
MIN_EVENT_SOP2 <- snakemake@params[["min_event2"]]
MIN_EVENT_SOP3 <- snakemake@params[["min_event3"]]

#
# 0) Read stuff
#

df_meta <- read_tsv(
  snakemake@input[["meta"]],
  col_types = cols(
    run_date_real = "D",
    index = "i",
    sop = "i",
    exp = "i",
    rep = "i",
    timestep = "d",
    version = "d",
    volume = "d",
    run_length = "d",
    total = "i",
    .default = "c"
  )
)

df_time_all <- read_tsv(
  snakemake@input[["events"]],
  col_types = "iid"
)

df_meta_min <- df_meta %>%
  mutate(
    min_events = case_when(
      sop == 1 ~ MIN_EVENT_SOP1,
      sop == 2 ~ MIN_EVENT_SOP2,
      sop == 3 ~ MIN_EVENT_SOP3
    )
  ) %>%
  select(index, min_events)

#
# 1) Get all fcs files that don't have the required number of events in them.
# Exclude these from the files we will analyze later to lower the computational
# burden ;)
#

df_low_event <- df_time_all %>%
  group_by(index) %>%
  tally() %>%
  left_join(df_meta_min, by = "index") %>%
  filter(n < min_events) %>%
  mutate(diff = min_events - n)

#
# 2) Find anomalies in the time channel.
#
# There are 2 anomalies we care about
# - large pauses: These are usually due to a clog, bubble, tripping over power
#   cables, etc. They appear as large "gaps" in the time vs event index curve.
# - large non-monotonic deviations: In theory, the time vs event curve should
#   increase monotonically, since events with a higher index occur after those
#   with a lower index. In some cases, the time channel is "noisy" enough so
#   that this isn't true, but the non-monotonicity is usually small. In other
#   cases the time channel appears to restart from zero (ish) in the middle of
#   the run (unclear why this happens). We care about the latter case, since
#   this likely indicated some machine/user error.
#
# Note also large negative spikes. For some reason some cytometers will set one
# random event to have a zero time value in the middle of a run, which appears
# as a gigantic "negative spike" in the time vs event index curve. We don't care
# about these for now but they will end up getting removed later since they make
# the data easier to compute.

df_time_issues <- df_time_all %>%
  anti_join(df_low_event, by = "index") %>%
  group_by(index) %>%
  mutate(
    tdiff = time - lag(time),
    tdiff_norm = tdiff / (max(time) - min(time))
  ) %>%
  ungroup() %>%
  mutate(
    large_neg = tdiff < -1000 & lead(tdiff) > 1000,
    issue = case_when(
      # large negative spike
      large_neg ~ 1,
      # large pause (clog or bubble); make sure it isn't immediately after a negative spike
      tdiff_norm > 0.05 & tdiff >= 0 & !lag(large_neg) ~ 2,
      # non-monotonic (machine spazzed out...or something)
      tdiff < -100 ~ 3,
      !is.na(tdiff) ~ 0
    )
  ) %>%
  select(-tdiff, -tdiff_norm, -time, -large_neg)

#
# 3) Place a gate at the boundary of each issue from above and filter out the
#    files that don't have enough events in any gates.
#

df_gates <- df_time_issues %>%
  # only filter out issues that aren't random spikes (since these probably
  # aren't fatal)
  filter(!is.na(issue)) %>%
  filter(issue > 1) %>%
  right_join(df_meta, by = "index") %>%
  group_by(index) %>%
  group_map(billd_gates) %>%
  discard(~ nrow(.x) == 0) %>%
  bind_rows() %>%
  anti_join(df_low_event, by = "index") %>%
  left_join(df_meta_min, by = "index") %>%
  rename(gate_issue = issue) %>%
  mutate(has_min = 1 + end - start >= min_events) %>%
  group_by(index) %>%
  mutate(
    gate_index = row_number(),
    has_min_n = sum(has_min)
  ) %>%
  ungroup() %>%
  select(-min_events)

df_time_diff_gated <- df_time_all %>%
  anti_join(df_low_event, by = "index") %>%
  left_join(df_gates, join_by(index, between(event_index, start, end))) %>%
  select(-start, -end)

df_time_diff_clean <- df_time_diff_gated %>%
  add_column(issue = df_time_issues$issue) %>%
  filter(has_min) %>%
  filter(!is.na(gate_index)) %>%
  # ASSUME at this point there should be very few issues left, since
  # non-monotonic switches and large gaps should be at the gate boundaries and
  # filtered out by removing rows without a gate index above. Therefore the only
  # things that should be left are a few large negative spikes, which can easily
  # be removed. This slightly warps the data since the time diffs surrounding
  # the spike will be combined, but there are so few this shouldn't matter.
  filter(is.na(issue) | issue == 0) %>%
  group_by(index, gate_index) %>%
  mutate(
    tdiff = time - lag(time),
  ) %>%
  ungroup() %>%
  select(-has_min, -has_min_n, -issue)

#
# 4) Identify regions where the time channel is "flat"
#
# In an ideal world, the events occurring in a flow cytometer would be Poissonic
# and therefore the differences in time between events would be exponentially
# distributed. This is not the case here (unfortunately) because a) some
# cytometers are note precise enough to distinguish between two events and
# therefore create a difference of 0 (not in the domain of an exp distribution)
# b) some difference are negative (see (2) above) c) cytometers just aren't that
# precise and deviations from Poissonc behavior may not indicate anything is
# "wrong".
#
# Therefore, we hack this problem in a much simpler manner by recursively
# regressing a step function to the differentiated time vs event index curve.
# Most of the deviations we care about are caused by flow rate changes, which
# will manifest as changes in slope or changes in the mean time difference
# between events. We can measure this by fitting a step function to the
# differences, where the step occurs at a flow rate change. Then we can place a
# gate at all the step changes that are above a certain threshold.
#
# Then we can repeat the gating filtration process we did above where we remove
# FCS files that don't have a gate with the minimum required event number
# inside.

df_step_gates <- df_time_diff_clean %>%
  filter(!is.na(gate_index)) %>%
  filter(!is.na(tdiff)) %>%
  group_by(index, gate_index) %>%
  group_map(~ build_step_gates(.x, .y$index[[1]], THRESH, MIN_SIZE)) %>%
  bind_rows()

df_step_gates_min <- df_step_gates %>%
  left_join(df_meta_min, by = "index") %>%
  mutate(
    gate_length = 1 + end - start,
    has_min = gate_length >= min_events
  ) %>%
  group_by(index) %>%
  mutate(has_min_n = sum(has_min)) %>%
  ungroup()

df_top_gates <- df_step_gates_min %>%
  filter(has_min_n > 0) %>%
  group_by(index) %>%
  filter(gate_length == max(gate_length)) %>%
  ungroup()

#
# 5) For all remaining FCS files, compute a linear regression and rank by R^2.
#
# This will allow us to manually inspect the "worst curves" that we may
# potentially let through.
#

df_time_diff_gated <- df_time_all %>%
  right_join(df_top_gates, join_by(index, between(event_index, start, end))) %>%
  select(-start, -end)

df_time_r2 <- df_time_diff_gated %>%
  nest(data = c(event_index, time)) %>%
  mutate(r2 = map_dbl(data, ~ summary(lm(time ~ event_index, data = .x))$r.squared))

#
# 6) Export everything
#

df_time_r2 %>%
  select(index, r2) %>%
  left_join(df_meta, by = "index") %>%
  left_join(df_top_gates, by = "index") %>%
  select(-has_min, -has_min_n) %>%
  relocate(r2, .after = everything()) %>%
  write_tsv(snakemake@output[["valid"]])

df_low_event %>%
  write_tsv(snakemake@output[["insufficient"]])

df_gates_has_issues <- df_gates %>%
  filter(has_min_n == 0)

df_gates_has_issues %>%
  write_tsv(snakemake@output[["indices_issues"]])

df_time_all %>%
  filter(index %in% unique(df_gates_has_issues$index)) %>%
  write_tsv(snakemake@output[["events_issues"]])

df_step_gates_nonlinear <- df_step_gates_min %>%
  filter(has_min_n == 0)

df_step_gates_nonlinear %>%
  write_tsv(snakemake@output[["indices_nonlinear"]])

df_time_all %>%
  filter(index %in% unique(df_step_gates_nonlinear$index)) %>%
  write_tsv(snakemake@output[["events_nonlinear"]])

df_time_r2 %>%
  arrange(r2) %>%
  left_join(df_meta, by = "index") %>%
  mutate(
    i = row_number(),
    om = sprintf("%s_%s", org, machine)
  ) %>%
  ggplot(aes(i, r2)) +
  geom_point() +
  labs(x = NULL, y = "R^2") +
  facet_wrap(c("om"), ncol = 4)
ggsave(snakemake@output[["pareto"]], width = 8, height = 11)
