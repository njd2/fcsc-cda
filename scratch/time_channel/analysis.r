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

build_step_gates <- function(df, index) {
  i0 <- min(df$event_index)
  i1 <- max(df$event_index)
  thresh <- log10(100 / 50)
  inner_gates <- get_step_gates(integer(), i0, df$tdiff, thresh, 500)
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

df_meta <- read_tsv(
  "../../results/intermediate/meta/fcs_header.tsv.gz",
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

df_meta_min <- df_meta %>%
  mutate(
    min_events = case_when(
      sop == 1 ~ 2e4,
      sop == 2 ~ 1.5e4,
      sop == 3 ~ 5e4
    )
  ) %>%
  select(index, min_events)

df_time_all <- read_tsv("../../results/intermediate/fcs_events_time.tsv.gz", col_types = "iid")

df_low_event <- df_time_all %>%
  group_by(index) %>%
  tally() %>%
  left_join(df_meta_min, by = "index") %>%
  filter(n < min_events) %>%
  mutate(diff = min_events - n)

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

index_wonky <- df_gates %>%
  filter(has_min_n == 0) %>%
  pull(index) %>%
  unique()

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

df_step_gates <- df_time_diff_clean %>%
  filter(!is.na(gate_index)) %>%
  filter(!is.na(tdiff)) %>%
  group_by(index, gate_index) %>%
  group_map(~ build_step_gates(.x, .y$index[[1]])) %>%
  bind_rows()

df_step_gates_min <- df_step_gates %>%
  left_join(df_meta_min, by = "index") %>%
  mutate(
    gate_length = 1 + end - start,
    has_min = gate_length >= min_events
  ) %>%
  group_by(index) %>%
  mutate(has_min_n = sum(has_min)) %>%
  ungroup() %>%
  select(-min_events)

index_no_min <- df_step_gates_min %>%
  filter(has_min_n == 0) %>%
  pull(index) %>%
  unique()

df_top_gates <- df_step_gates_min %>%
  filter(has_min_n > 0) %>%
  group_by(index) %>%
  filter(gate_length == max(gate_length)) %>%
  ungroup()

df_time_all %>%
  filter(index %in% index_wonky) %>%
  ggplot(aes(event_index, time)) +
  geom_point() +
  facet_wrap(c("index"), scales = "free")

df_time_all %>%
  filter(index %in% index_no_min) %>%
  group_by(index) %>%
  ggplot(aes(event_index, time)) +
  geom_point() +
  facet_wrap(c("index"), scales = "free")

df_time_diff_gated <- df_time_all %>%
  right_join(df_top_gates, join_by(index, between(event_index, start, end))) %>%
  select(-start, -end)

future::plan(future::multisession, workers = 4)

df_time_r2 <- df_time_diff_gated %>%
  nest(data = c(event_index, time)) %>%
  mutate(r2 = map_dbl(data, ~ summary(lm(time ~ event_index, data = .x))$r.squared))

# what to do with all these weird bendy cases :(
df_time_r2 %>%
  ggplot(aes(fct_reorder(factor(index), r2), r2)) +
  geom_point() +
  theme(axis.text.x = element_blank(), axis.minor.x = element_blank()) +
  labs(x = NULL, y = "R^2")

df_time_all %>%
  filter(index == 687) %>%
  ## filter(between(event_index, 4950, 5050)) %>%
  ggplot(aes(event_index, time)) +
  geom_point() +
  geom_smooth(method = lm)
