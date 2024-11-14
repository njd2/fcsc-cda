library(tidyverse)

df_insufficient <- read_tsv("../../results/final/time/insufficient_events.tsv.gz")

df_anomalies <- read_tsv("../../results/final/time/indices_with_issues.tsv.gz")

df_nonlinear <- read_tsv("../../results/final/time/indices_nonlinear.tsv.gz")

df_meta <- read_tsv("../../results/intermediate/meta/fcs_header.tsv.gz")

df_ins <- df_insufficient %>%
  mutate(percent_complete = n / min_events * 100) %>%
  left_join(df_meta, by = "index") %>%
  add_column(reason = "insufficient events")

df_ano <- df_anomalies %>%
  left_join(df_meta, by = "index") %>%
  mutate(
    min_events = case_when(sop == 1 ~ 2e4, sop == 2 ~ 1.5e4, sop == 3 ~ 5e4),
    n = end - start + 1,
    percent_complete = n / min_events * 100
  ) %>%
  group_by(index) %>%
  filter(n == max(n)) %>%
  ungroup() %>%
  add_column(reason = "anomalies")
  
df_nl <- df_nonlinear %>%
  left_join(df_meta, by = "index") %>%
  mutate(
    min_events = case_when(sop == 1 ~ 2e4, sop == 2 ~ 1.5e4, sop == 3 ~ 5e4),
    n = end - start + 1,
    percent_complete = n / min_events * 100
  ) %>%
  group_by(index) %>%
  filter(n == max(n)) %>%
  ungroup() %>%
  add_column(reason = "nonlinear")

bind_rows(
  df_ins,
  df_ano,
  df_nl
) %>%
  arrange(desc(percent_complete), org, machine, sop, exp) %>%
  select(org, machine, sop, exp, material, rep, n, min_events, percent_complete, reason) %>%
  write_tsv("files_with_too_few_events.tsv")
