library(tidyverse)

df_meta_issues <- read_tsv(
  snakemake@input[["issues"]],
  col_types = cols(
    file_index = "i",
    filepath = "c",
    om = "c",
    group = "c",
    material = "c",
    exp = "i",
    sop = "i",
    rep = "i",
    total = "-",
    has_voltage_variation = "l",
    has_gain_variation =  "l",
    percent_complete = "d",
    has_multi_serial = "l",
    has_multi_cytometer = "l",
    has_multi_system = "l",
    missing_time = "l",
    missing_colors = "l",
    missing_scatter_area = "l",
    missing_scatter_height = "l",
    .default = "-"
  )
)

anomaly_idx <- unique(
  read_tsv(
    snakemake@input[["anomaly"]],
    col_types = "i---",
    col_names = c("index")
  )[[1]]
)

bad_time_idx <- unique(
  read_tsv(
    snakemake@input[["events"]],
    col_types = "i--",
    col_names = c("index")
  )[[1]]
)

df_meta_issues %>%
  mutate(
    missing_file = is.na(file_index),
    has_insufficient_events = percent_complete < 100,
    has_time_anomaly = file_index %in% bad_time_idx & !(file_index %in% anomaly_idx),
    has_time_nonlinear = file_index %in% bad_time_idx & file_index %in% anomaly_idx,
    # adjust as needed
    has_any_error = has_voltage_variation |
      has_gain_variation |
      has_multi_serial |
      has_multi_cytometer |
      has_multi_system |
      missing_time |
      missing_colors |
      missing_scatter_area |
      missing_scatter_height |
      missing_file |
      has_insufficient_events |
      has_time_anomaly |
      has_time_nonlinear
  ) %>%
  select(-percent_complete) %>%
  group_by(om, group) %>%
  mutate(group_has_any_error = any(has_any_error)) %>%
  ungroup() %>%
  arrange(om, group, file_index) %>%
  write_tsv(snakemake@output[[1]])
