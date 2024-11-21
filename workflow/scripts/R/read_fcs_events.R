library(tidyverse)

ensure_empty <- function(df, msg) {
  if (nrow(df) > 0) {
    print(df)
    stop(msg)
  }
}

find_fuzzy_channel <- function(pat, channels) {
  if (is.na(pat)) {
    NA
  } else {
    .pat <- pat %>%
      str_replace("\\(", "\\\\(") %>%
      str_replace("\\)", "\\\\)") %>%
      sprintf("^%s((-| )(A|Area))?$", .)
    cs <- channels[str_detect(channels, .pat)]
    if (length(cs) == 1) {
      # return the single area channel
      cs[[1]]
    } else {
      NA
    }
  }
}

get_timestep <- function(fcs) {
  x <- flowCore::keyword(fcs)[["$TIMESTEP"]]
  ifelse(is.null(x), NA, as.double(x))
}

# TODO deal with truncation warnings
fcs_to_event_df <- function(path, i, df_mapping, df_tfs) {
  sf <- str_split_1(basename(path), "_")
  .org <- sf[3]
  .machine <- sf[4]
  fcs <- flowCore::read.FCS(path, truncate_max_range = FALSE, emptyValue = FALSE)
  fcs_channel_names <- fcs@exprs %>% colnames()
  ## channel_mapping <- df_mapping %>%
  ##   filter(org == .org, machine == .machine) %>%
  ##   select(std_name, machine_name)
  ## tfs_mapping <- df_tfs %>%
  time_channel <- df_tfs %>%
    filter(std_name == "time") %>%
    filter(org == .org, machine == .machine) %>%
    ## select(std_name, machine_name) %>%
    pull(machine_name) %>%
    first()
  ## all_mapping <-  bind_rows(channel_mapping, tfs_mapping)
  ## time_channel <- find_time_channel(fcs_channel_names)
  ## all_std_names <- all_mapping$std_name
  n <- nrow(fcs@exprs)
  if (n == 0) {
    NULL
  } else {
    # ASSUME that each org/machine has all 11 channels and that they are
    # all in the same order
    ## m <- all_mapping$machine_name %>%
    ##   map_chr(~ find_fuzzy_channel(.x, fcs_channel_names)) %>%
    ##   map(~ if (is.na(.x)) { rep(NA, n) } else { fcs@exprs[, .x] }) %>%
    ##   unlist() %>%
    ##   matrix(nrow = n, byrow = FALSE)
    m <- fcs@exprs[, time_channel]
    colnames(m) <- all_std_names
    m <- cbind(m, index = i, event_index = 1:n)
    m
  }
}

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
    .default = "c"
  ),
)

df_channels <- read_tsv(
  snakemake@input[["linkage"]],
  col_types = cols(
    ex = "i",
    em = "i",
    width = "i",
    .default = "c"
  )
)

df_tfs <- read_tsv(snakemake@input[["tfs_mapping"]], col_types = "cccc")

df_meta %>%
  anti_join(df_channels, by = c("org", "machine")) %>%
  select(org, machine) %>%
  unique() %>%
  ensure_empty("all org/machine combos should have a channel linkage")

# TODO booooooo this is redundant
df_meta %>%
  anti_join(df_tfs, by = c("org", "machine")) %>%
  select(org, machine) %>%
  unique() %>%
  ensure_empty("all org/machine combos should have a tfs linkage")

root <- snakemake@input[["fcs"]]

future::plan(future::multisession, workers = snakemake@threads)

df <- df_meta %>%
  mutate(paths = map2(file.path(root, filename), index, list)) %>%
  pull(paths) %>%
  furrr::future_map(~ fcs_to_event_df(.x[[1]], .x[[2]], df_channels, df_tfs)) %>%
  do.call(rbind, .) %>%
  as_tibble()

df %>%
  select(index, event_index, time) %>%
  filter(!is.na(time)) %>%
  write_tsv(snakemake@output[["time"]])

df %>%
  group_by(index) %>%
  summarize(
    across(
      c(-event_index),
      list(
        missing = ~ sum(is.na(.x)) / n(),
        mean = ~ mean(.x, na.rm = TRUE),
        sd = ~ sd(.x, na.rm = TRUE),
        min = ~ if (all(is.na(.x))) { NA } else { min(.x, na.rm = TRUE) },
        q1 = ~ quantile(.x, 0.25, na.rm = TRUE),
        med = ~ median(.x, na.rm = TRUE),
        q3 = ~ quantile(.x, 0.75, na.rm = TRUE),
        max = ~ if (all(is.na(.x))) { NA } else { max(.x, na.rm = TRUE) }
      )
    )
  ) %>%
  pivot_longer(cols = -index) %>%
  separate_wider_delim(name, "_", names = c("channel", "metric")) %>%
  write_tsv(snakemake@output[["summary"]])
