library(tidyverse)

df_combos <- read_tsv(snakemake@input[["combos"]], col_types = "ciiicc")

df_event_sum <- read_tsv(snakemake@input[["event"]], col_types = "iccd")

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

df_channels <- read_tsv(snakemake@input[["linkage"]], col_types = "cciiicccc")

df_raw_channels <- read_tsv(snakemake@input[["params"]], col_types = "ccccddd")

df_event_missing_channels <- df_event_sum %>%
  filter(metric == "missing" & !channel %in% c("time", "fsc", "ssc")) %>%
  group_by(index) %>%
  summarize(
    all_missing = all(value > 0),
    any_missing = any(value > 0)
  ) %>%
  ungroup()

df_missing_time <- df_event_sum %>%
  filter(metric == "mean", channel == "time") %>%
  mutate(
    is_missing_time = is.na(value)
  ) %>%
  select(index, is_missing_time)

df_meta_errors <- df_meta %>%
  left_join(df_event_missing_channels, by = "index") %>%
  left_join(df_missing_time, by = "index") %>%
  right_join(
    df_combos,
    by = c("org", "machine", "rep", "exp", "material", "sop")
  ) %>%
  group_by(org, machine) %>%
  mutate(
    om = sprintf("%s_%s", org, machine),
    group = case_when(
      sop == 1 ~ "SOP 1",
      sop == 2 ~ case_when(
        exp == 1 ~ "SOP 2: Matrix 1",
        exp == 2 ~ "SOP 2: Matrix 2",
        exp == 3 ~ "SOP 2: Matrix 3",
        exp == 4 ~ "SOP 2: Matrix 3/4"
      ),
      sop == 3 ~ case_when(
        str_detect(material, "fmo") ~ "SOP 3 - Test FMO",
        exp == 1 ~ "SOP 3: Test Pheno",
        exp == 2 ~ "SOP 3: Test Count",
        exp == 3 ~ "SOP 3: QC Count",
        exp == 4 ~ "SOP 3: QC Pheno"
      )
    ),
    is_missing_file = is.na(index),
    is_missing_timestep = is.na(timestep),
    has_multi_serial = length(unique(na.omit(serial))) > 1,
    is_empty = total == 0
  ) %>%
  ungroup() %>%
  relocate(index, machine, org, group, matches("^(has|is|any|all)_"))

df_meta_errors %>%
  mutate(
    category = case_when(
      is_missing_file ~ "missing file",
      is_empty ~ "no events",
      is_missing_time ~ "no time channel",
      all_missing ~ "all channels missing",
      any_missing ~ "some channels missing",
      TRUE ~ "OK"
    )
  ) %>%
  ## mutate(category = fct_relevel(category, "OK", after = Inf)) %>%
  ## mutate(category = fct_relevel(category, "all channels missing", after = 3)) %>%
  ggplot(aes(y = om, fill = category)) +
  geom_bar() +
  facet_wrap(c("group"), nrow = 1) +
  labs(x = "Number of Files", y = NULL, fill = "category")
ggsave(snakemake@output[["errors"]], width = 16)

df_event_sum %>%
  filter(metric == "missing") %>%
  filter(value > 0) %>%
  left_join(df_meta, by = "index") %>%
  select(org, machine, channel, sop, exp, material, rep) %>%
  write_tsv(snakemake@output[["missing"]])
