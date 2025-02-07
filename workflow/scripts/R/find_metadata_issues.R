library(tidyverse)

ps <- snakemake@params

MIN_EVENT_SOP1 <- ps[["min_events"]][["sop1"]]
MIN_EVENT_SOP2 <- ps[["min_events"]][["sop2"]]
MIN_EVENT_SOP3 <- ps[["min_events"]][["sop3"]]
VDIFF_LIMIT <- ps[["detector_limits"]][["voltage"]]
GDIFF_LIMIT <- ps[["detector_limits"]][["gain"]]

ensure_empty <- function(df, msg) {
  if (nrow(df) > 0) {
    print(df)
    stop(sprintf("ERROR %s", msg))
  }
}

df_exclusions <- read_tsv(
  snakemake@input[["exclusions"]],
  col_types = "ciiiccc"
) %>%
  rename(exclusion_comment = comment)

df_combos <- read_tsv(
  snakemake@input[["combos"]],
  col_types = "ciiicc"
) %>%
  mutate(
    om = sprintf("%s_%s", org, machine),
    group = case_when(
      sop == 1 ~ "SOP 1",
      sop == 2 ~ case_when(
        eid == 1 ~ "SOP 2: Matrix 1",
        eid == 2 ~ "SOP 2: Matrix 2",
        eid == 3 ~ "SOP 2: Matrix 3",
        eid == 4 ~ "SOP 2: Matrix 3/4"
      ),
      sop == 3 ~ case_when(
        str_detect(material, "fmo") ~ "SOP 3 - Test FMO",
        eid == 1 ~ "SOP 3: Test Pheno",
        eid == 2 ~ "SOP 3: Test Count",
        eid == 3 ~ "SOP 3: QC Count",
        eid == 4 ~ "SOP 3: QC Pheno"
      )
    )
  ) %>%
  # Matrix 2/3 and the cryoPBMC test panels are required for the dataset to
  # be considered "complete"
  mutate(
    required = (sop == 2 & eid != 1) |
      (sop == 3 & eid == 1 & !str_detect(material, "fmo"))
  )

df_meta <- read_tsv(
  snakemake@input[["meta"]],
  col_types = cols(
    file_index = "i",
    sop = "i",
    eid = "i",
    rep = "i",
    tot = "i",
    timestep = "d",
    vol = "d",
    lost = "i",
    abrt = "i",
    version = "c",
    total_time = "d",
    date = "D",
    .default = "c"
  )
) %>%
  select(-group, -om)

df_params <- read_tsv(
  snakemake@input[["params"]],
  col_types = cols(
    file_index = "i",
    param_index = "i",
    eid = "-",
    sop = "-",
    rep = "-",
    material = "-",
    maxrange = "d",
    log_decades = "d",
    log_zero = "d",
    gain = "d",
    wavelength = "d",
    power = "d",
    percent_emitted = "d",
    detector_voltage = "d",
    .default = "c"
  )
)

df_channels <- read_tsv(
  snakemake@input[["channels"]],
  col_types = "ccccc----"
)

df_params_std <- df_params %>%
  rename(machine_name = shortname) %>%
  ## mutate(machine_name = str_replace(shortname, "-(A|Area)$", "")) %>%
  left_join(df_channels, by = c("org", "machine", "machine_name")) %>%
  filter(!is.na(std_name))

# if this is false we need to add lots of extra logic to deal with log scales
df_params_std %>%
  filter(log_decades > 0) %>%
  ensure_empty("Some parameters are logarithmic")

#
# verify all color voltages/gains are the same compared to SOP 1 (as written)
#

df_sop1_voltgain <- df_params_std %>%
  select(-org, -machine) %>%
  left_join(df_meta, by = "file_index") %>%
  filter(sop == 1) %>%
  filter(!std_name %in% c("fsc_a", "fsc_h", "ssc_a", "ssc_h", "time")) %>%
  select(org, machine, std_name, gain, detector_voltage) %>%
  group_by(org, machine, std_name) %>%
  summarize(
    gain1_sd = sd(gain),
    voltage1_sd = sd(detector_voltage),
    gain1_n = sum(!is.na(gain)),
    voltage1_n = sum(!is.na(detector_voltage)),
    gain1 = mean(gain),
    voltage1 = mean(detector_voltage),
    .groups = "drop"
  )

df_sop1_voltgain %>%
  filter(between(gain1_n, 1, 8) | between(voltage1_n, 1, 8)) %>%
  ensure_empty("Voltage/gains are not entirely present/absent in SOP1")

df_sop1_voltgain %>%
  filter(!(is.na(gain1_sd) | gain1_sd == 0 | is.na(voltage1_sd) | voltage1_sd == 0)) %>%
  ensure_empty("Voltage/gains are not the same in SOP1")

df_voltgain_diff <- df_params_std %>%
  filter(!std_name %in% c("fsc_a", "fsc_h", "ssc_a", "ssc_h", "time")) %>%
  select(-org, -machine) %>%
  left_join(df_meta, by = "file_index") %>%
  filter(sop != 1) %>%
  left_join(df_sop1_voltgain, by = c("org", "machine", "std_name")) %>%
  mutate(
    vdiff = (detector_voltage - voltage1) / voltage1,
    gdiff = (gain - gain1) / gain1
  ) %>%
  select(file_index, org, machine, om, group, material, std_name, std_name_long, vdiff, gdiff)

df_voltgain_diff %>%
  write_tsv(snakemake@output[["voltgain"]])


vdiff_issue_indices <- df_voltgain_diff %>%
  filter(vdiff > VDIFF_LIMIT) %>%
  pull(file_index) %>%
  unique()

gdiff_issue_indices <- df_voltgain_diff %>%
  filter(gdiff > GDIFF_LIMIT) %>%
  pull(file_index) %>%
  unique()

#
# check channel presence/absence
#
# deal with time, scatter, and "color" separately since these all have different
# failure modes

df_allowed_missing_scatter <- df_channels %>%
  filter(std_name %in% c("fsc_a", "ssc_a", "fsc_h", "ssc_h")) %>%
  filter(is.na(machine_name)) %>%
  select(org, machine) %>%
  unique() %>%
  add_column(scatter_can_be_missing = TRUE)

df_height_required <- df_meta %>%
  # require height scatter channels if the events are cells, since height is
  # needed for doublet exclusion
  mutate(height_required = str_detect(material, "PBMC|lyoLeuk")) %>%
  select(file_index, height_required)

df_file_channels <- df_params_std %>%
  pivot_wider(
    id_cols = c(file_index, org, machine),
    names_from = std_name,
    values_from = param_index
  ) %>%
  right_join(select(df_meta, file_index), by = "file_index") %>%
  left_join(df_allowed_missing_scatter, by = c("org", "machine")) %>%
  replace_na(list(scatter_can_be_missing = FALSE)) %>%
  left_join(df_height_required, by = "file_index") %>%
  mutate(
    missing_time = is.na(time),
    missing_colors = if_any(c(v450, v500, fitc, pc55, pe, pc7, apc, ac7), is.na),
    missing_scatter_area = if_any(c(fsc_a, ssc_a), is.na) &
      !scatter_can_be_missing,
    missing_scatter_height = if_any(c(fsc_h, ssc_h), is.na) &
      !scatter_can_be_missing &
      height_required
  )

df_file_channels %>%
  write_tsv(snakemake@output[["channels"]])

df_file_channels_short <- df_file_channels %>%
  select(file_index, starts_with("missing"))

df_multi_issues <- df_meta %>%
  group_by(org, machine) %>%
  mutate(
    has_multi_serial = length(unique(cytsn)) > 1,
    has_multi_cytometer = length(unique(cyt)) > 1,
    has_multi_system = length(unique(sys)) > 1
  ) %>%
  ungroup() %>%
  select(file_index, starts_with("has_"))

df_issues <- df_combos %>%
  left_join(df_meta, by = c("org", "machine", "sop", "eid", "rep", "material")) %>%
  left_join(df_exclusions, by = c("org", "machine", "sop", "eid", "rep", "material")) %>%
  mutate(
    missing_file = is.na(file_index),
    min_events = case_when(
      sop == 1 ~ MIN_EVENT_SOP1,
      sop == 2 ~ MIN_EVENT_SOP2,
      sop == 3 ~ MIN_EVENT_SOP3
    ),
    has_manual_exclusion = !is.na(exclusion_comment),
    has_voltage_variation = file_index %in% vdiff_issue_indices,
    has_gain_variation = file_index %in% gdiff_issue_indices,
    percent_complete = tot / min_events * 100,
    has_insufficient_events = percent_complete < 100
  ) %>%
  left_join(df_multi_issues, by = "file_index") %>%
  left_join(df_file_channels_short, by = "file_index") %>%
  group_by(org, machine) %>%
  mutate(
    has_incomplete_set = sum(missing_file & required) > 0 |
      ## sum(has_voltage_variation & required) > 0 |
      ## sum(has_gain_variation & required) > 0 |
      sum(has_insufficient_events & required) > 0
  ) %>%
  ungroup() %>%
  select(
    file_index, org, machine, material, sop, eid, rep, om, group, filepath,
    percent_complete, missing_file, has_voltage_variation, has_gain_variation,
    has_insufficient_events, has_multi_serial, has_multi_cytometer,
    has_multi_system, missing_time, missing_colors, missing_scatter_area,
    missing_scatter_height, has_incomplete_set, has_manual_exclusion,
    exclusion_comment
  ) %>%
  arrange(file_index)

df_issues %>%
  write_tsv(snakemake@output[["issues"]])

## df_issues %>%
##   filter(
##     !(
##       has_voltage_variation |
##         has_gain_variation |
##         percent_complete < 100 |
##         ## has_multi_serial |
##         ## has_multi_cytometer |
##         ## has_multi_system |
##         missing_time |
##         missing_colors |
##         missing_scatter_area |
##         missing_scatter_height
##     )
##   ) %>%
##   write_tsv(snakemake@output[["clean"]])
