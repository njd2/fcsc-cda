library(tidyverse)
library(openxlsx)

ensure_empty <- function(df, msg) {
  if (nrow(df) > 0) {
    print(df)
    stop(msg)
  }
}

channel_file <- snakemake@input[[1]]

df_channel <- read.xlsx(channel_file) %>%
  as_tibble() %>%
  set_names(c("site", "inst", "filt", "machine_name", "std_name_long", "desc", "notes")) %>%
  fill(site, inst, .direction = "down") %>%
  mutate(filt = str_replace(filt, "-", "_")) %>%
  separate_wider_delim(filt, "_", names = c("ex", "em", "width")) %>%
  mutate(inst = str_replace(inst, "ILS_FCSC_WG2-001_", "")) %>%
  # the order is arbitrary but these two IDs appear to be the same anyways, so
  # it doesn't matter
  mutate(inst = str_replace(inst, "_U1097", "")) %>%
  mutate(inst = str_replace(inst, "_U0804", "-2")) %>%
  separate_wider_delim(inst, "_", names = c("org", "machine")) %>%
  mutate(
    ex = as.integer(ex),
    em = as.integer(em),
    width = as.integer(width),
    laser = case_when(
      ex < 400 ~ "uv",
      between(ex, 400, 450) ~ "violet",
      between(ex, 450, 500) ~ "blue",
      between(ex, 500, 550) ~ "green",
      between(ex, 550, 600) ~ "yellow",
      ex > 600 ~ "red",
      )
  ) %>%
  select(-site, -desc, -notes) %>%
  mutate(
    std_name = case_when(
      std_name_long == "APC" ~ "apc",
      std_name_long == "APC-Cy7" ~ "ac7",
      std_name_long == "FITC" ~ "fitc",
      std_name_long == "PE" ~ "pe",
      std_name_long == "PE-Cy7" ~ "pc7",
      std_name_long == "PerCP-Cy5.5" ~ "pc55",
      std_name_long == "V450 (FC Beads)/PB " ~ "v450",
      std_name_long == "V500-C (FC Beads)/Aqua" ~ "v500"
    )
  ) %>%
  # fix typos
  mutate(org = if_else(org == "BMSSeattle" & machine == "CantoSORP", "BMSWarren", org)) %>%
  mutate(std_name_long = str_replace(std_name_long, " \\(FC Beads\\)", ""))


# TEST: all lasers and filters in each org/machine/channel should be the same
# (we don't care about X-A vs X-W channels, we strip off the ends so both become
# X and only keep one below)
df_channel %>%
  mutate(f = as.integer(ex * 1e6 + em * 1e3 + width)) %>%
  select(machine_name, org, machine, f) %>%
  group_by(machine_name, org, machine) %>%
  filter(length(unique(f)) > 1) %>%
  ungroup() %>%
  ensure_empty("all channels each lasers/filters/machine/org combo should be the same")

# TEST: all cytometers (except the spectral beasts) should have only one channel
# per FF (otherwise I guess I just pick one?)
df_channel %>%
  select(std_name, machine_name, ex, em, width, org, machine) %>%
  unique() %>%
  group_by(std_name, org, machine) %>%
  filter(n() > 1) %>%
  ungroup() %>%
  ensure_empty("Each std fluor should occur once for each machine/org")

# SANITY CHECK: make sure all standardized names appear on sane lasers and filters
df_channel %>%
  filter(!is.na(ex)) %>%
  ggplot(aes(ex, em)) +
  geom_jitter() +
  facet_wrap(c("std_name")) +
  labs(y = "emission (nm)", x = "excitation (nm)")
ggsave(snakemake@output[["ex_em"]])

df_channel %>%
  mutate(has_name = !is.na(machine_name)) %>%
  mutate(om = sprintf("%s_%s", org, machine)) %>%
  ggplot(aes(std_name_long, y = om, fill = has_name)) +
  geom_tile() +
  labs(x = NULL, y = NULL, fill = "Defined") +
  theme(axis.text.x = element_text(hjust = 1, angle = 90))
ggsave(snakemake@output[["defined"]])

df_channel %>%
  write_tsv(snakemake@output[["table"]])
