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
  mutate(
    machine_name = case_when(
      # just this one channel is malformed
      machine == "AriaIII" & org == "FDACBER" ~
        if_else(machine_name == "Pacific Blue", "BV421", machine_name),
      # imagestream concats this thingy to the front
      machine == "ImageStreamX-1" ~ sprintf("Intensity_MC_%s", machine_name),
      # the cell stream is a fun thing
      str_detect(machine, "CellStream") & org == "LMNXSEA" ~ case_when(
        std_name == "v450" ~ "UCI_405 - 456/51 - A2",
        std_name == "v500" ~ "UCI_405 - 528/46 - A3",
        std_name == "fitc" ~ "UCI_488 / 730 - 528/46 - C3",
        std_name == "pe" ~ "UCI_488 / 730 - 583/24 - C4",
        std_name == "pc7" ~ "UCI_488 / 730 - 773/56 - C1",
        std_name == "pc55" ~ "UCI_488 / 730 - 702/87 - C6",
        std_name == "apc" ~ "UCI_375 / 642 - 702/87 - B6",
        std_name == "ac7" ~ "UCI_375 / 642 - 773/56 - B1",
      ),
      str_detect(machine, "CellStream") & org == "CellBio" ~ case_when(
        std_name == "v450" ~ "UCI_405 - 456/51 - A2",
        std_name == "v500" ~ "UCI_405 - 528/46 - A3",
        std_name == "fitc" ~ "UCI_488 - 528/46 - C3",
        std_name == "pe" ~ "UCI_488 - 583/24 - C4",
        std_name == "pc7" ~ "UCI_488 - 773/56 - C1",
        std_name == "pc55" ~ "UCI_488 - 702/87 - C6",
        std_name == "apc" ~ "UCI_375/642 - 702/87 - B6",
        std_name == "ac7" ~ "UCI_375/642 - 773/56 - B1",
      ),
      # not sure what happened here
      machine %in% c("MQA10-1", "MQA10-2") & org == "BMSWarren" ~ case_when(
        machine_name == "FL1" ~ "V1",
        machine_name == "FL2" ~ "V2",
        machine_name == "FL3" ~ "B1",
        machine_name == "FL4" ~ "B2",
        machine_name == "FL5" ~ "B3",
        machine_name == "FL6" ~ "B4",
        machine_name == "FL7" ~ "R1",
        machine_name == "FL8" ~ "R2",
      ),
      # not sure what happened here either
      org == "AgilentSD" & machine == "Penteon" ~ case_when(
        machine_name == "B530" ~ "B525",
        machine_name == "V530" ~ "V525",
        TRUE ~ machine_name
      ),
      # or here
      org == "NIBSC" & str_starts(machine, "CantoII") & machine_name == "PerCP-Cy5.5" ~ "PerCP-Cy5-5",
      # or here
      org == "UDel" & str_starts(machine, "Fusion") & machine_name == "PerCP-Cy5.5" ~ "PerCP-Cy5-5",
      # or here
      org == "BDSJ" ~ case_when(
        machine_name == "488/527/32" ~ "FITC",
        machine_name == "488/586/42" ~ "PE",
        machine_name == "488/700/54" ~ "PerCP-Cy5.5",
        machine_name == "488/783/56" ~ "PE-Cy7",
        machine_name == "640/660/10" ~ "APC",
        machine_name == "640/783/56" ~ "APC-Cy7",
        machine_name == "405/448/45" ~ "Pacific Blue",
        machine_name == "405/528/45" ~ "Aqua Dye"
      ),
      TRUE ~ machine_name
    )
  ) %>%
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
