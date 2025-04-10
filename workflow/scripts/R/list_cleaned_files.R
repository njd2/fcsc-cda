library(tidyverse)

hashes <- read_tsv(snakemake@input[["checksums"]],
                   col_types = "cc",
                   col_names = c("hash", "filepath"))

files <- read_tsv(snakemake@input[["cleaned"]],
                  col_types = "-c", col_names = "filepath") %>%
  mutate(filepath = basename(filepath))

good <- hashes %>%
  right_join(files, by = "filepath")

orgs <- good %>%
  separate_wider_delim(
    filepath,
    "_",
    names = c(NA, NA, "org", "machine", rep(NA, 7))
  ) %>%
  group_by(org, machine) %>%
  tally()

good %>%
  write_tsv(snakemake@output[["cleaned_files"]])

orgs %>%
  write_tsv(snakemake@output[["cleaned_orgs"]])
