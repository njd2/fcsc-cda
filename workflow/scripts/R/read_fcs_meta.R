library(tidyverse)
library(lubridate)

get_timestep <- function(fcs) {
  x <- flowCore::keyword(fcs)[["$TIMESTEP"]]
  ifelse(is.null(x), NA, as.double(x))
}

# the date kw isn't standard :/
parse_date <- function(s) {
  # ASSUME everything is like X-Y-Z
  xs <- str_split(s, "-") %>%
    do.call(rbind, .)
  yearfirst <- str_length(xs[, 1]) == 4
  year_n <- as.integer(if_else(yearfirst, xs[, 1], xs[, 3]))
  day_n <- as.integer(if_else(yearfirst, xs[, 3], xs[, 1]))
  # ASSUME month is in the middle and like JAN or jan
  month <- tolower(xs[, 2])
  month_n <- case_when(
    month == "jan" ~ 1,
    month == "feb" ~ 2,
    month == "mar" ~ 3,
    month == "apr" ~ 4,
    month == "may" ~ 5,
    month == "jun" ~ 6,
    month == "jul" ~ 7,
    month == "aug" ~ 8,
    month == "sep" ~ 9,
    month == "oct" ~ 10,
    month == "nov" ~ 11,
    month == "dec" ~ 12
  )
  make_date(year_n, month_n, day_n)
}

get_kw <- function(kws, x) {
  y <- kws[[x]]
  if (is.null(y)) { NA } else { y }
}

fcs_to_metadata_df <- function(path, i) {
  sf <- str_split_1(basename(path), "_")
  fcs <- flowCore::read.FCS(path, truncate_max_range = FALSE, emptyValue = FALSE)
  kw <- flowCore::keyword(fcs)
  tibble(
    # primary key
    index = i,
    # metadata from filename
    org = sf[3],
    machine = sf[4],
    material = sf[5],
    sop = as.integer(str_sub(sf[6], 5)),
    exp = as.integer(str_sub(sf[7], 2, 2)),
    rep = as.integer(sf[9]),
    filename = basename(path),
    # timestep (which will be important for event analysis)
    timestep = get_timestep(fcs),
    # lots of keywords for sanity checking
    version = get_kw(kw, "FCSversion"),
    btime = get_kw(kw, "$BTIM"),
    etime = get_kw(kw, "$ETIM"),
    volume = get_kw(kw, "$VOL"),
    run_date = get_kw(kw, "$DATE"),
    operator = get_kw(kw, "$OP"),
    serial = get_kw(kw, "$CYTSN"),
    total = get_kw(kw, "$TOT")
    # TODO add ranges?
  )
}

root <- snakemake@input[[1]]

all_fcs_paths <- list.files(root, full.names = TRUE, pattern = "*.fcs")

# cpu go weeeeeeeeeeeeee
future::plan(future::multisession, workers = snakemake@threads)

df_pre <- all_fcs_paths %>%
  furrr::future_imap_dfr(~ fcs_to_metadata_df(.x, .y))

df <- df_pre %>%
  mutate(
    volume = as.double(volume),
    run_length = period_to_seconds(hms(etime) - hms(btime)),
    run_date_real = parse_date(run_date),
    total = as.integer(total)
  ) %>%
  relocate(machine, org) %>%
  arrange(machine, org, sop, exp)

df %>%
  write_tsv(snakemake@output[[1]])
