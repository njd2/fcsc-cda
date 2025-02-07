A pipeline to run all analysis for NIST [Flow Cytometry Standards
Consortium](https://www.nist.gov/programs-projects/nist-flow-cytometry-standards-consortium)
in an automated fashion. For now, this only covers Working Group 2, but other
Working Groups will likely be added given that the same machinery can be reused
in many cases.

# Key Features

NOTE: this pipeline is still a work in progress and details are subject to
change at any moment. Steps marked "TODO" below have yet to be implemented.

## FCS file standardization and clean up

Input FCS files will be checked for completeness and metadata integrity prior to
doing any real analysis.

Passing FCS files will be saved with standardized channel names and metadata,
and unused channels will be removed.

Other event-level clean up will also be performed, such as time channel anomaly
detection and edge event removal.

## Calibration (SOP1)

Using FC beads with known equivalent reference fluorophore (ERF) values, each
channel for each instrument will be put on a common scale.

## Compensation (SOP2)

TODO

## Cell Assay Analysis (SOP3)

TODO

# Installation

Everything is managed with conda/mamba, so ensure you have a working
installation of either. This readme will be presented in terms of mamba, which
is preferred for speed.

Currently tested with:

```
> mamba --version
mamba 1.5.3
conda 23.10.0
```

To install the base pipeline software, run the following:

```
mamba env create -f env.yml
```

This will create an environment called `fscs-cda`. Activate with:

```
conda activate fcsc-cda
```

# Usage

The pipeline is implemented in `snakemake` which will allow the runtime to scale
to large compute clusters if the need arises. For now, the pipeline can run on a
reasonably fast computer (13th+ Gen Intel i7 with 16-20 cores) in a few minutes
on all WG2 files. Most steps are heavily multitheaded so having many cores will
increase performance proportionately.

## Obtaining input files

Input fcs files are not included with the pipeline for privacy reasons. Obtain a
copy of the WG2 FCS files and copy them to `resources/fcs_wg2`.

A list of all the required FCS input files can be viewed at
`static/nist_wg2_md5.tsv`. The FCS file names and contents must match exactly,
the pipeline will refuse to run if any checksums fail or if any files are
missing.

## Running locally

Once input files have been obtained, run the following (in the activated conda
environment built from above):

```
 snakemake -c <cores> \
   --use-conda \
   --rerun-incomplete \
   --configfile=config/config.yml \
   all
```

Substitute `<cores>` with the number of CPU cores on your machine you wish to
sacrifice to the cause.

## Output files

The layout of output files is still in flux and could change drastically in
the future as more steps are added.

However, the following files can generally be found in `results`

* tsv files showing every keyword value in all inputs, organized in a somewhat
  sane manner.
* tsv files showing the results of passing/failing input files at various steps
* tsv files with calibrated slopes for each channel
* modified input FCS files with various stages of cleanup (unused channel
  removal, standardized naming, edge event removal, time channel clean-up, etc)
* html reports for passing/failing inputs, anomaly detection, calibration, etc

## Configuration

The configuration file is `config/config.yml` and is currently hardcoded with
this git repository.

In the future, this file will be drastically expanded, and a more user-friendly
way to configure your own custom pipeline (if you want) will be made available.

