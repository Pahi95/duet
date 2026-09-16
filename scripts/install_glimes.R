#!/usr/bin/env Rscript
# install_glimes.R -- install GLIMES (Wu, Zhou & Chen 2025, Genome Biol 26:58) into a
# project-local library, so nothing is written to the user's R library.
#
#   Rscript scripts/install_glimes.R [lib]
#
# lib defaults to <project root>/../_Rlib; scripts that use GLIMES put the same
# directory first on .libPaths(). Its dependencies (MASS, Matrix, edgeR,
# SummarizedExperiment, MAST) must already be installed; the script stops if not.

args <- commandArgs(trailingOnly = TRUE)
here <- tryCatch(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))),
                 error = function(e) getwd())
lib <- if (length(args) >= 1) args[1] else normalizePath(file.path(here, "..", "..", "_Rlib"), mustWork = FALSE)
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .libPaths()))
cat("[glimes] library:", lib, "\n[glimes] tempdir:", tempdir(), "\n")

deps <- c("MASS", "Matrix", "edgeR", "SummarizedExperiment", "MAST", "remotes")
ok <- vapply(deps, requireNamespace, logical(1), quietly = TRUE)
print(ok)
if (!all(ok)) stop("missing dependencies: ", paste(deps[!ok], collapse = ", "))

remotes::install_github("C-HW/GLIMES", lib = lib, dependencies = FALSE, upgrade = "never",
                        build_vignettes = FALSE, force = FALSE)
suppressPackageStartupMessages(library(GLIMES, lib.loc = lib))
ref <- tryCatch(remotes:::package2remote("GLIMES", lib = lib)$sha, error = function(e) NA)
cat("[glimes] GLIMES", as.character(packageVersion("GLIMES", lib.loc = lib)),
    "commit", ref, "\n[glimes] exports:", paste(ls("package:GLIMES"), collapse = ", "), "\n")
