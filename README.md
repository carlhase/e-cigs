# Partial replication package: Enforcement Spillovers in Product Markets: Evidence from E-cigarettes

This repository contains a select portion of the replication package for my working paper **Enforcement Spillovers in Product Markets: Evidence from E-cigarettes**. The repository contains a modular, testable, and production-ready Python pipeline for constructing store-level e-cigarette price indexes from convenience-store scanner data.

## Highlights
- Modular code in `src/`
- Automated tests in `tests/`
- Data validation through pandera schemas
- Production scripts in `scripts/`
- Batch scripts for quick execution
- GitHub Actions CI for continuous integration

The pipeline processes store-product-month files from approx. 35,000 stores and computes Young price indexes using a multi-stage weighting algorithm, and outputs:
- Per-store vape price index files
- A concatenated panel of all store-month results

## Paper
- https://drive.google.com/file/d/1DcD6880gdh3eS6qZhdoi1LeL64maZkCi/view?usp=sharing

