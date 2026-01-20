# Torus prime projections (genus 1)

This repository contains a reference implementation for enumerating
unsensed prime projections (4-regular maps) on the torus \(T^2\).

---

## What this code does

For a given number of crossings \(N\), the script:

1. Enumerates all **unsensed candidate projections** on the torus,
   using the permutation model of maps.
2. Applies explicit **projection-level primeness witnesses**
   (2-edge-cuts and split-witnesses).
3. Outputs:
   - the numbers appearing in Table 1 of the paper;
   - machine-readable lists of all **unsensed prime projections**,
     with permutation-level passports.

The implementation is dependency-free and runs with standard Python 3.

---

## Quick start

```bash
python torus_g1_prime_projections.py --N 4


To reproduce multiple rows of Table 1 at once:

python torus_g1_prime_projections.py --Ns 3 4 5 6

For larger values (e.g. N=8):

python torus_g1_prime_projections.py --N 8 --progress


Output

For each N, the script writes to ./out/:

table1_row_N<N>.tex
A LaTeX-ready row for Table 1.

prime_projections_N<N>.json
Machine-readable data for all unsensed prime projections,
including permutation passports and rejection certificates.

prime_projections_N<N>.txt
Human-readable passports.