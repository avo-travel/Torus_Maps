# Torus_Maps

Reference implementation for enumeration of projections and diagrams on the torus

This repository contains a standalone Python reference implementation for the
enumeration of knot and link projections on the torus, and for the subsequent
enumeration of knot and link diagrams supported by these projections.

The code accompanies the paper

> **Prime projections and diagrams on the torus: unsensed maps, canonical enumeration, and state-sum invariants**
> A. Omelchenko

and implements, in a fully reproducible way, the computational pipeline described
in Sections 3–4 and Appendices A–B of the paper.

The repository is intended for verification and reproduction of the results,
rather than as a general-purpose software library.

---

## Scope and conventions

The implementation follows the conventions stated explicitly in the paper:

* projections are treated up to **unsensed equivalence** (including orientation reversal);
* bigon faces are allowed at the projection level, but the **bigon reduction rule**
  is applied at the diagram level;
* an optional **global crossing switch** is used to factor out mirror symmetry;
* for link diagrams, an optional **over/under participation convention** is applied
  when reproducing published tables.

All conventions are implemented explicitly in the code and can be enabled or
disabled via command-line options.

---

## Repository structure

```
Torus_Maps/
│
├── projections/
│   └── torus_g1_prime_projections.py
│
├── diagrams/
│   └── torus_g1_diagrams_from_prime_projections.py
│
├── invariants/
│   └── generalized_kauffman_X_torus.py
│
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
│
├── out/        (created automatically; not tracked by git)
└── cache/      (created automatically; not tracked by git)
```

* `projections/`
  Enumeration of unsensed prime projections on the torus (Section 3, Appendix A).

* `diagrams/`
  Enumeration of knot and link diagrams supported by fixed projections, together
  with state-sum invariant evaluation (Section 4, Appendix B).

* `invariants/`
  Implementation of generalized Kauffman-type invariants on the torus.

The directories `out/` and `cache/` are created automatically during execution
and are intentionally excluded from version control.

---

## Requirements

* Python **3.10+** (tested with Python 3.12–3.14)
* No third-party packages are required.

The scripts are fully standalone and do not require installation.

---

## Running the code

All scripts are executed directly from the repository root.

### 1. Enumerating prime projections on the torus

To enumerate unsensed prime projections with $N$ crossings:

```bash
python projections/torus_g1_prime_projections.py --N 4
```

This writes the projection data to `out/`, including:

* a LaTeX-ready row for the projection count table;
* machine-readable JSON files `prime_projections_N{N}.json`;
* human-readable summaries `prime_projections_N{N}.txt`.

The projection enumeration is the computationally expensive step and is typically
run once per value of $N$.

---

### 2. Enumerating diagrams and computing invariants

To enumerate knot and link diagrams supported by the precomputed projections and
to evaluate their state-sum invariants:

```bash
python diagrams/torus_g1_diagrams_from_prime_projections.py --N 6
```

This script:

* reads `prime_projections_N{N}.json` from `out/`;
* enumerates admissible crossing assignments;
* applies diagram-level filters;
* computes generalized bracket and polynomial invariants;
* performs incremental “new-at-$N$” classification.

Results are written to `out/`, and an internal classification library is stored
in `cache/` for reuse in subsequent runs.

---

## Reproducibility

All numerical results reported in the paper are obtained by running the scripts
in this repository with the stated options.

For each diagram, the code records:

* the unsensed canonical representative of the underlying projection;
* the crossing assignment;
* the computed state-sum invariants and classification keys.

This makes every table entry and comparison directly checkable.

---

## Citation

If you use this code in academic work, please cite it as:

```
A. Omelchenko,
Torus_Maps: Reference implementation for enumeration of projections and diagrams on the torus,
https://github.com/avo-travel/Torus_Maps
```

A machine-readable citation entry is provided in `CITATION.cff`.

---

## License

This code is released under the MIT License. See the file `LICENSE` for details.
