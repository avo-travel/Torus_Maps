#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
torus_g1_diagrams_from_prime_projections.py

Enumerate torus (genus-one) knot/link diagrams supported by a precomputed pool of
unsensed prime projections, and classify them by state-sum invariants.

This script is designed to work together with:

  - torus_g1_prime_projections.py
      Produces JSON files  out/prime_projections_N{N}.json  containing the prime
      torus projections (maps) at a fixed crossing number N.

  - invariants/generalized_kauffman_X_torus.py
      Implements the generalized bracket <D>(a,x) on the torus and the normalized
      polynomial X_D(a,x) for knots, following standard state-sum definitions.

The main feature of THIS script is that it does *not* enumerate projections.
Instead, it loads them from the JSON produced by torus_g1_prime_projections.py.
This is much faster for larger N (e.g. N=6,7,8), and makes computations reproducible:
the projection pool is fixed by an input file.

------------------------------------------------------------------------
Quick start

  # Compute "new at N=6" knot/link classes (relative to smaller N),
  # using prime projections from ./out and writing results to ./out
  python diagrams/torus_g1_diagrams_from_prime_projections.py --N 6

  # Same, but store caches and outputs in custom directories
  python diagrams/torus_g1_diagrams_from_prime_projections.py --N 8 --proj-dir out --out out --cache cache

------------------------------------------------------------------------
What is being enumerated?

For each prime projection P with N crossings (a connected 4-regular map on T^2),
a diagram is determined by an over/under choice at each crossing:
    b : Vert(P) -> {0,1}.
So each projection supports 2^N crossing assignments in principle.

This script optionally applies three common conventions/filters:

  - Global crossing switch (optional):
      b ~ 1-b (simultaneous switch at all crossings), implemented by fixing one bit.

  - Bigon rule (optional):
      a fast local rejection rule along bigon faces. You can choose:
        * off
        * naive_equal   : require b(u)=b(v) for every 2-face (u,v)
        * parity        : a slightly refined equality rule (recommended)

  - Link over/under participation (optional, links only):
      for multi-component diagrams, require that each straight-ahead component
      appears at least once as an overpass and at least once as an underpass
      among *mixed* crossings.

All of these switches are explicit command-line options.

------------------------------------------------------------------------
Classification key (invariants)

For links we use the generalized bracket  <D>(a,x).
For knots we use the normalized polynomial  X_D(a,x)=(-a)^(-3w(D)) <D>(a,x).

Diagrams are grouped by a configurable *canonicalized* polynomial key:
  - raw           : exact polynomial key as produced
  - tabulation    : also identify mirror/inversion a <-> a^{-1}, overall sign,
                    and an overall power shift a^k (used to compare with some
                    published genus-one tabulations)

------------------------------------------------------------------------
Performance note (important for N=7,8)

A direct implementation would compute a full 2^N-state sum for every crossing
assignment b. That is expensive.

This script uses a standard speed-up: for a fixed projection, the *geometry* of a
smoothed state depends only on the XOR pattern
    t = b XOR s,
where s is the state choice at crossings. Therefore:

  - We precompute, once per projection, for all t in {0,1}^N:
        (gamma(t), delta(t)) = (#contractible loops, #essential loops)

  - For each diagram b we then evaluate the state sum using these cached values.
    This removes the expensive loop-counting step from the inner diagram loop.

------------------------------------------------------------------------
Outputs

For a given target N, the script writes (into --out):

  - *_summary.txt
      readable summary of settings and counts

  - *_new_knots_classes.csv
    *_new_links_classes.csv
      one representative per NEW class at N (relative to the cached library)

A persistent cache file (pickle) is written into --cache to avoid rebuilding the
"seen up to N-1" library from scratch each time.

------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from pathlib import Path

# --------------------------------------------------------------------
# Repository-root import setup
#
# This script lives in ./diagrams. We add the repository root to sys.path
# so that we can import the local `invariants` package regardless of the
# current working directory (CLI, PyCharm, etc.).
# --------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import csv
import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Local dependency: invariant computations and basic polynomial helpers.
# Expected location: ./invariants/generalized_kauffman_X_torus.py
try:
    from invariants import generalized_kauffman_X_torus as kx
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "ERROR: Could not import invariants/generalized_kauffman_X_torus.py.\n"
        "Make sure you are running from a clone of the repository with this structure:\n"
        "  invariants/generalized_kauffman_X_torus.py\n"
        "  diagrams/torus_g1_diagrams_from_prime_projections.py\n"
        "If you are running from the repository root, try:\n"
        "  python diagrams/torus_g1_diagrams_from_prime_projections.py --N 4\n"
        f"Import error: {exc}"
    )
# -----------------------------
# Types (small, explicit)
# -----------------------------

Laurent = Dict[int, int]              # exponent -> integer coefficient
PolyX = Dict[int, Laurent]            # x-degree -> Laurent polynomial in a


# -----------------------------
# Configuration / options
# -----------------------------

@dataclass(frozen=True)
class DiagramEnumConfig:
    # Target crossing number
    N: int

    # Where to read projections from:
    # expects files prime_projections_N{N}.json
    proj_dir: Path = REPO_ROOT / "out"

    # Output/caching
    out_dir: Path = REPO_ROOT / "out"
    cache_dir: Path = REPO_ROOT / "cache"

    # Diagram-level switches
    quotient_global_switch: bool = True
    bigon_rule: str = "parity"  # off | naive_equal | parity
    link_over_under_participation: bool = True

    # Invariant / key options
    link_invariant: str = "bracket"  # bracket | X  (X is orientation-dependent for links)
    poly_canon: str = "tabulation"   # raw | tabulation
    count_key: str = "poly"          # poly | skeleton

    # Writhe sign convention (passed to kx.FourRegularMap.writhe / normalized_X)
    sign_convention: int = 1

    # Incremental mode: if True, count only NEW classes at N (relative to smaller N)
    new_at_N: bool = True
    N_min: int = 2

    # Seed a tiny library of known N=0 and N=1 knot types so they are not
    # counted as "new" later in new-at-N mode (only used for count_key="poly").
    seed_trivial_knots: bool = True

    def validate(self) -> None:
        if self.N < 0:
            raise ValueError("N must be nonnegative.")
        if self.N_min < 0:
            raise ValueError("N_min must be nonnegative.")
        if self.N_min > self.N:
            raise ValueError("N_min must be <= N.")
        if self.bigon_rule not in ("off", "naive_equal", "parity"):
            raise ValueError("bigon_rule must be one of: off, naive_equal, parity")
        if self.link_invariant not in ("bracket", "X"):
            raise ValueError("link_invariant must be one of: bracket, X")
        if self.poly_canon not in ("raw", "tabulation"):
            raise ValueError("poly_canon must be one of: raw, tabulation")
        if self.count_key not in ("poly", "skeleton"):
            raise ValueError("count_key must be one of: poly, skeleton")
        if self.sign_convention not in (-1, 1):
            raise ValueError("sign_convention must be ±1")

    def fingerprint(self) -> str:
        """
        Stable cache key for the *meaning* of "a class".

        If you change any option that affects:
          - which diagrams are accepted
          - how the polynomial key is canonicalized
        then you must not reuse the old library cache. The fingerprint is used
        to automatically separate caches for different settings.
        """
        payload = {
            "N_min": self.N_min,
            "quotient_global_switch": self.quotient_global_switch,
            "bigon_rule": self.bigon_rule,
            "link_over_under_participation": self.link_over_under_participation,
            "link_invariant": self.link_invariant,
            "poly_canon": self.poly_canon,
            "count_key": self.count_key,
            "sign_convention": self.sign_convention,
            "seed_trivial_knots": self.seed_trivial_knots,
            "library_format": 2,
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:12]

    def tag(self) -> str:
        """
        Human-readable string used in output filenames.
        """
        parts = [
            f"g1",
            f"N{self.N}",
            "new" if self.new_at_N else "all",
            "gswitch" if self.quotient_global_switch else "nogswitch",
            f"bigon{self.bigon_rule}",
            "ou" if self.link_over_under_participation else "noou",
            f"link{self.link_invariant}",
            f"canon{self.poly_canon}",
            f"key{self.count_key}",
        ]
        return "_".join(parts)


# -----------------------------
# Library cache (seen classes)
# -----------------------------

@dataclass
class SeenEntry:
    first_N: int
    proj_id: int
    bits: str
    raw_poly_key: Tuple
    poly_latex: str
    skeleton_key: Optional[Tuple] = None


@dataclass
class LibraryDB:
    """
    Persistent storage for "first occurrence" representatives.

    Keys are *canonicalized* polynomial keys (hashable tuples).
    """
    fingerprint: str
    max_N: int = -1
    knots: Dict[Tuple, SeenEntry] = field(default_factory=dict)
    links: Dict[Tuple, SeenEntry] = field(default_factory=dict)

    def contains(self, is_knot: bool, key: Tuple) -> bool:
        return key in (self.knots if is_knot else self.links)

    def add(self, is_knot: bool, key: Tuple, entry: SeenEntry) -> None:
        if is_knot:
            self.knots[key] = entry
        else:
            self.links[key] = entry


# -----------------------------
# Polynomial key helpers
# -----------------------------

def poly_key(poly: PolyX) -> Tuple:
    """
    Convert PolyX to a deterministic, hashable key.

    We sort by x-degree, and within each coefficient we sort by exponent of a.
    """
    items: List[Tuple[int, Tuple[Tuple[int, int], ...]]] = []
    for m in sorted(poly.keys()):
        la = poly[m]
        la_items = tuple(sorted((e, c) for (e, c) in la.items() if c != 0))
        if la_items:
            items.append((m, la_items))
    return tuple(items)


def invert_a_poly(poly: PolyX) -> PolyX:
    """
    Apply a -> a^{-1} to the whole polynomial in x (coefficientwise).
    """
    out: PolyX = {}
    for m, la in poly.items():
        out[m] = {-e: c for e, c in la.items() if c != 0}
    return out


def shift_a_poly(poly: PolyX, k: int) -> PolyX:
    """
    Multiply the whole polynomial by a^k (coefficientwise).
    """
    if k == 0:
        return {m: dict(la) for m, la in poly.items()}
    out: PolyX = {}
    for m, la in poly.items():
        out[m] = {e + k: c for e, c in la.items() if c != 0}
    return out


def scale_poly(poly: PolyX, s: int) -> PolyX:
    """
    Multiply the whole polynomial by an integer scalar s (typically ±1).
    """
    if s == 1:
        return {m: dict(la) for m, la in poly.items()}
    out: PolyX = {}
    for m, la in poly.items():
        out[m] = {e: s * c for e, c in la.items() if c != 0}
    return out


def normalize_poly_sign(poly: PolyX) -> PolyX:
    """
    Multiply by -1 if the lexicographically first nonzero coefficient is negative.
    """
    for m in sorted(poly.keys()):
        la = poly[m]
        for e in sorted(la.keys()):
            c = la[e]
            if c != 0:
                return scale_poly(poly, -1) if c < 0 else {mm: dict(ll) for mm, ll in poly.items()}
    return {m: dict(la) for m, la in poly.items()}


def min_a_exponent(poly: PolyX) -> Optional[int]:
    """
    Minimum exponent of a among all nonzero terms, or None for the zero polynomial.
    """
    mins: List[int] = []
    for la in poly.values():
        for e, c in la.items():
            if c != 0:
                mins.append(e)
    return min(mins) if mins else None


def canonical_poly_key_tabulation(poly: PolyX) -> Tuple:
    """
    Canonical polynomial key used for "tabulation-style" comparisons.

    Operations considered equivalent:
      - inversion a -> a^{-1}  (mirror)
      - overall sign ±1
      - overall shift a^k      (we shift so the minimal a-exponent becomes 0)

    We do NOT change x-degrees.
    """
    candidates: List[Tuple] = []
    for p in (poly, invert_a_poly(poly)):
        me = min_a_exponent(p)
        if me is None:
            p2 = p
        else:
            p2 = shift_a_poly(p, -me)
        p3 = normalize_poly_sign(p2)
        candidates.append(poly_key(p3))
    return min(candidates)


def compute_count_key(M: kx.FourRegularMap, poly: PolyX, cfg: DiagramEnumConfig) -> Tuple[Tuple, Tuple, Optional[Tuple]]:
    """
    Return (count_key, raw_poly_key, skeleton_key_or_None).

    count_key is what we use to decide if two diagrams are in the same class.
    raw_poly_key is always the exact poly_key(poly) before canonicalization.
    skeleton_key is computed only if cfg.count_key == "skeleton".
    """
    raw = poly_key(poly)
    if cfg.poly_canon == "raw":
        canon = raw
    else:
        canon = canonical_poly_key_tabulation(poly)

    if cfg.count_key == "poly":
        return canon, raw, None

    # skeleton key: derived from the *raw* polynomial (as in the earlier script)
    sk = M.skeleton_from_poly(poly)
    sk_canon = M.canonicalize_skeleton(sk)
    sk_key = tuple((m, tuple(sk_canon[m])) for m in sorted(sk_canon.keys()))
    return sk_key, raw, sk_key


# -----------------------------
# Input: prime projections JSON
# -----------------------------

def projections_json_path(proj_dir: Path, N: int) -> Path:
    return proj_dir / f"prime_projections_N{N}.json"


def load_prime_projections(proj_dir: Path, N: int) -> List[dict]:
    """
    Load the list under the 'prime_projections' key from prime_projections_N{N}.json.
    """
    path = projections_json_path(proj_dir, N)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing projection file: {path}\n"
            "Generate it first using torus_g1_prime_projections.py."
        )
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "prime_projections" not in payload:
        raise ValueError(f"File {path} does not look like a prime_projections_N{{N}}.json output.")
    return payload["prime_projections"]


# -----------------------------
# Diagram-level filters
# -----------------------------

BigonConstraint = Tuple[int, int, int]  # (vertex_u, vertex_v, required_xor)

def bigon_constraints_for_map(M: kx.FourRegularMap, mode: str) -> List[BigonConstraint]:
    """
    Precompute XOR constraints between crossing bits imposed by the bigon rule.

    We identify bigons as 2-cycles of the face permutation φ = σ∘α.
    For each bigon (i j) incident to two distinct vertices u!=v, we add a constraint
    between the crossing bits b(u), b(v).

    mode:
      - "off"         : no constraints
      - "naive_equal" : require b(u)=b(v)  (xor=0)
      - "parity"      : require b(u) xor b(v) = parity(i) xor parity(j),
                        where parity is the position parity inside the σ-cycle.
                        (This is the refinement used in the parity implementation.)
    """
    if mode == "off":
        return []

    if mode not in ("naive_equal", "parity"):
        raise ValueError("bigon_rule must be one of: off, naive_equal, parity")

    sigma = M.sigma
    alpha = M.alpha
    n = M.n

    # dart -> vertex index, and dart -> parity (position mod 2 inside the σ-cycle)
    d2v = [0] * (n + 1)
    d2par = [0] * (n + 1)
    for vi, v in enumerate(M.vertices):
        for pos, d in enumerate(v.darts):
            d2v[d] = vi
            d2par[d] = pos & 1

    # φ = σ ∘ α  (composition convention matches kx module)
    phi = [0] * (n + 1)
    for d in range(1, n + 1):
        phi[d] = alpha[sigma[d]]

    seen = [False] * (n + 1)
    constraints: List[BigonConstraint] = []

    for i in range(1, n + 1):
        if seen[i]:
            continue
        j = phi[i]
        if j != i and phi[j] == i:
            seen[i] = True
            seen[j] = True
            u = d2v[i]
            v = d2v[j]
            if u != v:
                rhs = 0 if mode == "naive_equal" else (d2par[i] ^ d2par[j])
                constraints.append((u, v, rhs))

    return constraints


def passes_bigon_constraints(bits_int: int, constraints: List[BigonConstraint]) -> bool:
    """
    Check all XOR constraints b(u) xor b(v) == rhs on an integer-encoded bitstring.
    """
    for u, v, rhs in constraints:
        if (((bits_int >> u) ^ (bits_int >> v)) & 1) != rhs:
            return False
    return True


@dataclass(frozen=True)
class LinkParticipationData:
    """
    Precomputed data for the over/under participation filter on a fixed projection.

    - comp_count: number of straight-ahead components
    - mixed_vertices: indices of mixed crossings (two distinct components meet)
    - comp_even[vi]: component id of darts in positions 0&2 at vertex vi
    - comp_odd[vi] : component id of darts in positions 1&3 at vertex vi
    """
    comp_count: int
    mixed_vertices: Tuple[int, ...]
    comp_even: Tuple[int, ...]
    comp_odd: Tuple[int, ...]


def link_participation_data_from_passport(
    M: kx.FourRegularMap, passport: dict
) -> Optional[LinkParticipationData]:
    """
    Build link participation precompute data from the JSON passport if possible.

    The projection JSON includes 'n_components' and 'components_orbits' (a partition
    of darts by straight-ahead component). Using these avoids recomputing components.
    """
    comp_count = int(passport.get("n_components", 0) or 0)
    if comp_count <= 1:
        return None

    components_orbits = passport.get("components_orbits", None)
    if not isinstance(components_orbits, list) or len(components_orbits) != comp_count:
        # Fallback: if the JSON doesn't have the data, we can rebuild component ids
        # from the permutation model.
        return link_participation_data_from_map(M)

    n = M.n
    comp_id = [-1] * (n + 1)
    for cid, orbit in enumerate(components_orbits):
        for d in orbit:
            comp_id[int(d)] = cid
    if any(comp_id[d] < 0 for d in range(1, n + 1)):
        return link_participation_data_from_map(M)

    comp_even: List[int] = []
    comp_odd: List[int] = []
    mixed: List[int] = []
    for vi, v in enumerate(M.vertices):
        d0, d1, d2, d3 = v.darts
        c0 = comp_id[d0]
        c1 = comp_id[d1]
        comp_even.append(c0)
        comp_odd.append(c1)
        if c0 != c1:
            mixed.append(vi)

    return LinkParticipationData(
        comp_count=comp_count,
        mixed_vertices=tuple(mixed),
        comp_even=tuple(comp_even),
        comp_odd=tuple(comp_odd),
    )


def link_participation_data_from_map(M: kx.FourRegularMap) -> Optional[LinkParticipationData]:
    """
    Fallback construction of participation data directly from permutations.

    Straight-ahead components are orbits of the group generated by:
      - α (edge flip)
      - σ^2 (opposite darts in a crossing)

    This is computed once per projection (not per diagram).
    """
    # For knots, filter is irrelevant
    comp_count = len(M.component_pairs)
    if comp_count <= 1:
        return None

    n = M.n
    alpha = M.alpha
    sigma2 = M.sigma2

    comp_id = [-1] * (n + 1)
    cid = 0
    for d0 in range(1, n + 1):
        if comp_id[d0] >= 0:
            continue
        stack = [d0]
        comp_id[d0] = cid
        while stack:
            d = stack.pop()
            for nd in (alpha[d], sigma2[d]):
                if comp_id[nd] < 0:
                    comp_id[nd] = cid
                    stack.append(nd)
        cid += 1

    comp_even: List[int] = []
    comp_odd: List[int] = []
    mixed: List[int] = []
    for vi, v in enumerate(M.vertices):
        d0, d1, d2, d3 = v.darts
        c0 = comp_id[d0]
        c1 = comp_id[d1]
        comp_even.append(c0)
        comp_odd.append(c1)
        if c0 != c1:
            mixed.append(vi)

    return LinkParticipationData(
        comp_count=cid,
        mixed_vertices=tuple(mixed),
        comp_even=tuple(comp_even),
        comp_odd=tuple(comp_odd),
    )


def passes_link_over_under_participation(bits_int: int, data: LinkParticipationData) -> bool:
    """
    Check the participation condition using bitmasks.

    We only consider mixed vertices. At a mixed vertex vi:
      - if b(vi)=0 then over is comp_even[vi], under is comp_odd[vi]
      - if b(vi)=1 then over is comp_odd[vi],  under is comp_even[vi]
    """
    full = (1 << data.comp_count) - 1
    over_mask = 0
    under_mask = 0

    for vi in data.mixed_vertices:
        bit = (bits_int >> vi) & 1
        if bit == 0:
            over = data.comp_even[vi]
            under = data.comp_odd[vi]
        else:
            over = data.comp_odd[vi]
            under = data.comp_even[vi]
        over_mask |= 1 << over
        under_mask |= 1 << under

        # Early exit if we've already seen all components in both roles
        if over_mask == full and under_mask == full:
            return True

    return over_mask == full and under_mask == full


# -----------------------------
# Fast bracket evaluation
# -----------------------------

@dataclass
class GlobalStateSumTables:
    """
    Global (per N) tables reused across projections.

    - exp_table_by_b[b][t] = N - 2*popcount(b xor t)
      (weights of a for each diagram bitstring b and smoothing-type pattern t)

    - shifted_loop_pows[gamma][exp] = tuple((a_exp, coeff), ...)
      representing  a^exp * (-a^2 - a^{-2})^gamma
      as exponent/coeff pairs in a.
    """
    N: int
    exp_table_by_b: List[List[int]]
    shifted_loop_pows: List[Dict[int, Tuple[Tuple[int, int], ...]]]


def build_global_tables(N: int, max_gamma: int) -> GlobalStateSumTables:
    """
    Precompute all tables that depend only on N and the loop factor.

    This is lightweight for N<=8.
    """
    # exp_table_by_b
    size = 1 << N
    exp_for_popcount = [N - 2 * k for k in range(N + 1)]
    exp_table_by_b: List[List[int]] = []
    for b in range(size):
        row = [0] * size
        for t in range(size):
            row[t] = exp_for_popcount[(b ^ t).bit_count()]
        exp_table_by_b.append(row)

    # loop factor powers
    loop_factor = {2: -1, -2: -1}  # -a^2 - a^{-2}
    loop_pows: List[Laurent] = [kx.laurent_pow(loop_factor, g) for g in range(max_gamma + 1)]

    shifted_loop_pows: List[Dict[int, Tuple[Tuple[int, int], ...]]] = []
    all_exps = set(exp_for_popcount)
    for g in range(max_gamma + 1):
        table_g: Dict[int, Tuple[Tuple[int, int], ...]] = {}
        base = loop_pows[g]
        for exp in all_exps:
            shifted = tuple(sorted((e + exp, c) for e, c in base.items() if c != 0))
            table_g[exp] = shifted
        shifted_loop_pows.append(table_g)

    return GlobalStateSumTables(N=N, exp_table_by_b=exp_table_by_b, shifted_loop_pows=shifted_loop_pows)


def precompute_state_geometry(M: kx.FourRegularMap) -> Tuple[List[int], List[int], int]:
    """
    For each t in {0,1}^N (t is an integer 0..2^N-1), build the smoothing involution
    τ_t and compute:
      gamma(t) = #contractible loops,
      delta(t) = #essential loops.

    This is done ONCE per projection and then reused for all crossing assignments b.
    """
    N = M.N
    n = M.n
    verts = [v.darts for v in M.vertices]

    gamma_by_t = [0] * (1 << N)
    delta_by_t = [0] * (1 << N)

    for t in range(1 << N):
        tau = [0] * (n + 1)
        for vi, (d0, d1, d2, d3) in enumerate(verts):
            if (t >> vi) & 1 == 0:
                # type 0 pairing: (d0 d1)(d2 d3)
                tau[d0] = d1
                tau[d1] = d0
                tau[d2] = d3
                tau[d3] = d2
            else:
                # type 1 pairing: (d1 d2)(d3 d0)
                tau[d1] = d2
                tau[d2] = d1
                tau[d3] = d0
                tau[d0] = d3
        g, d = M._physical_loops_gamma_delta(tau)
        gamma_by_t[t] = g
        delta_by_t[t] = d

    return gamma_by_t, delta_by_t, max(gamma_by_t) if gamma_by_t else 0


def fast_generalized_bracket(
    bits_int: int,
    gamma_by_t: List[int],
    delta_by_t: List[int],
    tables: GlobalStateSumTables,
) -> PolyX:
    """
    Evaluate <D>(a,x) using precomputed (gamma(t),delta(t)) for t=b xor s.
    """
    N = tables.N
    size = 1 << N
    exp_row = tables.exp_table_by_b[bits_int]

    poly: PolyX = {}
    # Main accumulation loop over t in {0,1}^N
    for t in range(size):
        d = delta_by_t[t]
        g = gamma_by_t[t]
        exp = exp_row[t]
        term_pairs = tables.shifted_loop_pows[g][exp]

        la = poly.get(d)
        if la is None:
            la = {}
            poly[d] = la
        for ae, c in term_pairs:
            la[ae] = la.get(ae, 0) + c

    # Clean zeros
    out: PolyX = {}
    for m, la in poly.items():
        la2 = {e: c for e, c in la.items() if c != 0}
        if la2:
            out[m] = la2
    return out


def normalize_by_writhe(poly: PolyX, w: int) -> PolyX:
    """
    Multiply poly by (-a)^(-3w) = (-1)^w * a^(-3w).
    """
    sign = -1 if (w & 1) else 1
    shift = -3 * w

    out: PolyX = {}
    for m, la in poly.items():
        la2 = {}
        for e, c in la.items():
            cc = sign * c
            if cc != 0:
                la2[e + shift] = la2.get(e + shift, 0) + cc
        if la2:
            out[m] = la2
    return out


# -----------------------------
# Enumeration / output
# -----------------------------

@dataclass
class LayerStats:
    N: int
    projections: int = 0
    diagrams_tested: int = 0
    diagrams_kept: int = 0
    within_knots: int = 0
    within_links: int = 0
    new_knots: int = 0
    new_links: int = 0


def bitstring(bits_int: int, N: int) -> str:
    return "".join("1" if (bits_int >> i) & 1 else "0" for i in range(N))


def stable_class_ids(prefix: str, keys: Sequence[Tuple]) -> Dict[Tuple, str]:
    """
    Assign deterministic IDs K0001, K0002, ... by sorting keys.
    """
    sorted_keys = sorted(keys)
    width = max(4, len(str(len(sorted_keys))))
    return {k: f"{prefix}{i+1:0{width}d}" for i, k in enumerate(sorted_keys)}


def write_new_classes_csv(
    path: Path,
    prefix: str,
    new_entries: Dict[Tuple, SeenEntry],
) -> None:
    ids = stable_class_ids(prefix, list(new_entries.keys()))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "first_N", "proj_id", "bits", "count_key", "raw_poly_key", "poly_latex", "skeleton_key"])
        for key in sorted(new_entries.keys()):
            e = new_entries[key]
            w.writerow([
                ids[key],
                e.first_N,
                e.proj_id,
                e.bits,
                repr(key),
                repr(e.raw_poly_key),
                e.poly_latex,
                repr(e.skeleton_key) if e.skeleton_key is not None else "",
            ])


def write_summary(path: Path, cfg: DiagramEnumConfig, stats: LayerStats, elapsed_s: float, lib: Optional[LibraryDB]) -> None:
    lines: List[str] = []
    lines.append("torus_g1_diagrams_from_prime_projections.py")
    lines.append("")
    lines.append("Settings")
    lines.append("--------")
    lines.append(f"N = {cfg.N}")
    lines.append(f"proj_dir = {cfg.proj_dir}")
    lines.append(f"out_dir  = {cfg.out_dir}")
    lines.append(f"cache_dir= {cfg.cache_dir}")
    lines.append(f"new_at_N = {cfg.new_at_N}   (N_min={cfg.N_min})")
    lines.append(f"quotient_global_switch = {cfg.quotient_global_switch}")
    lines.append(f"bigon_rule = {cfg.bigon_rule}")
    lines.append(f"link_over_under_participation = {cfg.link_over_under_participation}")
    lines.append(f"link_invariant = {cfg.link_invariant}")
    lines.append(f"poly_canon = {cfg.poly_canon}")
    lines.append(f"count_key = {cfg.count_key}")
    lines.append(f"sign_convention = {cfg.sign_convention}")
    lines.append("")
    lines.append("Counts")
    lines.append("------")
    lines.append(f"projections processed      : {stats.projections}")
    lines.append(f"diagrams tested            : {stats.diagrams_tested}")
    lines.append(f"diagrams kept (after filters): {stats.diagrams_kept}")
    lines.append(f"distinct knot keys at N    : {stats.within_knots}")
    lines.append(f"distinct link keys at N    : {stats.within_links}")
    lines.append(f"NEW knot keys at N         : {stats.new_knots}")
    lines.append(f"NEW link keys at N         : {stats.new_links}")
    lines.append("")
    if lib is not None:
        lines.append("Library")
        lines.append("-------")
        lines.append(f"fingerprint = {lib.fingerprint}")
        lines.append(f"max_N = {lib.max_N}")
        lines.append(f"known knot classes = {len(lib.knots)}")
        lines.append(f"known link classes = {len(lib.links)}")
        lines.append("")
    lines.append(f"Elapsed: {elapsed_s:.2f} s")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def library_path(cfg: DiagramEnumConfig) -> Path:
    return cfg.cache_dir / f"diagram_library_{cfg.fingerprint()}.pkl"


def load_library(cfg: DiagramEnumConfig) -> Optional[LibraryDB]:
    path = library_path(cfg)
    if not path.exists():
        return None
    with path.open("rb") as f:
        lib = pickle.load(f)
    if not isinstance(lib, LibraryDB) or lib.fingerprint != cfg.fingerprint():
        return None
    return lib


def save_library(cfg: DiagramEnumConfig, lib: LibraryDB) -> None:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    path = library_path(cfg)
    with path.open("wb") as f:
        pickle.dump(lib, f)


def seed_library_if_requested(cfg: DiagramEnumConfig, lib: LibraryDB) -> None:
    """
    Optional seeds for new-at-N mode.

    These are two small knot types of complexity 0 and 1 that can otherwise
    appear as "new" at higher N if one is only using polynomial keys.

    The exact seed polynomials are hard-coded (they come from the invariant
    definitions themselves, not from projection enumeration).
    """
    if not (cfg.new_at_N and cfg.seed_trivial_knots and cfg.count_key == "poly"):
        return

    # Seed 0-crossing knot: X = x
    poly0: PolyX = {1: {0: 1}}
    key0 = canonical_poly_key_tabulation(poly0) if cfg.poly_canon == "tabulation" else poly_key(poly0)
    if key0 not in lib.knots:
        lib.knots[key0] = SeenEntry(
            first_N=0,
            proj_id=0,
            bits="",
            raw_poly_key=poly_key(poly0),
            poly_latex=kx.polyX_to_latex(poly0),
            skeleton_key=None,
        )

    # Seed 1-crossing knot: X = a^2 + a^6 - a^2 x^2  (as in the earlier script)
    poly1: PolyX = {0: {2: 1, 6: 1}, 2: {2: -1}}
    key1 = canonical_poly_key_tabulation(poly1) if cfg.poly_canon == "tabulation" else poly_key(poly1)
    if key1 not in lib.knots:
        lib.knots[key1] = SeenEntry(
            first_N=1,
            proj_id=0,
            bits="",
            raw_poly_key=poly_key(poly1),
            poly_latex=kx.polyX_to_latex(poly1),
            skeleton_key=None,
        )


def ensure_library_upto(cfg: DiagramEnumConfig, upto_N: int) -> LibraryDB:
    """
    Ensure we have a library built up to upto_N (inclusive), in new-at-N mode.
    """
    lib = load_library(cfg)
    if lib is None:
        lib = LibraryDB(fingerprint=cfg.fingerprint(), max_N=-1)
        seed_library_if_requested(cfg, lib)

    if lib.max_N >= upto_N:
        return lib

    # Build missing layers
    for N in range(max(cfg.N_min, lib.max_N + 1), upto_N + 1):
        layer_cfg = DiagramEnumConfig(**{**cfg.__dict__, "N": N, "new_at_N": True})
        _stats, _new_knots, _new_links = enumerate_layer_and_update_library(layer_cfg, lib, write_outputs=False)
        lib.max_N = N

    return lib


def enumerate_layer_and_update_library(
    cfg: DiagramEnumConfig,
    lib: LibraryDB,
    write_outputs: bool = True,
) -> Tuple[LayerStats, Dict[Tuple, SeenEntry], Dict[Tuple, SeenEntry]]:
    """
    Enumerate diagrams at a fixed N, update the library, and optionally write outputs.
    """
    t0 = time.time()
    prime_list = load_prime_projections(cfg.proj_dir, cfg.N)

    # Precompute bit tuples once for this N for use by kx.writhe if needed.
    bit_tuples: List[Tuple[int, ...]] = []
    for b in range(1 << cfg.N):
        bit_tuples.append(tuple((b >> i) & 1 for i in range(cfg.N)))

    # Global tables for fast bracket: built once per layer, but require a max_gamma.
    # We'll initialize lazily after we see the first projection.
    tables: Optional[GlobalStateSumTables] = None
    tables_max_gamma = -1

    stats = LayerStats(N=cfg.N)
    new_knots: Dict[Tuple, SeenEntry] = {}
    new_links: Dict[Tuple, SeenEntry] = {}
    within_knots_set: set = set()
    within_links_set: set = set()

    # Determine iteration range over crossing assignments.
    # If we quotient by global switch, fix bit 0 to 0 (LSB) => iterate even integers.
    if cfg.quotient_global_switch:
        bits_iter = range(0, 1 << cfg.N, 2)
    else:
        bits_iter = range(0, 1 << cfg.N)

    for proj_idx, passport in enumerate(prime_list, start=1):
        sigma = [0] + list(passport["sigma_images"])
        alpha = [0] + list(passport["alpha_images"])
        M = kx.FourRegularMap(sigma, alpha)

        stats.projections += 1

        # Local diagram-level filters
        bigon_constraints = bigon_constraints_for_map(M, cfg.bigon_rule)
        part_data = None
        if cfg.link_over_under_participation:
            part_data = link_participation_data_from_passport(M, passport)

        # Precompute state geometry for fast bracket evaluation
        gamma_by_t, delta_by_t, max_gamma = precompute_state_geometry(M)

        if tables is None or max_gamma > tables_max_gamma:
            tables = build_global_tables(cfg.N, max_gamma)
            tables_max_gamma = max_gamma
        assert tables is not None

        comp_cnt = len(M.component_pairs)
        is_knot_projection = (comp_cnt == 1)

        # Enumerate crossing assignments
        for bits_int in bits_iter:
            stats.diagrams_tested += 1

            if bigon_constraints and not passes_bigon_constraints(bits_int, bigon_constraints):
                continue

            if part_data is not None:
                # only relevant for links (comp_cnt>=2), but data is None for knots anyway
                if not passes_link_over_under_participation(bits_int, part_data):
                    continue

            stats.diagrams_kept += 1

            # Compute invariant polynomial
            bracket = fast_generalized_bracket(bits_int, gamma_by_t, delta_by_t, tables)

            if is_knot_projection:
                # For knots we normalize by writhe to obtain X_D(a,x).
                w = M.writhe(bit_tuples[bits_int], component_orientations=None, sign_convention=cfg.sign_convention)
                poly = normalize_by_writhe(bracket, w)
            else:
                if cfg.link_invariant == "bracket":
                    poly = bracket
                else:
                    # Orientation-dependent link X; uses a default orientation choice.
                    w = M.writhe(bit_tuples[bits_int], component_orientations=None, sign_convention=cfg.sign_convention)
                    poly = normalize_by_writhe(bracket, w)

            # Compute key(s)
            count_k, raw_k, sk_k = compute_count_key(M, poly, cfg)

            # Track within-layer distinct keys
            if is_knot_projection:
                within_knots_set.add(count_k)
            else:
                within_links_set.add(count_k)

            # New-at-N logic: classes whose *minimal* crossing number equals N.
            if cfg.new_at_N:
                existing = (lib.knots.get(count_k) if is_knot_projection else lib.links.get(count_k))
                seen_before = existing is not None and existing.first_N < cfg.N
                if not seen_before:
                    # Ensure the library stores the minimal N for this key (robust even if run out of order).
                    if existing is None or existing.first_N > cfg.N:
                        poly_latex = kx.polyX_to_latex(poly)
                        existing = SeenEntry(
                            first_N=cfg.N,
                            proj_id=proj_idx,
                            bits=bitstring(bits_int, cfg.N),
                            raw_poly_key=raw_k,
                            poly_latex=poly_latex,
                            skeleton_key=sk_k,
                        )
                        lib.add(is_knot_projection, count_k, existing)
                    # Record the representative for this N (even if it was already known with first_N == N).
                    if is_knot_projection:
                        new_knots[count_k] = existing
                    else:
                        new_links[count_k] = existing
    stats.within_knots = len(within_knots_set)
    stats.within_links = len(within_links_set)
    stats.new_knots = len(new_knots)
    stats.new_links = len(new_links)

    elapsed = time.time() - t0

    if write_outputs:
        cfg.out_dir.mkdir(parents=True, exist_ok=True)

        stem = f"torus_{cfg.tag()}_{cfg.fingerprint()}"
        summary_path = cfg.out_dir / f"{stem}_summary.txt"
        write_summary(summary_path, cfg, stats, elapsed_s=elapsed, lib=lib)

        if cfg.new_at_N:
            knots_csv = cfg.out_dir / f"{stem}_new_knots_classes.csv"
            links_csv = cfg.out_dir / f"{stem}_new_links_classes.csv"
            write_new_classes_csv(knots_csv, "K", new_knots)
            write_new_classes_csv(links_csv, "L", new_links)

    return stats, new_knots, new_links


# -----------------------------
# Self-test (optional)
# -----------------------------

def self_test(cfg: DiagramEnumConfig, samples: int = 20) -> None:
    """
    Quick consistency test: compare fast_generalized_bracket() with kx.generalized_bracket()
    on a few random diagrams from the input projection file for cfg.N.
    """
    import random

    prime_list = load_prime_projections(cfg.proj_dir, cfg.N)
    if not prime_list:
        raise SystemExit("Self-test: no projections found.")

    # Choose a few projections
    proj_samples = random.sample(prime_list, min(len(prime_list), 3))
    print(f"[self-test] N={cfg.N}, testing up to {samples} random diagrams across {len(proj_samples)} projections")

    for passport in proj_samples:
        sigma = [0] + list(passport["sigma_images"])
        alpha = [0] + list(passport["alpha_images"])
        M = kx.FourRegularMap(sigma, alpha)

        gamma_by_t, delta_by_t, max_gamma = precompute_state_geometry(M)
        tables = build_global_tables(cfg.N, max_gamma)

        for _ in range(samples):
            bits_int = random.randrange(0, 1 << cfg.N)
            bits = tuple((bits_int >> i) & 1 for i in range(cfg.N))
            p_fast = fast_generalized_bracket(bits_int, gamma_by_t, delta_by_t, tables)
            p_ref = M.generalized_bracket(bits)
            if poly_key(p_fast) != poly_key(p_ref):
                raise SystemExit(
                    "Self-test FAILED: fast bracket does not match reference.\n"
                    f"bits_int={bits_int}\n"
                    f"fast={poly_key(p_fast)}\n"
                    f"ref ={poly_key(p_ref)}"
                )
    print("[self-test] OK")


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> DiagramEnumConfig:
    p = argparse.ArgumentParser(
        description="Enumerate genus-one (torus) diagrams from precomputed prime projection JSON files."
    )
    p.add_argument("--N", type=int, required=True, help="Crossing number (number of vertices) to process.")
    p.add_argument("--proj-dir", type=Path, default=Path("out"), help="Directory with prime_projections_N{N}.json files.")
    p.add_argument("--out", dest="out_dir", type=Path, default=Path("out"), help="Output directory for summaries/CSVs.")
    p.add_argument("--cache", dest="cache_dir", type=Path, default=Path("cache"), help="Cache directory for the library pickle.")

    p.add_argument("--no-global-switch", dest="quotient_global_switch", action="store_false",
                   help="Do NOT quotient crossing assignments by b~1-b (global switch).")
    p.add_argument("--bigon-rule", choices=["off", "naive_equal", "parity"], default="parity",
                   help="Bigon rule mode (default: parity).")
    p.add_argument("--no-link-ou", dest="link_over_under_participation", action="store_false",
                   help="Disable the link over/under participation filter.")

    p.add_argument("--link-invariant", choices=["bracket", "X"], default="bracket",
                   help="Invariant used for links (default: bracket).")
    p.add_argument("--poly-canon", choices=["raw", "tabulation"], default="tabulation",
                   help="Polynomial canonicalization mode (default: tabulation).")
    p.add_argument("--count-key", choices=["poly", "skeleton"], default="poly",
                   help="Classification key (default: poly).")
    p.add_argument("--sign-convention", type=int, choices=[-1, 1], default=1,
                   help="Writhe sign convention (+1 or -1).")

    # We classify by minimal crossing number ("new at N"). There is no "all-at-N" mode.

    p.add_argument("--N-min", type=int, default=2,
                   help="Minimal N used when building the incremental library (default: 2).")
    p.add_argument("--no-seed-trivial-knots", dest="seed_trivial_knots", action="store_false",
                   help="Do not seed the library with the standard N=0 and N=1 knot types.")

    p.add_argument("--self-test", action="store_true",
                   help="Run a quick self-test comparing the fast bracket to the reference implementation, then exit.")

    args = p.parse_args(argv)

    # Resolve paths relative to the repository root (so the script works from any CWD).
    proj_dir = (REPO_ROOT / args.proj_dir).resolve() if not args.proj_dir.is_absolute() else args.proj_dir.resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir.resolve()
    cache_dir = (REPO_ROOT / args.cache_dir).resolve() if not args.cache_dir.is_absolute() else args.cache_dir.resolve()

    cfg = DiagramEnumConfig(
        N=args.N,
        proj_dir=proj_dir,
        out_dir=out_dir,
        cache_dir=cache_dir,
        quotient_global_switch=args.quotient_global_switch,
        bigon_rule=args.bigon_rule,
        link_over_under_participation=args.link_over_under_participation,
        link_invariant=args.link_invariant,
        poly_canon=args.poly_canon,
        count_key=args.count_key,
        sign_convention=args.sign_convention,
        new_at_N=True,
        N_min=args.N_min,
        seed_trivial_knots=args.seed_trivial_knots,
    )
    cfg.validate()

    if args.self_test:
        self_test(cfg)
        raise SystemExit(0)

    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = parse_args(argv)

    # Build/extend the "first occurrence" library up to N-1, then enumerate diagrams at N.
    lib = ensure_library_upto(cfg, cfg.N - 1)
    stats, _, _ = enumerate_layer_and_update_library(cfg, lib, write_outputs=True)
    lib.max_N = max(lib.max_N, cfg.N)
    save_library(cfg, lib)
    print(f"[ok] wrote outputs to: {cfg.out_dir}")
    print(f"[ok] updated library: {library_path(cfg)} (max_N={lib.max_N})")
    print(f"[ok] new knots at N={cfg.N}: {stats.new_knots}")
    print(f"[ok] new links at N={cfg.N}: {stats.new_links}")

if __name__ == "__main__":  # pragma: no cover
    main()
