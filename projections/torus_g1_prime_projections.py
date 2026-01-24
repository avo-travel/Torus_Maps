#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""torus_g1_prime_projections.py

Reproduce Table 1 (genus-one / torus projections) and export *prime* unsensed projections.

This is a standalone, dependency-free Python script intended for public release (e.g. GitHub).
It implements the *projection-level* part of the pipeline described in the paper:

  - A projection on the torus is a connected 4-regular map (a cellularly embedded
    4-valent graph) *without* over/under information.
  - We encode a map by permutations (σ, α) on the dart set H = {1, …, 4N}:
      σ : vertex rotation (product of N disjoint 4-cycles),
      α : fixed-point-free involution pairing darts into edges,
      φ = σ∘α : face permutation.

The script:
  1) Enumerates labelled candidate encodings (α, σ0) with σ0 fixed in standard form,
     and applies the standing projection-level constraints used in the paper:
       - connectedness,
       - genus-one (torus) condition via the face permutation,
       - no monogons (no fixed points of φ).
     By default (as in the benchmark tables), edge-loops are forbidden and bigons are allowed.
  2) Deduplicates outputs up to **unsensed** equivalence (orientation-preserving and
     orientation-reversing homeomorphisms of (T^2, G)) by replacing each encoding with a
     deterministic canonical representative (see notes below).
  3) Applies the projection-level *primeness* witnesses used in the paper:
       - remove 2-edge-cut composite projections;
       - for links: also remove split-witnessed projections.
  4) Prints a Table-1-ready row and writes all *prime* unsensed projections (with passports)
     to machine-readable JSON and a human-readable TXT.

------------------------------------------------------------------------
Quick start

  # N=4 (should reproduce the Table 1 line: 28, 5, 0, 23, 10, 13)
  python torus_g1_prime_projections.py --N 4

  # N=8 (same defaults; output goes to ./out by default)
  python torus_g1_prime_projections.py --N 8

Outputs (per N) are written into the directory given by --out (default: ./out):
  - table1_row_N{N}.tex          : one LaTeX row to paste into Table 1
  - prime_projections_N{N}.json  : prime projections + summary counts
  - prime_projections_N{N}.txt   : readable passports (σ/α/φ cycle forms, components, ...)

------------------------------------------------------------------------
Notes on "unsensed" canonicalization / deduplication

In the paper, unsensed equivalence of labelled encodings is simultaneous conjugacy,
with the additional option σ ↦ σ^{-1} (orientation reversal).

For practical computation we realize canonical representatives via a rooted, deterministic
normal form of the dart graph:
  - for each root dart r and for σ and σ^{-1}, we compute a rooted normal form by a
    deterministic traversal (neighbor order: σ first, then α);
  - we take the lexicographically smallest rooted normal form across all roots and both
    orientations.

This yields a reproducible canonical representative that is constant on unsensed
equivalence classes, and it reproduces the benchmark counts reported in the paper.
If desired, this canonicalization layer can be replaced by an external graph canonizer
(e.g. nauty/bliss) without changing the rest of the pipeline.

------------------------------------------------------------------------
"""


from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ============================================================
# Basic permutation utilities (1-indexed with a dummy 0 entry)
# ============================================================


def make_sigma_4_regular(N: int) -> List[int]:
    """Return σ0 = (1 2 3 4)(5 6 7 8)... on 4N darts."""
    n = 4 * N
    sigma = [0] * (n + 1)
    for v in range(N):
        a = 4 * v + 1
        sigma[a] = a + 1
        sigma[a + 1] = a + 2
        sigma[a + 2] = a + 3
        sigma[a + 3] = a
    return sigma


def compose(p: Sequence[int], q: Sequence[int]) -> List[int]:
    """Composition p ∘ q (apply q first, then p), for 1-indexed permutations."""
    n = len(p) - 1
    r = [0] * (n + 1)
    for i in range(1, n + 1):
        r[i] = p[q[i]]
    return r


def invert(p: Sequence[int]) -> List[int]:
    """Inverse permutation (1-indexed)."""
    n = len(p) - 1
    inv = [0] * (n + 1)
    for i in range(1, n + 1):
        inv[p[i]] = i
    return inv


def count_cycles(p: Sequence[int]) -> int:
    """Number of cycles of a 1-indexed permutation p."""
    n = len(p) - 1
    seen = [False] * (n + 1)
    cnt = 0
    for i in range(1, n + 1):
        if not seen[i]:
            cnt += 1
            j = i
            while not seen[j]:
                seen[j] = True
                j = p[j]
    return cnt


def cycles_of_perm(p: Sequence[int]) -> List[List[int]]:
    """Cycle decomposition; each cycle rotated to start with its minimum; cycles sorted."""
    n = len(p) - 1
    seen = [False] * (n + 1)
    cycles: List[List[int]] = []
    for i in range(1, n + 1):
        if not seen[i]:
            cur: List[int] = []
            j = i
            while not seen[j]:
                seen[j] = True
                cur.append(j)
                j = p[j]
            m = min(cur)
            k = cur.index(m)
            cur = cur[k:] + cur[:k]
            cycles.append(cur)
    cycles.sort(key=lambda c: c[0])
    return cycles


def cycles_to_str(cycles: Sequence[Sequence[int]]) -> str:
    """Pretty-print a list of cycles."""
    return "".join("(" + " ".join(str(x) for x in cyc) + ")" for cyc in cycles)


# ============================================================
# Connectivity and deterministic canonical labelling
# ============================================================


def is_connected(sigma: Sequence[int], alpha: Sequence[int]) -> bool:
    """Connectedness <=> transitivity of <σ, α> on darts."""
    n = len(sigma) - 1
    seen = [False] * (n + 1)
    stack = [1]
    seen[1] = True
    while stack:
        u = stack.pop()
        for v in (sigma[u], alpha[u]):
            if not seen[v]:
                seen[v] = True
                stack.append(v)
    return all(seen[1:])


def canonicalize(sigma: Sequence[int], alpha: Sequence[int], root: int) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Deterministic DFS canonical relabelling (rooted).

    Starting at `root`, traverse the dart graph using the fixed neighbor order:
      first σ, then α.

    The traversal order induces a relabelling old↦new in {1..n}. We return the relabelled
    permutations as tuples `(sigma_canon, alpha_canon)` (images of 1..n).

    Returns None iff the traversal does not reach all darts (i.e. the map is disconnected).
    """
    n = len(sigma) - 1
    seen = [False] * (n + 1)
    order: List[int] = []

    # iterative stack DFS to avoid recursion overhead
    stack = [root]
    while stack:
        u = stack.pop()
        if seen[u]:
            continue
        seen[u] = True
        order.append(u)
        # push in reverse so that σ-neighbor is processed first
        # (stack is LIFO): we want visit σ(u) then α(u).
        stack.append(alpha[u])
        stack.append(sigma[u])

    if len(order) != n:
        return None

    relabel = {old: new for new, old in enumerate(order, start=1)}
    sigma_new = [0] * (n + 1)
    alpha_new = [0] * (n + 1)
    for old in range(1, n + 1):
        new = relabel[old]
        sigma_new[new] = relabel[sigma[old]]
        alpha_new[new] = relabel[alpha[old]]

    return (tuple(sigma_new[1:]), tuple(alpha_new[1:]))


# ============================================================
# Projection-level invariants and witnesses
# ============================================================


def has_monogon(sigma: Sequence[int], alpha: Sequence[int]) -> bool:
    """Monogon face <=> a fixed point of φ=σ∘α."""
    phi = compose(sigma, alpha)
    for i in range(1, len(phi)):
        if phi[i] == i:
            return True
    return False


def has_bigon(sigma: Sequence[int], alpha: Sequence[int]) -> bool:
    """Bigon face <=> a 2-cycle of φ=σ∘α."""
    phi = compose(sigma, alpha)
    n = len(phi) - 1
    for i in range(1, n + 1):
        j = phi[i]
        if j != i and phi[j] == i:
            return True
    return False


def dart_to_vertex_map(sigma_t: Tuple[int, ...]) -> Dict[int, int]:
    """Map dart -> vertex_id (0..N-1), using σ-cycles ordered by their minimum."""
    sigma = [0] + list(sigma_t)
    vcycles = cycles_of_perm(sigma)
    d2v: Dict[int, int] = {}
    for vid, cyc in enumerate(vcycles):
        for d in cyc:
            d2v[d] = vid
    return d2v


def has_edge_loop(sigma_t: Tuple[int, ...], alpha_t: Tuple[int, ...]) -> bool:
    """Edge-loop: an α-pair whose darts lie in the same σ-cycle."""
    d2v = dart_to_vertex_map(sigma_t)
    alpha = [0] + list(alpha_t)
    n = len(alpha) - 1
    used: Set[int] = set()
    for i in range(1, n + 1):
        if i in used:
            continue
        j = alpha[i]
        used.add(i)
        used.add(j)
        if d2v[i] == d2v[j]:
            return True
    return False


def straight_ahead_components(sigma_t: Tuple[int, ...], alpha_t: Tuple[int, ...]) -> List[List[int]]:
    """Straight-ahead components = orbits of <α, σ^2> on darts."""
    sigma = [0] + list(sigma_t)
    alpha = [0] + list(alpha_t)
    delta = compose(sigma, sigma)  # σ^2

    n = len(sigma) - 1
    seen = [False] * (n + 1)
    comps: List[List[int]] = []

    for i in range(1, n + 1):
        if not seen[i]:
            stack = [i]
            seen[i] = True
            orb: List[int] = []
            while stack:
                u = stack.pop()
                orb.append(u)
                for v in (alpha[u], delta[u]):
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            orb.sort()
            comps.append(orb)

    comps.sort(key=lambda c: c[0])
    return comps


def splitness_for_links(sigma_t: Tuple[int, ...], alpha_t: Tuple[int, ...]) -> Tuple[bool, List[int]]:
    """Splitness witness (projection-level) used in the draft.

    Compute straight-ahead components and count, for each component K,
    how many *mixed* vertices it participates in.

    A vertex (σ-cycle) is mixed iff its four darts are not all in the same component.
    For 4-regular maps, this is equivalent to saying the two consecutive darts d0 and d1
    lie in different components.

    We call the projection split-witnessed iff some component has m(K) <= 1.

    Returns:
      (is_split_witnessed, mixed_counts_per_component)
    """
    comps = straight_ahead_components(sigma_t, alpha_t)
    k = len(comps)
    if k <= 1:
        return (False, [])

    comp_id: Dict[int, int] = {}
    for cid, orb in enumerate(comps):
        for d in orb:
            comp_id[d] = cid

    sigma = [0] + list(sigma_t)
    vcycles = cycles_of_perm(sigma)

    counts = [0] * k
    for cyc in vcycles:
        a = cyc[0]
        b = sigma[a]
        ca = comp_id[a]
        cb = comp_id[b]
        if ca != cb:
            counts[ca] += 1
            counts[cb] += 1

    is_split = any(c <= 1 for c in counts)
    return (is_split, counts)


def is_composite_projection_via_2edgecut(sigma_t: Tuple[int, ...], alpha_t: Tuple[int, ...]) -> Tuple[bool, Optional[Set[int]]]:
    """2-edge-cut composite witness.

    Let G(P) be the underlying abstract multigraph on the σ-cycles (vertices), with α-pairs
    as edges. We declare P composite if G(P) has an edge cut of size 2.

    Returns (is_composite, witness_vertex_set).
    The witness is a subset S of vertices (ids 0..N-1) with 0 in S such that the cut size is 2.
    """
    d2v = dart_to_vertex_map(sigma_t)
    N = len(set(d2v.values()))

    alpha = [0] + list(alpha_t)
    n = len(alpha) - 1

    # Build edge list once: each α-pair gives one (multi)edge.
    edges: List[Tuple[int, int]] = []
    used: Set[int] = set()
    for i in range(1, n + 1):
        if i in used:
            continue
        j = alpha[i]
        used.add(i)
        used.add(j)
        u = d2v[i]
        v = d2v[j]
        edges.append((u, v))

    # Enumerate vertex subsets S with 0 in S (to avoid double counting S|V\S).
    # For N<=8 this brute-force check is tiny: we scan masks on {0,..,N-1}.
    full = (1 << N) - 1
    for mask in range(1, full):
        if (mask & 1) == 0:
            continue          # enforce 0 in S
        if mask == full:
            continue          # skip the whole set

        cut = 0
        for (u, v) in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
                if cut > 2:
                    break

        if cut == 2:
            witness = {vv for vv in range(N) if ((mask >> vv) & 1)}
            return (True, witness)

    return (False, None)


# ============================================================
# Map generation: unsensed representatives only
# ============================================================


def generate_unsensed_candidate_projections(
    N: int,
    *,
    target_genus: Optional[int] = 1,
    forbid_edge_loops: bool = True,
    forbid_monogons: bool = True,
    forbid_bigons: bool = False,
    progress: bool = False,
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Enumerate *unsensed* candidate projections on a surface of given genus.

    We fix σ = σ0 (standard product of 4-cycles). We generate α as a perfect matching
    using a canonical construction path ("activated vertices" trick) to prune symmetry.

    Important: we only keep unsensed canonical representatives; we do NOT output rooted/sensed.

    The returned list is sorted deterministically.
    """

    n = 4 * N
    sigma = make_sigma_4_regular(N)
    sigma_inv = invert(sigma)

    # For construction-path pruning, we work in the initial σ0 labelling.
    # Vertex of a dart d is simply its 4-block index.
    def vertex_of_dart(d: int) -> int:
        return (d - 1) // 4 + 1  # 1..N

    darts_of_vertex = {v: [4 * v - 3, 4 * v - 2, 4 * v - 1, 4 * v] for v in range(1, N + 1)}

    # α will be built in-place.
    alpha = [0] * (n + 1)

    # Bitmask of unused darts: bit (d-1) is 1 iff dart d is unused.
    unused_mask0 = (1 << n) - 1

    # Bitmask of activated vertices: bit (v-1) is 1 iff vertex v has appeared.
    active_mask0 = 0

    unsensed_set: Set[Tuple[Tuple[int, ...], Tuple[int, ...]]] = set()

    # Simple counters for optional progress output.
    leaves_seen = 0
    leaves_kept = 0

    # Precompute torus/genre face-count target when possible.
    face_target: Optional[int] = None
    if target_genus is not None:
        # For 4-regular maps: V=N, E=2N, and chi = V-E+F = F-N.
        # For orientable genus g: chi = 2-2g, hence F = N + 2 - 2g.
        face_target = N + 2 - 2 * target_genus

    def backtrack(unused_mask: int, active_mask: int) -> None:
        nonlocal leaves_seen, leaves_kept

        if unused_mask == 0:
            leaves_seen += 1

            # --- full matching α built: apply *fast* projection-level filters first ---
            if not is_connected(sigma, alpha):
                return

            if face_target is not None:
                phi = compose(sigma, alpha)
                if count_cycles(phi) != face_target:
                    return
            else:
                phi = compose(sigma, alpha)

            if forbid_monogons:
                # (If forbid_edge_loops=True, monogons are automatically impossible,
                # but we keep the check for clarity and for the allow-edge-loops mode.)
                for i in range(1, n + 1):
                    if phi[i] == i:
                        return

            if forbid_bigons:
                # 2-cycle in φ
                for i in range(1, n + 1):
                    j = phi[i]
                    if j != i and phi[j] == i:
                        return

            # Edge-loops: if we forbade them during construction, this is redundant,
            # but keeping it is a cheap correctness safety net.
            if forbid_edge_loops:
                for i in range(1, n + 1):
                    j = alpha[i]
                    if i < j and vertex_of_dart(i) == vertex_of_dart(j):
                        return

            # --- compute unsensed canonical representative (σ or σ^{-1}, any root) ---
            best_u: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None
            best_key: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]] = None

            for cur_sigma in (sigma, sigma_inv):
                for r in range(1, n + 1):
                    canon = canonicalize(cur_sigma, alpha, root=r)
                    if canon is None:
                        continue
                    sigma_c, alpha_c = canon
                    key = (alpha_c, sigma_c)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_u = (sigma_c, alpha_c)

            assert best_u is not None
            unsensed_set.add(best_u)
            leaves_kept += 1

            if progress and leaves_seen % 50000 == 0:
                print(f"  explored full matchings: {leaves_seen:,}; kept candidates: {leaves_kept:,}; unique unsensed so far: {len(unsensed_set):,}")

            return

        # Pick the smallest unused dart i.
        lowbit = unused_mask & -unused_mask
        i = lowbit.bit_length()  # dart index (1..n)
        vi = vertex_of_dart(i)

        local_active = active_mask | (1 << (vi - 1))

        # Candidate darts j to pair with i.
        candidates: List[int] = []

        # (1) pair i with any unused dart on an already activated vertex
        for v in range(1, N + 1):
            if (local_active >> (v - 1)) & 1 == 0:
                continue
            for d in darts_of_vertex[v]:
                if d == i:
                    continue
                if (unused_mask >> (d - 1)) & 1 == 0:
                    continue

                # Early pruning:
                # - forbid edge-loops: never pair darts from the same vertex
                if forbid_edge_loops and v == vi:
                    continue
                # - forbid monogons: avoid α(i)=σ^{-1}(i)
                if forbid_monogons and d == sigma_inv[i]:
                    continue

                candidates.append(d)

        # (2) ...or with the smallest unused dart of the first not-yet-activated vertex
        for v in range(1, N + 1):
            if (local_active >> (v - 1)) & 1:
                continue
            # In this construction path, if v is not activated yet, all its darts are unused.
            # Still, we check unused_mask for robustness.
            dv = [d for d in darts_of_vertex[v] if (unused_mask >> (d - 1)) & 1]
            if dv:
                dmin = min(dv)
                if dmin != i:
                    # Edge-loop cannot happen here because v is not active and vi is active,
                    # hence v != vi.
                    candidates.append(dmin)
                break

        # Remove duplicates and iterate in deterministic order.
        for j in sorted(set(candidates)):
            alpha[i] = j
            alpha[j] = i

            next_unused = unused_mask & ~(1 << (i - 1)) & ~(1 << (j - 1))
            vj = vertex_of_dart(j)
            next_active = local_active | (1 << (vj - 1))

            backtrack(next_unused, next_active)

            alpha[i] = 0
            alpha[j] = 0

    backtrack(unused_mask0, active_mask0)

    # Deterministic order for output.
    unsensed = sorted(unsensed_set, key=lambda p: (p[1], p[0]))
    return unsensed


# ============================================================
# Passports + Table 1 statistics + export
# ============================================================


@dataclass
class ProjectionPassport:
    """A human- and machine-readable record for one canonical representative."""

    N: int
    kind: str
    prime: bool
    composite: bool
    split_witnessed: bool

    # Useful witnesses / extra data
    composite_witness_vertices: Optional[List[int]]
    split_mixed_counts: List[int]
    face_degrees: List[int]
    n_components: int
    components_orbits: List[List[int]]

    # The actual permutations (images of 1..4N) for the canonical representative.
    sigma_images: List[int]
    alpha_images: List[int]
    phi_images: List[int]

    # Pretty cycle notation (for quick visual inspection in papers/logs)
    sigma_cycles: str
    alpha_cycles: str
    phi_cycles: str


def passport(sigma_t: Tuple[int, ...], alpha_t: Tuple[int, ...]) -> ProjectionPassport:
    """Compute the projection passport used for printing/export."""

    N = len(sigma_t) // 4

    sigma = [0] + list(sigma_t)
    alpha = [0] + list(alpha_t)
    phi = compose(sigma, alpha)

    # Face degrees
    faces = cycles_of_perm(phi)
    face_degs = sorted(len(c) for c in faces)

    # Components
    comps = straight_ahead_components(sigma_t, alpha_t)
    ncomp = len(comps)

    # Splitness (links only)
    is_split, split_counts = splitness_for_links(sigma_t, alpha_t)

    # Composite witness
    is_comp, witness = is_composite_projection_via_2edgecut(sigma_t, alpha_t)

    if ncomp == 1:
        kind = "knot projection"
        is_prime = (not is_comp)
    else:
        kind = f"link projection ({ncomp} components)"
        is_prime = (not is_comp) and (not is_split)

    return ProjectionPassport(
        N=N,
        kind=kind,
        prime=is_prime,
        composite=is_comp,
        split_witnessed=is_split,
        composite_witness_vertices=sorted(witness) if witness is not None else None,
        split_mixed_counts=split_counts,
        face_degrees=face_degs,
        n_components=ncomp,
        components_orbits=comps,
        sigma_images=list(sigma_t),
        alpha_images=list(alpha_t),
        phi_images=list(phi[1:]),
        sigma_cycles=cycles_to_str(cycles_of_perm(sigma)),
        alpha_cycles=cycles_to_str(cycles_of_perm(alpha)),
        phi_cycles=cycles_to_str(cycles_of_perm(phi)),
    )


@dataclass
class Table1Row:
    N: int
    unsensed: int
    removed_comp: int
    removed_split: int
    prime_total: int
    knots: int
    links: int

    def to_latex_row(self) -> str:
        """Return a LaTeX row suitable for Table 1."""
        return f"{self.N} & {self.unsensed} & {self.removed_comp} & {self.removed_split} & {self.prime_total} & {self.knots} & {self.links} \\\\"


def compute_table1_and_primes(
    N: int,
    *,
    target_genus: int = 1,
    forbid_edge_loops: bool = True,
    forbid_monogons: bool = True,
    forbid_bigons: bool = False,
    progress: bool = False,
) -> Tuple[Table1Row, List[ProjectionPassport]]:
    """Generate unsensed candidates, apply primeness filters, and return:

      - one Table1Row (counts)
      - list of passports for *prime* projections only
    """

    unsensed = generate_unsensed_candidate_projections(
        N,
        target_genus=target_genus,
        forbid_edge_loops=forbid_edge_loops,
        forbid_monogons=forbid_monogons,
        forbid_bigons=forbid_bigons,
        progress=progress,
    )

    removed_comp = 0
    removed_split = 0
    prime_passports: List[ProjectionPassport] = []

    knots = 0
    links = 0

    for sigma_t, alpha_t in unsensed:
        p = passport(sigma_t, alpha_t)

        if p.composite:
            removed_comp += 1
            continue

        if p.n_components >= 2 and p.split_witnessed:
            removed_split += 1
            continue

        # survives primeness filters
        prime_passports.append(p)

        if p.n_components == 1:
            knots += 1
        else:
            links += 1

    row = Table1Row(
        N=N,
        unsensed=len(unsensed),
        removed_comp=removed_comp,
        removed_split=removed_split,
        prime_total=len(prime_passports),
        knots=knots,
        links=links,
    )

    # Deterministic ordering of prime outputs (matches the ordering of unsensed list).
    return row, prime_passports


def write_outputs(
    row: Table1Row,
    primes: List[ProjectionPassport],
    *,
    out_dir: Path,
    print_passports: bool = False,
) -> None:
    """Write Table 1 row + prime projection passports to files."""

    out_dir.mkdir(parents=True, exist_ok=True)

    # (a) one LaTeX-ready row
    tex_path = out_dir / f"table1_row_N{row.N}.tex"
    tex_path.write_text(row.to_latex_row() + "\n", encoding="utf-8")

    # (b) machine-readable JSON
    json_path = out_dir / f"prime_projections_N{row.N}.json"
    payload = {
        "table1_row": asdict(row),
        "latex_row": row.to_latex_row(),
        "prime_projections": [asdict(p) for p in primes],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # (c) human-readable passports
    txt_path = out_dir / f"prime_projections_N{row.N}.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"Prime unsensed projections on the torus (g=1), N={row.N}\n")
        f.write(f"Table-1 row: {row.to_latex_row()}\n\n")

        for idx, p in enumerate(primes, start=1):
            f.write(f"[{idx}] {p.kind}; prime={p.prime}; composite={p.composite}; split={p.split_witnessed}\n")
            f.write(f"    face degrees: {p.face_degrees}\n")
            if p.split_mixed_counts:
                f.write(f"    mixed-vertex counts per component: {p.split_mixed_counts}\n")
            if p.composite_witness_vertices is not None:
                f.write(f"    composite witness vertex-set (σ-cycle ordering): {p.composite_witness_vertices}\n")
            f.write(f"    sigma = {p.sigma_cycles}\n")
            f.write(f"    alpha = {p.alpha_cycles}\n")
            f.write(f"    phi   = {p.phi_cycles}\n")
            f.write(f"    components (dart-orbits under <α, σ^2>): {p.components_orbits}\n")
            f.write("\n")

    # Optional: also print passports to stdout (OK for small N like 4).
    if print_passports:
        print("\n" + "=" * 80)
        print(f"PRIME UNSENSED representatives (N={row.N})")
        print("=" * 80)
        for idx, p in enumerate(primes, start=1):
            print(f"\n[{idx}] {p.kind}; prime={p.prime}; composite={p.composite}; split={p.split_witnessed}")
            print(f"    face degrees: {p.face_degrees}")
            if p.split_mixed_counts:
                print(f"    mixed-vertex counts per component: {p.split_mixed_counts}")
            if p.composite_witness_vertices is not None:
                print(f"    composite witness vertex-set (σ-cycle ordering): {p.composite_witness_vertices}")
            print(f"    sigma = {p.sigma_cycles}")
            print(f"    alpha = {p.alpha_cycles}")
            print(f"    phi   = {p.phi_cycles}")
            print(f"    components (dart-orbits under <α, σ^2>): {p.components_orbits}")


# ============================================================
# CLI
# ============================================================


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate prime unsensed 4-regular torus projections and reproduce Table 1 counts.",
    )

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--N", type=int, help="Number of crossings (vertices).")
    g.add_argument("--Ns", type=int, nargs="+", help="Compute several N values in one run.")

    p.add_argument("--genus", type=int, default=1, help="Target orientable genus (Table 1 uses g=1).")

    p.add_argument(
        "--allow-edge-loops",
        action="store_true",
        help="Allow loop edges (α-pairs inside one σ-cycle). Table 1 uses NO edge-loops.",
    )
    p.add_argument(
        "--allow-monogons",
        action="store_true",
        help="Allow monogon faces (φ fixed points). Table 1 forbids monogons.",
    )
    p.add_argument(
        "--forbid-bigons",
        action="store_true",
        help="Forbid bigon faces (φ 2-cycles). Table 1 allows bigons at projection level.",
    )

    p.add_argument(
        "--out",
        type=str,
        default="out",
        help="Output directory (created if missing).",
    )

    p.add_argument(
        "--print-passports",
        action="store_true",
        help="Also print all prime passports to stdout (useful for small N like 4).",
    )

    p.add_argument(
        "--progress",
        action="store_true",
        help="Print occasional progress lines while exploring full matchings (useful for larger N).",
    )

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    Ns: List[int] = [args.N] if args.N is not None else list(args.Ns)

    # Switches matching Table 1 defaults
    forbid_edge_loops = not args.allow_edge_loops
    forbid_monogons = not args.allow_monogons
    forbid_bigons = args.forbid_bigons

    out_dir = Path(args.out)

    print("=" * 80)
    print("Prime unsensed 4-regular projections on a surface (permutation model)")
    print("=" * 80)
    print(f"Target genus: g={args.genus}")
    print("Projection-level switches:")
    print(f"  forbid_edge_loops = {forbid_edge_loops}")
    print(f"  forbid_monogons   = {forbid_monogons}")
    print(f"  forbid_bigons     = {forbid_bigons}")
    print(f"Output directory: {out_dir.resolve()}")
    print("=")

    for N in Ns:
        t0 = time.perf_counter()

        row, primes = compute_table1_and_primes(
            N,
            target_genus=args.genus,
            forbid_edge_loops=forbid_edge_loops,
            forbid_monogons=forbid_monogons,
            forbid_bigons=forbid_bigons,
            progress=args.progress,
        )

        dt = time.perf_counter() - t0

        # Print summary (the part you paste into the paper).
        print("\n" + "-" * 80)
        print(f"N={N}: Table 1 row")
        print(f"  {row.to_latex_row()}")
        print("Counts:")
        print(f"  unsensed       = {row.unsensed}")
        print(f"  removed comp.  = {row.removed_comp}")
        print(f"  removed split  = {row.removed_split}")
        print(f"  prime total    = {row.prime_total}")
        print(f"  knots          = {row.knots}")
        print(f"  links          = {row.links}")
        print(f"(internal) computed in {dt:.3f} seconds")

        write_outputs(row, primes, out_dir=out_dir, print_passports=args.print_passports)

        print(f"Wrote: {out_dir / f'table1_row_N{N}.tex'}")
        print(f"Wrote: {out_dir / f'prime_projections_N{N}.json'}")
        print(f"Wrote: {out_dir / f'prime_projections_N{N}.txt'}")


if __name__ == "__main__":
    main()
