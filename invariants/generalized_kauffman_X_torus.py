# -*- coding: utf-8 -*-
"""
generalized_kauffman_X_torus.py

Compute the generalized Kauffman bracket polynomial <D> and the normalized polynomial
X(a,x)_D for link diagrams in the thickened torus T^2 x I, following the definition used
by Akimova–Matveev–Tarkaev (Steklov 2018) and Akimova's later tabulations.

Input format is *purely combinatorial* (maps via permutations on darts):

- A 4-regular *projection* (an embedded 4-regular graph) is encoded by a pair of
  permutations (alpha, sigma) on darts H={1,...,2E}:
    * alpha: fixed-point-free involution pairing the two darts of each edge.
    * sigma: product of 4-cycles giving cyclic order around each vertex.

- A *diagram* is obtained by an over/under choice at each vertex.
  We encode this by a bit per vertex:
    bit=0  -> the opposite pair (0,2) in the sigma-cycle is the "over" strand,
    bit=1  -> the opposite pair (1,3) in the sigma-cycle is the "over" strand.

The generalized bracket for the torus is (state sum over A/B smoothings):
    <D> = sum_s  a^{α(s)-β(s)} * (-a^2 - a^{-2})^{γ(s)} * x^{δ(s)},
where:
    α(s) = number of A-smoothings,
    β(s) = number of B-smoothings,
    γ(s) = number of *cut* circles in the state (separating / null-homologous on T^2),
    δ(s) = number of *non-cut* circles in the state (essential / nontrivial in H_1).

Normalization (oriented invariant for knots; for links depends on component orientations):
    X(a,x)_D = (-a)^{-3 w(D)} * <D>,
where w(D) is the writhe (sum of crossing signs).

Notes:
- For *knots* (1 component), w(D) is independent of reversing the knot orientation, so
  X(a,x)_D is well-defined from the diagram alone.
- For *links* (>=2 components), w(D) depends on the choice of orientations of components.
  This module allows you to specify the orientation choice as a bit per component.

This file is intentionally self-contained (no external dependencies).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import re


# ============================================================
# Basic permutation helpers (1-indexed lists with dummy 0)
# ============================================================

def invert(perm: List[int]) -> List[int]:
    """Inverse of a 1-indexed permutation list perm (perm[0] unused)."""
    n = len(perm) - 1
    inv = [0] * (n + 1)
    for i in range(1, n + 1):
        inv[perm[i]] = i
    return inv


def compose(p: List[int], q: List[int]) -> List[int]:
    """Composition p ∘ q for 1-indexed permutation lists."""
    n = len(p) - 1
    r = [0] * (n + 1)
    for i in range(1, n + 1):
        r[i] = p[q[i]]
    return r


def cycles_of_perm(perm: List[int]) -> List[List[int]]:
    """Return cycles of a 1-indexed permutation (as lists)."""
    n = len(perm) - 1
    vis = [False] * (n + 1)
    res: List[List[int]] = []
    for i in range(1, n + 1):
        if not vis[i]:
            cur = []
            j = i
            while not vis[j]:
                vis[j] = True
                cur.append(j)
                j = perm[j]
            res.append(cur)
    return res


# ============================================================
# GF(2) linear algebra on bitmasks (Python int as vector)
# ============================================================

def gf2_add_to_basis(basis: List[int], vec: int) -> None:
    """
    Add vec to a row-echelon basis (in-place) over GF(2).
    Basis vectors are ints; XOR is addition.
    """
    x = vec
    for b in basis:
        x = min(x, x ^ b)
    if x:
        basis.append(x)
        basis.sort(reverse=True)


def gf2_in_span(basis: List[int], vec: int) -> bool:
    """Test if vec belongs to span(basis) over GF(2)."""
    x = vec
    for b in basis:
        x = min(x, x ^ b)
    return x == 0


# ============================================================
# Laurent polynomials in a: dict exponent -> integer coeff
# ============================================================

Laurent = Dict[int, int]  # exponent -> coeff
PolyX = Dict[int, Laurent]  # power of x -> Laurent in a


def laurent_add(p: Laurent, q: Laurent) -> Laurent:
    r = dict(p)
    for e, c in q.items():
        r[e] = r.get(e, 0) + c
        if r[e] == 0:
            del r[e]
    return r


def laurent_mul(p: Laurent, q: Laurent) -> Laurent:
    if not p or not q:
        return {}
    r: Laurent = {}
    for e1, c1 in p.items():
        for e2, c2 in q.items():
            e = e1 + e2
            r[e] = r.get(e, 0) + c1 * c2
    # cleanup zeros
    r = {e: c for e, c in r.items() if c != 0}
    return r


def laurent_pow(p: Laurent, k: int) -> Laurent:
    """Fast exponentiation p^k in Laurent ring."""
    if k < 0:
        raise ValueError("Negative powers not supported for Laurent polynomials (use invert separately if needed).")
    res: Laurent = {0: 1}
    base = p
    n = k
    while n:
        if n & 1:
            res = laurent_mul(res, base)
        base = laurent_mul(base, base)
        n >>= 1
    return res


def laurent_shift(p: Laurent, shift: int) -> Laurent:
    """Multiply p by a^{shift} (i.e., shift all exponents)."""
    return {e + shift: c for e, c in p.items()}


def laurent_scale(p: Laurent, factor: int) -> Laurent:
    """Multiply p by an integer factor."""
    return {e: c * factor for e, c in p.items()}


# ============================================================
# Pretty-printing
# ============================================================

def laurent_to_latex(p: Laurent) -> str:
    """LaTeX-ish string for a Laurent polynomial in a."""
    if not p:
        return "0"
    pieces: List[str] = []
    for e in sorted(p.keys(), reverse=True):
        c = p[e]
        # monomial part
        if e == 0:
            mon = ""
        elif e == 1:
            mon = "a"
        elif e == -1:
            mon = "a^{-1}"
        else:
            mon = f"a^{{{e}}}"
        # coefficient + monomial
        if e == 0:
            term = str(c)
        else:
            if c == 1:
                term = mon
            elif c == -1:
                term = "-" + mon
            else:
                term = f"{c}{mon}"
        pieces.append(term)

    # glue with +/-
    out = pieces[0]
    for term in pieces[1:]:
        if term.startswith("-"):
            out += " - " + term[1:]
        else:
            out += " + " + term
    return out


def polyX_to_latex(poly: PolyX) -> str:
    """LaTeX-ish string for Σ_m P_m(a) x^m."""
    if not poly:
        return "0"
    parts: List[str] = []
    for m in sorted(poly.keys()):
        P = poly[m]
        parts.append(f"({laurent_to_latex(P)})x^{{{m}}}")
    return " + ".join(parts)


# ============================================================
# Parsing cycle notation (optional convenience)
# ============================================================

_cycle_re = re.compile(r"\(([^()]*)\)")


def parse_cycle_notation(text: str, n: int) -> List[int]:
    """
    Parse permutation in disjoint cycle notation like "(1 2 3)(4 5)".
    Missing points are fixed. Returns 1-indexed list perm of length n+1.
    """
    perm = list(range(n + 1))
    for cyc in _cycle_re.findall(text):
        cyc = cyc.strip()
        if not cyc:
            continue
        nums = [int(x) for x in cyc.split()]
        if len(nums) == 1:
            perm[nums[0]] = nums[0]
            continue
        for i in range(len(nums)):
            perm[nums[i]] = nums[(i + 1) % len(nums)]
    return perm


def perm_to_cycle_str(perm: List[int]) -> str:
    """Human-readable cycle string (each cycle starts at its minimum, cycles sorted)."""
    n = len(perm) - 1
    vis = [False] * (n + 1)
    cycles: List[List[int]] = []
    for i in range(1, n + 1):
        if not vis[i]:
            cur = []
            j = i
            while not vis[j]:
                vis[j] = True
                cur.append(j)
                j = perm[j]
            # normalize cycle start
            m = min(cur)
            k = cur.index(m)
            cur = cur[k:] + cur[:k]
            cycles.append(cur)
    cycles.sort(key=lambda c: c[0])
    return "".join("(" + " ".join(map(str, c)) + ")" for c in cycles)


# ============================================================
# Core: 4-regular map + diagram operations
# ============================================================

@dataclass(frozen=True)
class VertexData:
    """A sigma-cycle representing one vertex, stored in cyclic order."""
    darts: Tuple[int, int, int, int]  # (d0,d1,d2,d3) in sigma order


class FourRegularMap:
    """
    A connected 4-regular map given by permutations (alpha, sigma).
    Works for the torus case of the generalized Kauffman bracket.
    """

    def __init__(self, sigma: List[int], alpha: List[int]):
        if len(sigma) != len(alpha):
            raise ValueError("sigma and alpha must have the same length.")
        self.sigma = sigma
        self.alpha = alpha
        self.n = len(sigma) - 1
        if self.n % 2 != 0:
            raise ValueError("Number of darts must be even.")

        # Precompute sigma^2 (opposite darts)
        self.sigma2 = [0] * (self.n + 1)
        for i in range(1, self.n + 1):
            self.sigma2[i] = sigma[sigma[i]]

        # Extract vertices as sigma cycles, normalize and sort
        raw_cycles = cycles_of_perm(sigma)
        verts: List[VertexData] = []
        for cyc in raw_cycles:
            if len(cyc) != 4:
                raise ValueError(f"sigma must be a product of 4-cycles; got a cycle of length {len(cyc)}: {cyc}")
            # normalize to start at min
            m = min(cyc)
            k = cyc.index(m)
            cyc = cyc[k:] + cyc[:k]
            verts.append(VertexData(tuple(cyc)))  # type: ignore
        verts.sort(key=lambda v: v.darts[0])
        self.vertices: List[VertexData] = verts
        self.N = len(self.vertices)

        # Build dart->(vertex_index, position_in_cycle) table for fast lookup
        self.dart_pos: Dict[int, Tuple[int, int]] = {}
        for vi, v in enumerate(self.vertices):
            for pos, d in enumerate(v.darts):
                self.dart_pos[d] = (vi, pos)

        # Edge representatives and edge-id map (for GF(2) masks)
        reps: List[int] = []
        for d in range(1, self.n + 1):
            if d < alpha[d]:
                reps.append(d)
        reps.sort()
        self.edge_rep_to_id = {rep: i for i, rep in enumerate(reps)}
        self.E = len(reps)

        # Boundary subspace (span of face boundaries) in Z2^{E}
        self.boundary_basis: List[int] = []
        self._build_boundary_basis()

        # rho = sigma^2 ∘ alpha (used for component cycles / writhe)
        self.rho = [0] * (self.n + 1)
        for d in range(1, self.n + 1):
            self.rho[d] = self.sigma2[self.alpha[d]]
        self.rho_cycles = cycles_of_perm(self.rho)  # oriented component-cycles (2 per component)
        self.component_pairs = self._pair_rho_cycles()

    @staticmethod
    def make_sigma_4_regular(N: int) -> List[int]:
        """Convenience: sigma=(1 2 3 4)(5 6 7 8)... for N vertices."""
        n = 4 * N
        sigma = [0] * (n + 1)
        for v in range(N):
            a = 4 * v + 1
            sigma[a] = a + 1
            sigma[a + 1] = a + 2
            sigma[a + 2] = a + 3
            sigma[a + 3] = a
        return sigma

    # ---------- Edge masks ----------

    def edge_id(self, dart: int) -> int:
        rep = dart if dart < self.alpha[dart] else self.alpha[dart]
        return self.edge_rep_to_id[rep]

    def cycle_to_edge_mask(self, cycle: Iterable[int]) -> int:
        mask = 0
        for d in cycle:
            mask ^= 1 << self.edge_id(d)
        return mask

    # ---------- Boundary basis (torus cut/noncut test) ----------

    def _build_boundary_basis(self) -> None:
        phi = compose(self.sigma, self.alpha)  # face permutation
        face_cycles = cycles_of_perm(phi)

        basis: List[int] = []
        for cyc in face_cycles:
            mask = self.cycle_to_edge_mask(cyc)
            if mask:
                gf2_add_to_basis(basis, mask)
        self.boundary_basis = basis

    def is_cut_loop_mask(self, mask: int) -> bool:
        """On T^2, a loop is cut <=> its Z2-homology class is 0 <=> mask is a boundary."""
        return gf2_in_span(self.boundary_basis, mask)

    # ---------- State loops for a smoothing pattern ----------

    def _physical_loops_gamma_delta(self, tau: Dict[int, int]) -> Tuple[int, int]:
        """
        Given tau (pairing of darts at vertices after smoothing), compute (gamma, delta):
          gamma = #cut circles
          delta = #noncut circles

        We build permutation p on darts:
            p(d) = tau(alpha(d))
        Its cycles are oriented state circles. Physical circles correspond to pairs of cycles
        related by alpha (orientation reversal).
        """
        p = [0] * (self.n + 1)
        for d in range(1, self.n + 1):
            p[d] = tau[self.alpha[d]]

        cycles_p = cycles_of_perm(p)

        seen: set[frozenset[int]] = set()
        physical_cycles: List[List[int]] = []
        for cyc in cycles_p:
            key = frozenset(cyc)
            if key in seen:
                continue
            partner = frozenset(self.alpha[d] for d in key)
            seen.add(key)
            seen.add(partner)
            physical_cycles.append(cyc)

        gamma = 0
        delta = 0
        for cyc in physical_cycles:
            mask = self.cycle_to_edge_mask(cyc)
            if self.is_cut_loop_mask(mask):
                gamma += 1
            else:
                delta += 1
        return gamma, delta

    # ---------- Generalized bracket <D> and X(a,x) ----------

    def generalized_bracket(self, crossing_bits: Tuple[int, ...]) -> PolyX:
        """
        Compute the generalized bracket polynomial <D> for the torus.

        crossing_bits is a tuple/list of length N (#vertices), ordered by increasing
        min-dart of the corresponding sigma-cycles (see self.vertices).

        Convention:
          - For a vertex sigma-cycle (d0 d1 d2 d3),
              bit=0 means over-strand is the opposite pair (d0,d2),
              bit=1 means over-strand is the opposite pair (d1,d3).
          - A/B smoothings:
              We encode a state by state_bit per vertex: 0=A, 1=B.
              The actual smoothing pairing type is:
                  type = crossing_bit XOR state_bit.
              type=0 pairs (d0,d1)(d2,d3), type=1 pairs (d1,d2)(d3,d0).
              This makes switching the crossing (flipping bit) swap A and B.
        """
        if len(crossing_bits) != self.N:
            raise ValueError(f"crossing_bits must have length N={self.N}")

        loop_factor: Laurent = {2: -1, -2: -1}  # (-a^2 - a^{-2})
        poly: PolyX = {}

        for state in range(1 << self.N):
            beta = state.bit_count()
            exp_ab = self.N - 2 * beta  # alpha(s)-beta(s)

            # Build tau as the smoothing pairing involution on darts
            tau: Dict[int, int] = {}
            for vi, vdata in enumerate(self.vertices):
                cb = crossing_bits[vi]
                sb = (state >> vi) & 1
                t = cb ^ sb  # 0 or 1
                d0, d1, d2, d3 = vdata.darts
                if t == 0:
                    tau[d0] = d1; tau[d1] = d0
                    tau[d2] = d3; tau[d3] = d2
                else:
                    tau[d1] = d2; tau[d2] = d1
                    tau[d3] = d0; tau[d0] = d3

            gamma, delta = self._physical_loops_gamma_delta(tau)

            term: Laurent = {exp_ab: 1}
            if gamma:
                term = laurent_mul(term, laurent_pow(loop_factor, gamma))

            poly[delta] = laurent_add(poly.get(delta, {}), term)

        # remove empty
        poly = {m: P for m, P in poly.items() if P}
        return poly

    # ---------- Writhe and normalization ----------

    def _pair_rho_cycles(self) -> List[Tuple[List[int], List[int]]]:
        """
        Pair rho-cycles into (C, C') representing the same physical component with opposite orientation.
        For 4-regular maps arising from link projections, there should be exactly 2 cycles per component.
        """
        cycles = self.rho_cycles
        sets = [frozenset(c) for c in cycles]
        # If there are duplicates (shouldn't), pairing by dict would fail; use a multimap.
        bucket: Dict[frozenset[int], List[int]] = {}
        for i, s in enumerate(sets):
            bucket.setdefault(s, []).append(i)

        used = [False] * len(cycles)
        pairs: List[Tuple[List[int], List[int]]] = []

        for i, cyc in enumerate(cycles):
            if used[i]:
                continue
            Si = sets[i]
            alphaSi = frozenset(self.alpha[d] for d in Si)
            # find partner index
            cand = bucket.get(alphaSi, [])
            j = None
            for k in cand:
                if not used[k]:
                    j = k
                    break
            if j is None:
                # Fallback: treat as unpaired (should not happen in valid projections)
                used[i] = True
                pairs.append((cyc, []))
            else:
                used[i] = True
                used[j] = True
                if i != j:
                    pairs.append((cycles[i], cycles[j]))
                else:
                    pairs.append((cycles[i], cycles[i]))
        return pairs

    def writhe(
        self,
        crossing_bits: Tuple[int, ...],
        component_orientations: Optional[Tuple[int, ...]] = None,
        sign_convention: int = +1,
    ) -> int:
        """
        Compute the writhe w(D) as sum of crossing signs.

        For knots, you can omit component_orientations (it won't affect w).
        For links, component_orientations selects an orientation for each component:
            component_orientations[k] = 0 -> use the first rho-cycle in the pair,
            component_orientations[k] = 1 -> use the second rho-cycle in the pair.

        sign_convention = +1 is the default. If you find all your writhes have opposite sign
        compared to another convention, set sign_convention=-1.
        """
        if len(crossing_bits) != self.N:
            raise ValueError(f"crossing_bits must have length N={self.N}")

        comps = self.component_pairs
        m = len(comps)
        if component_orientations is None:
            component_orientations = tuple(0 for _ in range(m))
        if len(component_orientations) != m:
            raise ValueError(f"component_orientations must have length {m} (number of components).")

        chosen_cycles: List[List[int]] = []
        for ori, (c1, c2) in zip(component_orientations, comps):
            if ori == 0:
                chosen_cycles.append(c1)
            else:
                chosen_cycles.append(c2)

        # Collect outgoing darts at each vertex from the chosen oriented cycles
        outs: List[List[int]] = [[] for _ in range(self.N)]
        for cyc in chosen_cycles:
            for d in cyc:
                vi, _pos = self.dart_pos[d]
                outs[vi].append(d)

        # local coordinate map for positions in sigma-cycle (0..3)
        coords = [(1, 0), (0, 1), (-1, 0), (0, -1)]

        w = 0
        for vi, vdata in enumerate(self.vertices):
            od = outs[vi]
            if len(od) != 2:
                # In a valid diagram, each crossing should be traversed by exactly two strands,
                # hence exactly two outgoing darts in the chosen orientations.
                raise RuntimeError(
                    f"Vertex {vi} has {len(od)} outgoing darts in chosen component orientations; expected 2.\n"
                    f"Outgoing darts list: {od}\n"
                    f"Vertex sigma-cycle: {vdata.darts}"
                )

            cb = crossing_bits[vi]  # 0: (0,2) over, 1: (1,3) over

            # Determine which of the two outgoing darts belongs to the over pair
            def over_pair_index(pos: int) -> int:
                # pair0: positions 0&2; pair1: positions 1&3
                return 0 if pos in (0, 2) else 1

            # cb==0 => over is pair0; cb==1 => over is pair1
            target_over_pair = cb

            d_over = None
            d_under = None
            for d in od:
                _v, pos = self.dart_pos[d]
                if over_pair_index(pos) == target_over_pair:
                    d_over = d
                else:
                    d_under = d

            if d_over is None or d_under is None:
                raise RuntimeError("Failed to split outgoing darts into over/under at a vertex.")

            _v, pos_over = self.dart_pos[d_over]
            _v, pos_under = self.dart_pos[d_under]

            ox, oy = coords[pos_over]
            ux, uy = coords[pos_under]
            det = ox * uy - oy * ux
            sign = 1 if det > 0 else -1

            w += sign_convention * sign

        return w

    def normalized_X(
        self,
        crossing_bits: Tuple[int, ...],
        component_orientations: Optional[Tuple[int, ...]] = None,
        sign_convention: int = +1,
    ) -> Tuple[PolyX, int]:
        """
        Compute X(a,x)_D = (-a)^(-3 w(D)) * <D>.

        Returns (Xpoly, w).
        """
        bracket = self.generalized_bracket(crossing_bits)
        w = self.writhe(crossing_bits, component_orientations, sign_convention=sign_convention)

        # (-a)^(-3w) = (-1)^{w} * a^{-3w}
        factor_sign = -1 if (w % 2) else 1
        shift = -3 * w

        X: PolyX = {}
        for m, P in bracket.items():
            Q = laurent_shift(P, shift)
            if factor_sign == -1:
                Q = laurent_scale(Q, -1)
            X[m] = Q
        return X, w

    # ---------- Skeleton (as used by Akimova for T^2) ----------

    @staticmethod
    def skeleton_from_poly(poly: PolyX) -> Dict[int, Tuple[int, ...]]:
        """
        Skeleton: replace each Laurent coefficient P_m(a)=Σ b_j a^j by the tuple of its nonzero
        coefficients (b_j) in increasing order of j, forgetting the degrees.
        """
        sk: Dict[int, Tuple[int, ...]] = {}
        for m, P in poly.items():
            coeffs: List[int] = []
            for e in sorted(P.keys()):
                c = P[e]
                if c != 0:
                    coeffs.append(c)
            if coeffs:
                sk[m] = tuple(coeffs)
        return sk

    @staticmethod
    def canonicalize_skeleton(sk: Dict[int, Tuple[int, ...]]) -> Dict[int, Tuple[int, ...]]:
        """
        Identify skeletons up to:
          - inversion (reverse order inside each tuple),
          - multiplication by -1 (flip signs of all coefficients).

        Returns a canonical representative (lexicographically minimal).
        """
        cand: List[Dict[int, Tuple[int, ...]]] = []
        # original
        cand.append(sk)
        # inverted
        cand.append({m: tuple(reversed(t)) for m, t in sk.items()})
        # negate
        cand.append({m: tuple(-c for c in t) for m, t in sk.items()})
        # negate + invert
        cand.append({m: tuple(-c for c in reversed(t)) for m, t in sk.items()})

        def key(rep: Dict[int, Tuple[int, ...]]) -> Tuple:
            return tuple((m, rep[m]) for m in sorted(rep.keys()))

        best = min(cand, key=key)
        return best


# ============================================================
# Demo (for your N=4 example)
# ============================================================

if __name__ == "__main__":
    # Example map from your message:
    # sigma = (1 2 3 4)(5 6 7 8)(9 10 11 12)(13 14 15 16)
    # alpha = (1 7)(2 11)(3 15)(4 5)(6 14)(8 9)(10 16)(12 13)
    N = 4
    sigma = FourRegularMap.make_sigma_4_regular(N)
    alpha = [0] * (4 * N + 1)
    alpha_pairs = [(1, 7), (2, 11), (3, 15), (4, 5), (6, 14), (8, 9), (10, 16), (12, 13)]
    for a, b in alpha_pairs:
        alpha[a] = b
        alpha[b] = a

    M = FourRegularMap(sigma, alpha)

    bits = (0, 0, 0, 0)
    bracket = M.generalized_bracket(bits)
    X, w = M.normalized_X(bits)

    print("Map:")
    print("  sigma =", perm_to_cycle_str(sigma))
    print("  alpha =", perm_to_cycle_str(alpha))
    print("Diagram crossing bits:", bits)
    print("\nGeneralized bracket <D> =", polyX_to_latex(bracket))
    print("Skeleton(<D>) =", M.canonicalize_skeleton(M.skeleton_from_poly(bracket)))
    print("\nWrithe w(D) =", w)
    print("Normalized X(a,x) =", polyX_to_latex(X))
