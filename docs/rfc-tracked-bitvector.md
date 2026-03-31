# RFC: TrackedBitVector for honest DMD measurement

## Problem

TrackedArray measures data movement for numpy operations, but submissions
can "escape" tracking by converting to plain Python types:

```python
x_block = np.asarray(x[:rows])  # strips TrackedArray wrapper
mask = 0
for j in range(n_bits):
    if x_block[i][j] > 0:       # reads happen but aren't tracked
        mask |= 1 << j          # Python int ops — invisible to tracker
```

After `np.asarray()`, all subsequent operations (bitwise XOR, comparisons,
shifts) happen on untracked Python integers. The DMD metric undercounts
because it only sees the initial numpy read, not the algorithmic work.

### Impact on current leaderboard

We measured the #1 submission (passive_gf2_rref_minimal_22rows by Hydral8)
which uses exactly this pattern — GF(2) elimination on Python ints:

| Measurement | DMC |
|------------|-----|
| Current (untracked elimination) | 45,904 |
| With TrackedBitVector (fully tracked) | 81,106 |
| Numpy GF2 same algorithm, n+1 samples | 1,631,797 |

The escape hatch undercounts DMC by ~1.6x for this submission. The numpy
version of the same algorithm reports 20x higher DMC — most of that gap is
real (bit-packing genuinely moves less data), but some is from untracked
operations.

## Proposal: TrackedBitVector

A Python int wrapper that records reads and writes on the same LRU stack
used by TrackedArray, measured in **bits** rather than floats:

```python
from sparse_parity.tracked_bitvector import TrackedBitVector, pack_row

tracker = LRUStackTracker()
# Pack numpy row into tracked bit vector
mask = pack_row(x[i], n_bits, "mask_0", tracker)

# All operations are tracked
mask_a ^= mask_b   # reads both (n_bits each), writes result
bit = mask_a[col]  # reads 1 bit
mask_a.swap_with(mask_b)  # reads both, writes both
```

### What's tracked

| Operation | Reads | Writes |
|-----------|-------|--------|
| `a ^ b` | a (n_bits), b (n_bits) | result (n_bits) |
| `a ^= b` | a (n_bits), b (n_bits) | a (n_bits) |
| `a[i]` | a (1 bit) | — |
| `a.swap_with(b)` | a (n_bits), b (n_bits) | a (n_bits), b (n_bits) |
| `bool(a)` | a (n_bits) | — |
| `int(a)` | a (n_bits) | — |

### Shared LRU stack

TrackedBitVector writes/reads go through the same `LRUStackTracker` as
TrackedArray. Bit-level and float-level accesses compete for stack positions
realistically: if you read a large numpy array between two reads of a
bitvector, the bitvector sinks deeper in the stack and costs more.

## Design tradeoffs

### Option A: Automatic wrapping (harness does it)

The evaluation harness could automatically wrap any Python ints derived
from numpy data. This is hard to implement — you'd need to intercept
`int()`, `float()`, `.item()`, and implicit scalar conversion. Python
doesn't provide hooks for integer arithmetic.

**Pros**: Zero burden on submitters. Honest measurement.
**Cons**: Extremely difficult to implement fully. Python ints are
fundamental types — there's no `__int_ufunc__` protocol.

### Option B: Require tracked types (coding standard)

Submissions that do non-numpy computation must use TrackedBitVector
(or a similar tracked type) for all data-bearing operations. The
harness provides TrackedBitVector alongside TrackedArray.

**Pros**: Feasible to implement. Honest measurement. Rewards genuine
algorithmic efficiency.
**Cons**: Burden on submitters. Can't enforce automatically — a
submission could still use plain ints and get undercounted. Requires
human review to verify compliance.

### Option C: Ban non-numpy computation (restrict the design space)

Require all computation to go through numpy operations. Disqualify
submissions that extract data to Python loops.

**Pros**: Simple to enforce. TrackedArray covers everything.
**Cons**: Disqualifies legitimate optimizations. Bit-packing IS more
cache-efficient — banning it penalizes good algorithms.

### Option D: Accept the limitation (document it)

Keep TrackedArray as-is. Document that the metric measures numpy-level
data movement. Accept that bit-packed implementations get lower scores
because they genuinely use less numpy-level bandwidth.

**Pros**: No code changes. Simple.
**Cons**: The metric doesn't measure what it claims to (total data
movement). Rankings may not reflect true energy cost.

## Recommendation

**Option B** with a documentation/tooling assist. Provide TrackedBitVector
in the challenge repo. Add a note in the rules that submissions doing
non-numpy computation should use TrackedBitVector for honest scoring.
The harness can warn (but not block) if it detects `np.asarray()` calls
in submission code.

This preserves the advantage of bit-packed algorithms (they genuinely
move less data per operation) while ensuring the measurement is honest.
The ~1.6x correction for Hydral8's submission shows the escape isn't
catastrophic — the bit-packed approach is still legitimately better than
numpy GF2, just not by as much as the uncorrected score suggests.

## Files in this PR

- `src/sparse_parity/tracked_bitvector.py` — TrackedBitVector implementation
- `tests/test_tracked_bitvector.py` — 16 tests (ops, DMD, integration)
- `docs/rfc-tracked-bitvector.md` — this document
