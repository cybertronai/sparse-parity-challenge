"""Tests for TrackedBitVector — bit-level DMD tracking."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import math
import numpy as np
from sparse_parity.lru_tracker import LRUStackTracker
from sparse_parity.tracked_bitvector import TrackedBitVector, pack_row, reset_counter


def setup_function():
    reset_counter()


# =============================================================================
# Basic operations
# =============================================================================

class TestBitVectorOps:
    def test_xor(self):
        t = LRUStackTracker()
        a = TrackedBitVector(0b1100, 4, "a", t)
        b = TrackedBitVector(0b1010, 4, "b", t)
        c = a ^ b
        assert c.value == 0b0110
        assert t.summary()["reads"] == 2  # read a, read b

    def test_ixor(self):
        t = LRUStackTracker()
        a = TrackedBitVector(0b1100, 4, "a", t)
        b = TrackedBitVector(0b1010, 4, "b", t)
        a ^= b
        assert a.value == 0b0110
        assert a.name == "a"  # in-place, same name
        assert t.summary()["reads"] == 2

    def test_and(self):
        t = LRUStackTracker()
        a = TrackedBitVector(0b1100, 4, "a", t)
        b = TrackedBitVector(0b1010, 4, "b", t)
        c = a & b
        assert c.value == 0b1000

    def test_or(self):
        t = LRUStackTracker()
        a = TrackedBitVector(0b1100, 4, "a", t)
        b = TrackedBitVector(0b1010, 4, "b", t)
        c = a | b
        assert c.value == 0b1110

    def test_invert(self):
        t = LRUStackTracker()
        a = TrackedBitVector(0b1100, 4, "a", t)
        b = ~a
        assert b.value == 0b0011

    def test_shift(self):
        t = LRUStackTracker()
        a = TrackedBitVector(0b1100, 4, "a", t)
        assert (a >> 2).value == 0b0011
        assert (a << 1).value == 0b11000

    def test_getitem(self):
        t = LRUStackTracker()
        a = TrackedBitVector(0b1010, 4, "a", t)
        assert a[0] == 0
        assert a[1] == 1
        assert a[3] == 1
        # Each bit extraction is 1 bit read
        assert t.summary()["reads"] == 3

    def test_eq(self):
        t = LRUStackTracker()
        a = TrackedBitVector(5, 4, "a", t)
        b = TrackedBitVector(5, 4, "b", t)
        assert a == b
        assert not (a != b)

    def test_bool(self):
        t = LRUStackTracker()
        a = TrackedBitVector(0, 4, "a", t)
        b = TrackedBitVector(1, 4, "b", t)
        assert not a
        assert b

    def test_swap(self):
        t = LRUStackTracker()
        a = TrackedBitVector(0b1100, 4, "a", t)
        b = TrackedBitVector(0b0011, 4, "b", t)
        a.swap_with(b)
        assert a.value == 0b0011
        assert b.value == 0b1100
        # Swap reads both, writes both
        assert t.summary()["reads"] == 2


# =============================================================================
# DMD accounting
# =============================================================================

class TestBitVectorDMD:
    def test_writes_are_free(self):
        t = LRUStackTracker()
        TrackedBitVector(0b1111, 4, "a", t)
        TrackedBitVector(0b0000, 4, "b", t)
        assert t.summary()["dmd"] == 0.0  # writes only, no cost

    def test_xor_dmd(self):
        t = LRUStackTracker()
        a = TrackedBitVector(0b1100, 4, "a", t)
        b = TrackedBitVector(0b1010, 4, "b", t)
        c = a ^ b
        # Per-element LRU stack: each bit is a separate entry.
        # After writing a(4 bits) then b(4 bits), stack has 8 entries.
        # Reading a then b incurs nonzero DMD from stack distances.
        dmd = t.summary()["dmd"]
        assert dmd > 0, "XOR should have nonzero DMD from reading inputs"
        assert t.summary()["reads"] == 2  # two read() calls (a and b)

    def test_shared_stack_with_lru(self):
        """TrackedBitVector and LRUStackTracker share the same stack."""
        t = LRUStackTracker()
        a = TrackedBitVector(0b11, 2, "a", t)
        t.write("big_buffer", 100)  # push 100 elements onto stack
        a._read()  # a is now deep in the stack
        # a's distance should reflect the 100 elements pushed after it
        dmd = t.summary()["dmd"]
        assert dmd > 2 * math.sqrt(100)  # at least sqrt(100) per bit


# =============================================================================
# Integration with numpy
# =============================================================================

class TestPackRow:
    def test_pack_basic(self):
        t = LRUStackTracker()
        row = np.array([1.0, -1.0, 1.0, -1.0])
        bv = pack_row(row, 4, "packed", t)
        assert bv.value == 0b0101  # bits 0 and 2 are +1

    def test_pack_gf2(self):
        t = LRUStackTracker()
        row = np.array([0, 1, 1, 0], dtype=np.uint8)
        bv = pack_row(row, 4, "packed", t)
        assert bv.value == 0b0110


# =============================================================================
# GF(2) elimination end-to-end
# =============================================================================

class TestGF2WithBitVector:
    def test_solves_parity(self):
        """Full GF(2) solve using TrackedBitVector."""
        rng = np.random.RandomState(42)
        n, k = 20, 3
        secret = sorted(rng.choice(n, k, replace=False).tolist())
        x = rng.choice([-1.0, 1.0], size=(22, n))
        y = np.prod(x[:, secret], axis=1)

        t = LRUStackTracker()
        target_flip = 1 if (k % 2 == 0) else 0

        masks = []
        rhs_list = []
        for i in range(22):
            bv = pack_row(((x[i] + 1) / 2).astype(np.uint8), n, f"m{i}", t)
            masks.append(bv)
            yval = (1 if y[i] > 0 else 0) ^ target_flip
            rhs_list.append(TrackedBitVector(yval, 1, f"r{i}", t))

        pivot_row = 0
        pivot_cols = []
        for col in range(n):
            found = pivot_row
            while found < 22:
                if masks[found][col]:
                    break
                found += 1
            if found == 22:
                continue
            if found != pivot_row:
                masks[pivot_row].swap_with(masks[found])
                rhs_list[pivot_row].swap_with(rhs_list[found])
            for row in range(pivot_row + 1, 22):
                if masks[row][col]:
                    masks[row] ^= masks[pivot_row]
                    rhs_list[row] ^= rhs_list[pivot_row]
            pivot_cols.append(col)
            pivot_row += 1

        solution = [0] * n
        for idx in range(pivot_row - 1, -1, -1):
            col = pivot_cols[idx]
            val = int(rhs_list[idx])
            for c2 in range(col + 1, n):
                if masks[idx][c2] and solution[c2]:
                    val ^= 1
            solution[col] = val

        found_secret = sorted([i for i in range(n) if solution[i]])
        assert found_secret == secret, f"Expected {secret}, got {found_secret}"

        # DMD should be nonzero (elimination did real work)
        assert t.summary()["dmd"] > 0
