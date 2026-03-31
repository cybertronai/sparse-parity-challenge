"""Tracked bit-packed integer for DMD measurement of non-numpy computation.

TrackedBitVector wraps a Python int representing a packed bit vector.
Every operation (XOR, AND, OR, comparison, bit extraction) records
reads and writes on the same LRU stack used by TrackedArray, measured
in bits rather than floats.

This closes the "escape to Python ints" loophole in TrackedArray:
algorithms that pack numpy data into Python integers for bitwise
computation can use TrackedBitVector to ensure all data movement
is accounted for.

Usage:
    from sparse_parity.tracked_bitvector import TrackedBitVector
    from sparse_parity.lru_tracker import LRUStackTracker

    tracker = LRUStackTracker()
    a = TrackedBitVector(0b1101, 4, "a", tracker)
    b = TrackedBitVector(0b1010, 4, "b", tracker)
    c = a ^ b   # reads a (4 bits), reads b (4 bits), writes c (4 bits)
    bit = a[2]  # reads 1 bit from a
"""

import math

_next_id = 0


def _auto_name(prefix="bv"):
    global _next_id
    _next_id += 1
    return f"_{prefix}_{_next_id}"


def reset_counter():
    global _next_id
    _next_id = 0


class TrackedBitVector:
    """A Python int with DMD tracking, measured in bits.

    Every read/write goes through the same LRUStackTracker used by
    TrackedArray, so bit-level and float-level operations share the
    same LRU stack and compete for stack positions realistically.
    """

    __slots__ = ('value', 'n_bits', 'name', 'tracker')

    def __init__(self, value, n_bits, name=None, tracker=None):
        self.value = int(value)
        self.n_bits = n_bits
        self.name = name or _auto_name()
        self.tracker = tracker
        if tracker is not None:
            tracker.write(self.name, n_bits)

    def _read(self, size=None):
        if self.tracker is not None:
            self.tracker.read(self.name, size or self.n_bits)

    def _make_result(self, value, prefix="bv"):
        name = _auto_name(prefix)
        result = TrackedBitVector.__new__(TrackedBitVector)
        result.value = int(value)
        result.n_bits = self.n_bits
        result.name = name
        result.tracker = self.tracker
        if self.tracker is not None:
            self.tracker.write(name, self.n_bits)
        return result

    # --- Bitwise operations ---

    def __xor__(self, other):
        self._read()
        if isinstance(other, TrackedBitVector):
            other._read()
            return self._make_result(self.value ^ other.value, "xor")
        return self._make_result(self.value ^ int(other), "xor")

    def __ixor__(self, other):
        # In-place XOR: read self + other, write back to self
        self._read()
        if isinstance(other, TrackedBitVector):
            other._read()
            self.value ^= other.value
        else:
            self.value ^= int(other)
        if self.tracker is not None:
            self.tracker.write(self.name, self.n_bits)
        return self

    def __and__(self, other):
        self._read()
        if isinstance(other, TrackedBitVector):
            other._read()
            return self._make_result(self.value & other.value, "and")
        return self._make_result(self.value & int(other), "and")

    def __or__(self, other):
        self._read()
        if isinstance(other, TrackedBitVector):
            other._read()
            return self._make_result(self.value | other.value, "or")
        return self._make_result(self.value | int(other), "or")

    def __invert__(self):
        self._read()
        mask = (1 << self.n_bits) - 1
        return self._make_result(self.value ^ mask, "not")

    def __lshift__(self, n):
        self._read()
        return self._make_result(self.value << n, "lsh")

    def __rshift__(self, n):
        self._read()
        return self._make_result(self.value >> n, "rsh")

    # --- Bit extraction ---

    def __getitem__(self, bit_idx):
        """Extract a single bit. Costs 1 bit read."""
        if self.tracker is not None:
            self.tracker.read(self.name, 1)
        return (self.value >> bit_idx) & 1

    # --- Comparison ---

    def __eq__(self, other):
        self._read()
        if isinstance(other, TrackedBitVector):
            other._read()
            return self.value == other.value
        return self.value == int(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __bool__(self):
        """bool(bv) — reads the vector to test if nonzero."""
        self._read()
        return bool(self.value)

    # --- Reverse operations (when int is on the left) ---

    def __rxor__(self, other):
        self._read()
        return self._make_result(int(other) ^ self.value, "xor")

    def __rand__(self, other):
        self._read()
        return self._make_result(int(other) & self.value, "and")

    def __ror__(self, other):
        self._read()
        return self._make_result(int(other) | self.value, "or")

    # --- Conversion ---

    def __int__(self):
        self._read()
        return self.value

    def __index__(self):
        """Allows use in slicing and bin(). Records a read."""
        self._read()
        return self.value

    def __repr__(self):
        return f"TrackedBitVector({self.value:#0{self.n_bits+2}b}, {self.n_bits} bits, {self.name!r})"

    # --- Swap support (for row swaps in elimination) ---

    def swap_with(self, other):
        """Swap values with another TrackedBitVector. Reads both, writes both."""
        self._read()
        other._read()
        self.value, other.value = other.value, self.value
        if self.tracker is not None:
            self.tracker.write(self.name, self.n_bits)
        if other.tracker is not None:
            other.tracker.write(other.name, other.n_bits)


def pack_row(array_row, n_bits, name=None, tracker=None):
    """Pack a numpy row of {-1, +1} or {0, 1} values into a TrackedBitVector.

    If the input is a TrackedArray, a read is recorded (via normal indexing).
    The resulting TrackedBitVector records a write.
    """
    mask = 0
    for j in range(n_bits):
        val = array_row[j]  # triggers TrackedArray read if applicable
        if hasattr(val, 'item'):
            val = val.item()
        if val > 0:
            mask |= 1 << j
    return TrackedBitVector(mask, n_bits, name, tracker)
