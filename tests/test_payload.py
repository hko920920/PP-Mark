import numpy as np

from ppmark_v02.payload import bytes_to_bits, bits_to_bytes, repeat_bits, majority_vote


def test_bit_conversions_roundtrip():
    original = bytes(range(16))
    bits = bytes_to_bits(original)
    recovered = bits_to_bytes(bits)
    assert recovered.startswith(original)


def test_repeat_and_majority():
    bits = np.array([0, 1, 0, 1], dtype=np.uint8)
    repeated = repeat_bits(bits, 3)
    noisy = repeated.copy()
    noisy[1] = 0
    collapsed = majority_vote(noisy, 3)
    assert collapsed.tolist() == bits.tolist()
