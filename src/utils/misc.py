import hashlib
import pickle
import sys


def stable_hash(value) -> int:
    """
    An alternative to the hash() function, behaving in a similar way.
    However, stable_hash aims to return the same hash values across multiple runs. Logically, references
    prevent this from working. Still, for primitive values it should return the same value.
    Also, it allows hashing a wider range of inputs than hash.
    Args:
        value: Any object which can be pickle dumped.
    Returns: int with 256 bits
    """
    b = pickle.dumps(value)
    return int.from_bytes(hashlib.sha256(b).digest(), byteorder=sys.byteorder)
