from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Iterable


@lru_cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs] # actual printable chars
    d = dict(zip(bs, characters)) # byte-int -> printable-char
    return d


@lru_cache
def gpt2_unicode_to_bytes() -> dict[str, int]:
    """inverts the mapping from printable unicode character to original bytes value : go back from serialized text to a raw bytes"""
    byte_encoder = gpt2_bytes_to_unicode()
    return {v: k for k, v in byte_encoder.items()} #{character : int}


def bytes_to_unicode_text(data: bytes) -> str:
    """converts raw bytes into a printable unicode strin representation: b' hi\n' → 'ĠhiĊ'"""
    byte_encoder = gpt2_bytes_to_unicode()
    return "".join(byte_encoder[b] for b in data)


def unicode_token_to_bytes(token: str) -> bytes:
    """converts  a serialized token back into rawy bytes: ĠhiĊ → b' hi\n'"""
    byte_decoder = gpt2_unicode_to_bytes()
    return bytes(byte_decoder[ch] for ch in token)


def unicode_counts_to_bytes(counts: Counter[str]) -> Counter[bytes]:
    """converts unicode key to byte tokens: "Ġhi": 2 ->b" hi": 2 """
    byte_decoder = gpt2_unicode_to_bytes()
    return Counter({bytes(byte_decoder[ch] for ch in s): c for s, c in counts.items()})
    
