#!/usr/bin/env python3
#? 
from __future__ import annotations #? what do we use this

import argparse #? why are we using this
import cProfile #? why are we using this
import os
import pstats #? why are we using this
import sys #? why are we using this
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count 
from typing import Callable 

import regex as re

#? 
# Make sibling imports work no matter where you run from.
HERE = os.path.dirname(os.path.abspath(__file__)) #? Don't understand what is abspath doing and what the following code for path is doing
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from pretokenization import find_chunk_boundaries  # noqa:E402
import pretokenizer_simple  # noqa: E402
import pretokenizer_compile  # noqa: E402


BASE_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

#? What is frozen=True do ?
@dataclass(frozen=True)
class PretokImpl:
    name: str
    fn: Callable[[str, str, list[str]], dict[bytes, int]]

IMPLS: list[PretokImpl] = [PretokImpl("simple", pretokenizer_simple.pretokenize), PretokImpl("compile",pretokenizer_compile.pretokenize),]


def read_text_range(path: str, start: int, end: int) ->str:
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    return data.decode("utf-8", errors="ignore")

#? why this name convention
def _split_on_special_tokens(text: str, special_tokens:
list[str]) -> list[str]:
    if not special_tokens:
        return [text]
    # Escape each token, join with alternation; longer first is usually safer.
    delim = "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True))
    return re.split(delim, text)

#? why does it take as input *
def count_serial(
    impl: PretokImpl, text: str, special_tokens: list[str],
*, strip_special_tokens_first: bool
) -> Counter[bytes]:
    if strip_special_tokens_first:
        total = Counter()
        for part in _split_on_special_tokens(text,
special_tokens):
            total.update(impl.fn(BASE_PATTERN, part, []))
        return total
    return Counter(impl.fn(BASE_PATTERN, text,
special_tokens))


def _worker_count_chunk(args: tuple[str, int, int, str,
list[str], bool]) -> Counter[bytes]:
    path, start, end, impl_name, special_tokens,strip_special_tokens_first = args
    impl = next(x for x in IMPLS if x.name == impl_name)
    text = read_text_range(path, start, end)
    return count_serial(impl, text, special_tokens,
strip_special_tokens_first=strip_special_tokens_first)


def count_chunked_serial(
    impl: PretokImpl,
    path: str,
    *,
    end_offset: int | None,
    desired_chunks: int,
    split_special_token: bytes,
    special_tokens: list[str],
    strip_special_tokens_first: bool,
) -> Counter[bytes]:
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, desired_chunks, split_special_token,
end_offset=end_offset
        )

    total = Counter()
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        text = read_text_range(path, s, e)
        total.update(count_serial(impl, text,
special_tokens,
strip_special_tokens_first=strip_special_tokens_first))
    return total


def count_chunked_parallel(
    impl: PretokImpl,
    path: str,
    *,
    end_offset: int | None,
    desired_chunks: int,
    split_special_token: bytes,
    special_tokens: list[str],
    strip_special_tokens_first: bool,
    processes: int,
) -> Counter[bytes]:
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, desired_chunks, split_special_token,
end_offset=end_offset
        )

    tasks = [
        (path, s, e, impl.name, special_tokens,
strip_special_tokens_first)
        for s, e in zip(boundaries[:-1], boundaries[1:])
    ]

    total = Counter()
    with Pool(processes=processes) as pool:
        for c in pool.imap_unordered(_worker_count_chunk,
tasks, chunksize=1):
            total.update(c)
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    #? what is HERE ?
    ap.add_argument("--data", default=os.path.join(HERE,"..", "data", "tinystories_val.txt"))
    ap.add_argument("--max-bytes", type=int, default=5_000_000, help="Read only first N bytes (for quick runs). 0=all")
    ap.add_argument("--special-token", default="<|endoftext|>")
    ap.add_argument("--strip-special-first", action="store_true", help="Split on special tokens before regex pretokenization (matches PDF guidance).")
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--processes", type=int, default=max(1, (cpu_count() or 2) - 1))
    ap.add_argument("--mode", choices=["serial", "chunked", "parallel"], default="serial")
    ap.add_argument("--impl", choices=[x.name for x in IMPLS] + ["all"], default="all")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--pstats-out", default="", help="If set, write pstats file here.")
    args = ap.parse_args() #? what is the difference between "args" and "ap"

    path = os.path.abspath(args.data)
    if not os.path.exists(path):
        raise SystemExit(f"missing data file: {path}")

    end_offset = None
    if args.max_bytes and args.max_bytes > 0:
        end_offset = args.max_bytes

    special_tokens = [args.special_token]
    split_special_token_bytes = args.special_token.encode("utf-8")

    def run_one(impl: PretokImpl) -> Counter[bytes]:
        if args.mode == "serial":
            text = read_text_range(path, 0, end_offset if end_offset is not None else os.path.getsize(path))
            return count_serial(impl, text, special_tokens, strip_special_tokens_first=args.strip_special_first)
        if args.mode == "chunked":
            return count_chunked_serial(
                impl,
                path,
                end_offset=end_offset,
                desired_chunks=args.chunks,
                split_special_token=split_special_token_bytes,
                special_tokens=special_tokens,
                strip_special_tokens_first=args.strip_special_first,
            )
        return count_chunked_parallel(
            impl,
            path,
            end_offset=end_offset,
            desired_chunks=args.chunks,
            split_special_token=split_special_token_bytes,
            special_tokens=special_tokens,
            strip_special_tokens_first=args.strip_special_first,
            processes=args.processes,
        )

    impls = IMPLS if args.impl == "all" else [next(x for x in IMPLS if x.name == args.impl)]
    results: dict[str, Counter[bytes]] = {}

    if args.profile:
        prof = cProfile.Profile()
        prof.enable()
        for impl in impls:
            results[impl.name] = run_one(impl)
        prof.disable()

        stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
        stats.print_stats(30)
        if args.pstats_out:
            stats.dump_stats(args.pstats_out)
    else:
        for impl in impls:
            results[impl.name] = run_one(impl)

    for name, c in results.items():
        total = sum(c.values())
        uniq = len(c)
        top = c.most_common(5)
        print(f"[{name}] total={total} uniq={uniq} top5={top}")

    if "simple" in results and "compile" in results:
        same = results["simple"] == results["compile"]
        print(f"[compare] simple == compile: {same}")
        if not same:
            # show a small diff
            s = results["simple"]
            t = results["compile"]
            only_s = next((k for k in s.keys() if s[k] != t.get(k, 0)), None)
            only_t = next((k for k in t.keys() if t[k] != s.get(k, 0)), None)
            print(f"[diff] example key simple: {only_s!r} -> {s.get(only_s,0)} vs {t.get(only_s,0)}")
            print(f"[diff] example key compile: {only_t!r} -> {t.get(only_t,0)} vs {s.get(only_t,0)}")
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())