#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
import re
from typing import Any

RE_KR_HEAD = re.compile(r'^\s*Không\s+rõ\b', flags=re.IGNORECASE)
RE_KR_ANY  = re.compile(r'\bKhông\s+rõ\b', flags=re.IGNORECASE)
RE_CONTEXT = re.compile(r'\bBối\s*cảnh\s*:\s*', flags=re.IGNORECASE)

def cut_earliest(s: str, chars=('.',)):
    idxs = [s.find(c) for c in chars if s.find(c) != -1]
    if not idxs:
        return s
    return s[:min(idxs)].strip()

def transform_text(v: Any) -> Any:
    if not isinstance(v, str):
        return v

    # 1) remove quotes
    s = v.replace('"', '').strip()

    # 2) "Không rõ" đứng đầu -> rỗng
    if RE_KR_HEAD.match(s):
        return ""

    # 3) cắt tại "Bối cảnh:"
    s = RE_CONTEXT.split(s, 1)[0].strip()

    # 4) cắt tại "Không rõ" đầu tiên
    m = RE_KR_ANY.search(s)
    if m:
        s = s[:m.start()].strip()

    # 5) cắt tại dấu '.' đầu tiên
    s = cut_earliest(s, chars=('.',))

    return s


def process(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: transform_text(v) for k, v in data.items()}
    if isinstance(data, list):
        return [transform_text(x) for x in data]
    return transform_text(data)


def read_json(path: str) -> Any:
    if path == "-":
        return json.loads(sys.stdin.read())
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str) -> None:
    if path == "-":
        json.dump(data, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")


def main():
    ap = argparse.ArgumentParser("UIT post-processing script (cut at '.' or '\\')")
    ap.add_argument("-i", "--input", default="-")
    ap.add_argument("-o", "--output", default="-")
    args = ap.parse_args()

    data = read_json(args.input)
    out = process(data)
    write_json(out, args.output)


if __name__ == "__main__":
    main()