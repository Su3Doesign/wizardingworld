"""Clean the generated UE stub (unreal-stub) into a parseable typings/unreal/__init__.pyi.

The UE-generated stub embeds raw UE text-format struct defaults (e.g. `(Left=0.0,Top=0.0)`)
that are not valid Python. We keep names, annotations and docstrings, and replace every
default value with `...` (valid in a .pyi).

usage: python make_stub.py [OUT_DIR]   (default: ./typings)
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys


def strip_defaults(line: str) -> str:
    """Rewrite `def f(a: T = <anything>, b=...)` so every default becomes `...`."""
    i = line.find("(")
    if i < 0:
        return line
    out = [line[: i + 1]]
    depth = 1
    j = i + 1
    n = len(line)
    in_default = False
    quote = None
    while j < n and depth > 0:
        ch = line[j]
        if quote:
            if not in_default:
                out.append(ch)
            if ch == "\\":
                j += 1
                if j < n and not in_default:
                    out.append(line[j])
            elif ch == quote:
                quote = None
            j += 1
            continue
        if ch in "\"'":
            quote = ch
            if not in_default:
                out.append(ch)
            j += 1
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                if in_default:
                    out.append("...")
                    in_default = False
                out.append(ch)
                j += 1
                break
        if depth == 1 and ch == "=" and line[j - 1] == " " and not in_default:
            out.append("= ")
            in_default = True
            j += 1
            while j < n and line[j] == " ":
                j += 1
            continue
        if depth == 1 and ch == "," and in_default:
            out.append("...")
            in_default = False
            out.append(ch)
            j += 1
            continue
        if not in_default:
            out.append(ch)
        j += 1
    out.append(line[j:])
    return "".join(out)


def clean(src: str) -> str:
    lines = src.split("\n")
    res = []
    for ln in lines:
        s = ln.lstrip()
        if s.startswith("def ") and " = " in ln and s.rstrip().endswith(":") is False:
            res.append(ln)
        elif s.startswith("def ") and " = " in ln:
            res.append(strip_defaults(ln))
        else:
            res.append(ln)
    return "\n".join(res)


def main(out_dir="typings"):
    spec = importlib.util.find_spec("unreal")
    p = os.path.join(os.path.dirname(spec.origin), "unreal.py")
    src = open(p, encoding="utf-8-sig").read()
    cleaned = clean(src)
    try:
        ast.parse(cleaned)
    except SyntaxError as e:
        line = cleaned.split("\n")[e.lineno - 1]
        print("still invalid:", e.msg, "line", e.lineno, repr(line[:200]))
        raise SystemExit(1)
    os.makedirs(os.path.join(out_dir, "unreal"), exist_ok=True)
    dst = os.path.join(out_dir, "unreal", "__init__.pyi")
    open(dst, "w", encoding="utf-8").write(cleaned)
    print(f"wrote {dst} ({len(cleaned) / 1e6:.1f} MB), parses cleanly")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "typings")
