"""Index the UE Python stub (unreal-stub) with `ast`: class -> members (methods, properties,
enum values) with inheritance, so scripts can be checked against the real API surface.

usage:  python stub_index.py build            # writes stub_index.pkl next to this file
        python stub_index.py check Class.member [Class.member ...]
"""
from __future__ import annotations

import ast
import os
import pickle
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, ".stub_index.pkl")


def stub_path():
    """The cleaned stub produced by make_stub.py (typings/unreal/__init__.pyi)."""
    p = os.path.join(HERE, "typings", "unreal", "__init__.pyi")
    if not os.path.exists(p):
        raise SystemExit("run `python make_stub.py typings` first")
    return p


def build():
    path = stub_path()
    src = open(path, encoding="utf-8-sig").read()
    tree = ast.parse(src)
    classes = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            members = {}
            bases = [b.id if isinstance(b, ast.Name) else getattr(b, "attr", "?") for b in node.bases]
            for b in node.body:
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    deco = [getattr(d, "id", getattr(d, "attr", "")) for d in b.decorator_list]
                    kind = "classmethod" if "classmethod" in deco else ("staticmethod" if "staticmethod" in deco else "method")
                    if "setter" in deco or any(d == "setter" for d in deco):
                        continue
                    if "property" in deco:
                        kind = "property"
                    args = [a.arg for a in b.args.args]
                    nreq = len(b.args.args) - len(b.args.defaults)
                    members[b.name] = (kind, args, nreq, ast.unparse(b.returns) if b.returns else None)
                elif isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name):
                    members[b.target.id] = ("attr", [], 0, ast.unparse(b.annotation))
                elif isinstance(b, ast.Assign):
                    for t in b.targets:
                        if isinstance(t, ast.Name):
                            members[t.id] = ("attr", [], 0, None)
            doc = ast.get_docstring(node, clean=False) or ""
            for mm in re.finditer(r"^\s*- ``(\w+)`` \(([^)]*)\):", doc, re.M):
                members.setdefault(mm.group(1), ("editprop", [], 0, mm.group(2)))
            classes[node.name] = (bases, members)
        elif isinstance(node, ast.FunctionDef):
            classes.setdefault("<module>", ([], {}))[1][node.name] = ("function", [a.arg for a in node.args.args], len(node.args.args) - len(node.args.defaults), None)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    classes.setdefault("<module>", ([], {}))[1][t.id] = ("attr", [], 0, None)
    pickle.dump(classes, open(CACHE, "wb"))
    return classes


def load():
    if not os.path.exists(CACHE):
        return build()
    return pickle.load(open(CACHE, "rb"))


def mro_members(classes, name, _seen=None):
    out = {}
    if name not in classes:
        return out
    bases, members = classes[name]
    for b in bases:
        out.update(mro_members(classes, b))
    out.update(members)
    return out


def has_member(classes, cls, member):
    return member in mro_members(classes, cls)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "find":
        c = load()
        cls = sys.argv[2]
        pats = [p.lower() for p in sys.argv[3:]]
        for name, m in sorted(mro_members(c, cls).items()):
            if any(p in name.lower() for p in pats):
                print(f"  {cls}.{name:50s} {m[0]:9s} {str(m[3])[:60]}")
        sys.exit(0)
    if cmd == "build":
        c = build()
        print(len(c), "classes indexed ->", CACHE)
    else:
        c = load()
        for q in sys.argv[2:]:
            cls, _, mem = q.partition(".")
            if not mem:
                print(f"{q:60s} {'OK' if cls in c else 'MISSING CLASS'}")
            else:
                m = mro_members(c, cls).get(mem)
                print(f"{q:60s} {'OK ' + str(m[0]) + ' args=' + str(m[1][:6]) if m else ('MISSING MEMBER' if cls in c else 'MISSING CLASS')}")
