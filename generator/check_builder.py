"""Static verification of the Unreal builder script against the generated UE 5.6 Python stub.

    python check_builder.py ../FloodedRotunda/Content/Python/build_rotunda.py

Checks (every finding is printed; exit code 1 if any hard error):
  1. every  unreal.<Class>[.<member>]  reference exists (class, enum value, static method, module function)
  2. sp(unreal.Class, obj, key=value, ...)  - every key is a property of Class (or a base class)
  3. <builder>.node(unreal.Class, {"prop": ...}) - every dict key is a property of Class
  4. MEL.<fn>(...) - MaterialEditingLibrary function exists and the call has an acceptable argument count
  5. weak: every other  obj.method(...)  /  set_editor_property("name") / get_editor_property("name") name exists in
     SOME stub class (catches typos on objects whose type is not statically known)
"""
from __future__ import annotations

import ast
import builtins
import sys

import stub_index as si

PY_NAMES = set(dir(builtins)) | set(dir(str)) | set(dir(list)) | set(dir(dict)) | set(dir(set)) | set(dir(tuple)) | set(dir(float)) | set(dir(int)) \
    | {"append", "extend", "items", "keys", "values", "get", "join", "format", "split", "strip", "endswith", "startswith", "lower", "upper",
       "capitalize", "pop", "update", "setdefault", "sort", "sorted", "reverse", "count", "index", "insert", "remove", "copy", "replace",
       "exists", "isfile", "isdir", "join", "abspath", "dirname", "normpath", "makedirs", "listdir", "getenv", "environ", "path", "load",
       "loads", "dump", "dumps", "time", "format_exc", "exit", "argv", "cos", "sin", "atan2", "radians", "degrees", "sqrt", "round", "ceil",
       "floor", "hypot", "pi", "fabs", "log", "step", "note", "conn", "out", "finish", "tex", "uv", "mask", "const", "const3", "scalar_param",
       "mul", "add", "sub", "lerp", "saturate", "smoothstep", "node", "ok", "fail", "warn", "t0", "name", "fails", "conns", "mat", "_x", "_y",
       "splitext", "frombytes", "read", "byteswap",
       # build_world.py's own MatBuilder / helper methods
       "_bin", "div", "unary", "append", "wp", "nws", "up", "tri_weights", "proj_uvs", "triplanar", "triplanar_perturb",
       "planar_perturb", "world_normal", "vec_param", "mpc", "get_name"}


def chain(node):
    out = []
    while isinstance(node, ast.Attribute):
        out.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        out.append(node.id)
        return list(reversed(out))
    return None


def main(path):
    classes = si.load()
    mod = classes.get("<module>", ([], {}))[1]
    all_members = set()
    for name, (bases, members) in classes.items():
        all_members.update(members)
        all_members.add(name)
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    errors, warns = [], []
    aliases = {"MEL": "MaterialEditingLibrary", "MP": "MaterialProperty"}

    def err(node, msg):
        errors.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    def warn(node, msg):
        warns.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    # alias detection:  X = unreal.Something
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            c = chain(node.value) if isinstance(node.value, ast.Attribute) else None
            if c and c[0] == "unreal" and len(c) == 2:
                aliases[node.targets[0].id] = c[1]

    seen = set()
    for node in ast.walk(tree):
        # 1. unreal.* references
        if isinstance(node, ast.Attribute):
            c = chain(node)
            if c and c[0] in ("unreal", *aliases) and len(c) >= 2:
                if c[0] == "unreal":
                    cls, mem = c[1], (c[2] if len(c) > 2 else None)
                else:
                    cls, mem = aliases[c[0]], c[1]
                key = (cls, mem)
                if key in seen:
                    continue
                seen.add(key)
                if cls not in classes and cls not in mod:
                    err(node, f"unreal.{cls} does not exist")
                elif mem is not None and cls in classes and not si.has_member(classes, cls, mem):
                    err(node, f"unreal.{cls}.{mem} does not exist")
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        fname = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
        # 2. sp(unreal.Class, obj, k=v)
        if fname == "sp" and node.args:
            c = chain(node.args[0]) if isinstance(node.args[0], ast.Attribute) else None
            if c and c[0] == "unreal" and len(c) == 2 and c[1] in classes:
                members = si.mro_members(classes, c[1])
                for kw in node.keywords:
                    if kw.arg and kw.arg not in members:
                        err(node, f"sp({c[1]}): property '{kw.arg}' not found")
        # 3. builder.node(unreal.Class, {..})
        if fname == "node" and node.args:
            c = chain(node.args[0]) if isinstance(node.args[0], ast.Attribute) else None
            if c and c[0] == "unreal" and len(c) == 2:
                if c[1] not in classes:
                    err(node, f"node(): unreal.{c[1]} does not exist")
                elif len(node.args) > 1 and isinstance(node.args[1], ast.Dict):
                    members = si.mro_members(classes, c[1])
                    for k in node.args[1].keys:
                        if isinstance(k, ast.Constant) and isinstance(k.value, str) and k.value not in members:
                            err(node, f"node({c[1]}): property '{k.value}' not found")
        # 4. MEL.fn(...)
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in aliases and aliases[f.value.id] == "MaterialEditingLibrary":
            m = si.mro_members(classes, "MaterialEditingLibrary").get(f.attr)
            if m is None:
                err(node, f"MaterialEditingLibrary.{f.attr} does not exist")
            else:
                nargs = len(node.args)
                n_params = len([a for a in m[1] if a not in ("self", "cls")])
                nreq = m[2] - (1 if m[1] and m[1][0] in ("self", "cls") else 0)
                if nargs < nreq or nargs > n_params:
                    err(node, f"MEL.{f.attr}: {nargs} args given, stub signature {m[1]} (required {nreq})")
        # 5. weak checks
        if isinstance(f, ast.Attribute):
            if f.attr in ("set_editor_property", "get_editor_property") and node.args and isinstance(node.args[0], ast.Constant):
                if node.args[0].value not in all_members:
                    err(node, f"{f.attr}('{node.args[0].value}'): no such property on any stub class")
            elif f.attr not in all_members and f.attr not in PY_NAMES:
                warn(node, f".{f.attr}(...) not found on any stub class (check the receiver type)")
    print(f"checked {path}")
    print(f"  {len(seen)} distinct unreal.* references, {len(errors)} errors, {len(warns)} warnings")
    for e in errors:
        print("  ERROR  " + e)
    for w in sorted(set(warns)):
        print("  warn   " + w)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
