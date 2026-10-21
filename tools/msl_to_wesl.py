#!/usr/bin/env python3
"""Translate this repo's Metal compute shaders into WGSL.

The .metal files are not modified. Each kernel is emitted as its own entry so
bindings stay in that kernel's argument order. Template calls are monomorphised.
"""

import re
import sys
from pathlib import Path

HOST_STRUCTS = {
    "Params",
    "Step",
    "BasisParams",
    "ModeParams",
    "GemvParams",
    "FieldParams",
    "MomentParams",
    "TraceParams",
}

NULLABLE = {"coef", "saved", "acc"}

# Callees that live in another module and take a pointer for the last two args.
BUILTIN_SIGS = {
    "bvh_collect": [
        ("origin_u", "value", "vec3<f32>", None, None),
        ("dir", "value", "vec3<f32>", None, None),
        ("body_scale", "value", "f32", None, None),
        ("launch_offset", "value", "f32", None, None),
        ("origin_abs", "value", "f32", None, None),
        ("merge_abs", "value", "f32", None, None),
        ("merge_rel", "value", "f32", None, None),
        ("merged", "array", "f32", 128, None),
        ("overflow", "mut", "bool", None, None),
    ]
}


def strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    out = []
    for line in src.splitlines():
        if "//" in line:
            line = line[: line.index("//")]
        out.append(line)
    return "\n".join(out)


def join_continuations(src: str) -> str:
    return src.replace("\\\n", " ")


def expand_macros(src: str) -> str:
    object_macros = {}
    function_macros = {}
    lines = []
    for line in src.splitlines():
        raw = line.strip()
        if raw.startswith("#define "):
            body = raw[len("#define ") :].strip()
            m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\(([^)]*)\)\s*(.*)", body)
            if m:
                name, args, repl = m.group(1), m.group(2), m.group(3)
                args = [a.strip() for a in args.split(",") if a.strip()]
                function_macros[name] = (args, repl.strip())
            else:
                parts = body.split(None, 1)
                if len(parts) == 2:
                    object_macros[parts[0]] = parts[1].strip()
            continue
        if raw.startswith("#"):
            continue
        lines.append(line)
    text = "\n".join(lines)

    def subst_object(text):
        for _ in range(8):
            changed = False
            for name, value in object_macros.items():
                nxt = re.sub(rf"\b{name}\b", value, text)
                changed = changed or nxt != text
                text = nxt
            if not changed:
                break
        return text

    text = subst_object(text)

    def expand_call(args, repl, actuals):
        out = repl
        for arg, actual in zip(args, actuals):
            out = out.replace(f"{arg}##", actual)
            out = out.replace(f"##{arg}", actual)
            out = re.sub(rf"\b{arg}\b", actual, out)
        return out

    def find_call(text, name, start=0):
        token = name + "("
        i = start
        while True:
            i = text.find(token, i)
            if i < 0:
                return None
            if i > 0 and (text[i - 1].isalnum() or text[i - 1] == "_"):
                i += 1
                continue
            j = i + len(token)
            depth = 1
            args = []
            cur = []
            while j < len(text) and depth:
                ch = text[j]
                if ch == "(":
                    depth += 1
                    cur.append(ch)
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        args.append("".join(cur).strip())
                        break
                    cur.append(ch)
                elif ch == "," and depth == 1:
                    args.append("".join(cur).strip())
                    cur = []
                else:
                    cur.append(ch)
                j += 1
            else:
                return None
            if len(args) == 1 and args[0] == "":
                args = []
            return i, j + 1, args

    for _ in range(16):
        changed = False
        for name, (args, repl) in list(function_macros.items()):
            found = find_call(text, name)
            if not found:
                continue
            i, j, actuals = found
            if len(actuals) != len(args):
                continue
            text = text[:i] + expand_call(args, repl, actuals) + text[j:]
            changed = True
            break
        if not changed:
            break
    return subst_object(text)


def jlen(ord_):
    return 4 if ord_ <= 1 else 10 if ord_ == 2 else 20 if ord_ == 3 else 35


def ncoeff(ord_):
    return 1 if ord_ <= 0 else 4 if ord_ == 1 else 10 if ord_ == 2 else 20 if ord_ == 3 else 35


def eval_arg(expr: str):
    expr = expr.strip()
    expr = re.sub(r"\bjlen\s*\(\s*(-?\d+)\s*\)", lambda m: str(jlen(int(m.group(1)))), expr)
    expr = re.sub(r"\bncoeff\s*\(\s*(-?\d+)\s*\)", lambda m: str(ncoeff(int(m.group(1)))), expr)
    expr = expr.replace("true", "1").replace("false", "0")
    if re.fullmatch(r"[0-9+\-*/() <>=!&|]+", expr):
        value = eval(expr)
        if isinstance(value, bool) or any(op in expr for op in (">", "<", "=")):
            return "1" if value else "0"
        return str(int(value))
    return expr


def param_name(raw: str) -> str:
    names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", raw)
    return names[-1] if names else ""


def array_len(name: str, ty: str, default: str) -> str:
    """Pad arrays that are passed through one shared pointer type.

    `acc[35]` is a local jet product, not the contraction accumulator, so it
    keeps the length written in the shader.
    """
    if name == "pows":
        return "33"
    if name == "saved" and str(ty).startswith("Jt"):
        return "25"
    if name == "saved" and ty == "ds":
        return "625"
    return default


def pointer_len(name: str, ty: str) -> str:
    if name == "acc" and ty == "dsacc":
        return "9"
    return array_len(name, ty, {"ds": "3", "dsacc": "9", "f32": "128"}.get(ty, "8"))


def const_int(expr: str):
    expr = expr.strip()
    expr = re.sub(r"(\d+)u\b", r"\1", expr)
    expr = re.sub(r"\bjlen\s*\(\s*(-?\d+)\s*\)", lambda m: str(jlen(int(m.group(1)))), expr)
    expr = re.sub(r"\bncoeff\s*\(\s*(-?\d+)\s*\)", lambda m: str(ncoeff(int(m.group(1)))), expr)
    if re.fullmatch(r"[0-9+\-*/() ]+", expr):
        return str(int(eval(expr)))
    return None


def _split(args: str, sep: str):
    """Split on sep outside (), [], {} and identifier<...> template lists."""
    out, cur = [], []
    paren = brace = bracket = angle = 0
    for i, ch in enumerate(args):
        if ch == sep and paren == brace == bracket == angle == 0:
            out.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
        if ch == "(":
            paren += 1
        elif ch == ")":
            paren -= 1
        elif ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket -= 1
        elif ch == "<" and i > 0 and (args[i - 1].isalnum() or args[i - 1] == "_"):
            angle += 1
        elif ch == ">" and angle:
            angle -= 1
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


def split_commas(args: str):
    return _split(args, ",")


def split_semis(args: str):
    return _split(args, ";")


def matching(text, open_at, open_ch, close_ch):
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
    raise RuntimeError(f"unbalanced {open_ch} at {open_at}")


def skip_ws(s, i):
    while i < len(s) and s[i] in " \t\n\r":
        i += 1
    return i


def read_balanced(s, i, open_ch, close_ch):
    return matching(s, i, open_ch, close_ch) + 1


def read_stmt(s, i):
    i = skip_ws(s, i)
    if i >= len(s):
        return "", i
    if s[i] == "{":
        return s[i : read_balanced(s, i, "{", "}")], read_balanced(s, i, "{", "}")
    m = re.match(r"(for|while|if)\b", s[i:])
    if m:
        k = skip_ws(s, i + len(m.group(1)))
        if k < len(s) and s[k] == "(":
            k = read_balanced(s, k, "(", ")")
        _, k = read_stmt(s, k)
        if m.group(1) == "if":
            t = skip_ws(s, k)
            if s.startswith("else", t) and (t + 4 == len(s) or not (s[t + 4].isalnum() or s[t + 4] == "_")):
                _, k = read_stmt(s, t + 4)
        return s[i:k], k
    j = i
    paren = brace = bracket = 0
    while j < len(s):
        ch = s[j]
        if ch == "(":
            paren += 1
        elif ch == ")":
            paren -= 1
        elif ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket -= 1
        elif ch == ";" and paren == brace == bracket == 0:
            return s[i : j + 1], j + 1
        j += 1
    return s[i:], len(s)


def iter_stmts(block_inside: str):
    i = 0
    while True:
        i = skip_ws(block_inside, i)
        if i >= len(block_inside):
            return
        stmt, i = read_stmt(block_inside, i)
        if stmt.strip():
            yield stmt


def preprocess(src: str, tg: int | None) -> str:
    src = strip_comments(src)
    src = join_continuations(src)
    if tg is not None:
        src = re.sub(r"#define\s+TG\s+\d+u", f"#define TG {tg}u", src)
    src = src.replace("metal::precise::", "")
    src = src.replace("precise::", "")
    src = re.sub(r"\bINFINITY\b", "bitcast<f32>(0x7f800000u)", src)
    src = expand_macros(src)
    src = re.sub(r"(threadgroup\s+(?:float|uint)\s+[A-Za-z0-9_]+\s*\[)1024(\])", r"\g<1>64\2", src)
    src = re.sub(r"\bstatic_assert\s*\([^;]*;", "", src)
    src = re.sub(r"\bconstexpr\s+int\s+\w+\s*\([^)]*\)\s*\{[^{}]*\}", "", src)
    src = re.sub(r"\busing\s+namespace\s+[A-Za-z0-9_:]+\s*;", "", src)
    # Drop the hardware intersector. bvh_collect is the wasm walk.
    src = remove_function(src, "collect_hits")
    src = replace_collect_calls(src)
    return src


def remove_function(src: str, name: str) -> str:
    m = re.search(rf"\b{name}\s*\(", src)
    if not m:
        return src
    # Walk back to the return type at line start.
    start = src.rfind("\n", 0, m.start())
    start = 0 if start < 0 else start + 1
    brace = src.find("{", m.end())
    if brace < 0:
        return src
    end = matching(src, brace, "{", "}") + 1
    return src[:start] + src[end:]


def replace_collect_calls(src: str) -> str:
    token = "collect_hits("
    out = []
    i = 0
    while True:
        j = src.find(token, i)
        if j < 0:
            out.append(src[i:])
            break
        end = matching(src, j + len("collect_hits"), "(", ")")
        args = split_commas(src[j + len(token) : end])
        # origin, dir, accel, scale, launch, origin_abs, merge_abs, merge_rel, merged, overflow
        if len(args) == 10:
            kept = [args[0], args[1], args[3], args[4], args[5], args[6], args[7], args[8], args[9]]
            out.append(src[i:j] + "bvh_collect(" + ", ".join(kept) + ")")
        else:
            out.append(src[i : end + 1])
        i = end + 1
    return "".join(out)


def parse_items(src: str):
    items = []
    i = 0
    n = len(src)
    while i < n:
        i = skip_ws(src, i)
        if i >= n:
            break
        if src.startswith("template", i) and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            gt = src.find(">", i)
            params = [p.strip() for p in src[src.find("<", i) + 1 : gt].split(",")]
            names = []
            for p in params:
                kind, name = p.split()
                names.append((kind, name))
            i = skip_ws(src, gt + 1)
            spec = None
            if src.startswith("struct", i):
                sm = re.match(r"struct\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:<([^>]*)>)?", src[i:])
                fname = sm.group(1)
                spec = sm.group(2)
                brace = src.find("{", i)
                end = matching(src, brace, "{", "}")
                body = src[brace : end + 1]
                if fname == "HighStore":
                    items.append(
                        {
                            "kind": "high",
                            "name": "high_apply",
                            "params": names,
                            "spec": spec,
                            "body": body,
                        }
                    )
                else:
                    items.append(
                        {
                            "kind": "tstruct",
                            "name": fname,
                            "params": names,
                            "body": body,
                        }
                    )
                i = end + 1
                if i < n and src[i] == ";":
                    i += 1
                continue
            header_end = src.find("{", i)
            paren = src.find("(", i)
            name_m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", src[i:header_end])
            fname = name_m.group(1)
            ret = src[i : i + name_m.start()].strip()
            ret = re.sub(r"^inline\s+", "", ret).strip()
            signature = src[paren:header_end].strip()
            end = matching(src, header_end, "{", "}")
            items.append(
                {
                    "kind": "template",
                    "name": fname,
                    "params": names,
                    "ret": ret,
                    "signature": signature,
                    "body": src[header_end : end + 1],
                }
            )
            i = end + 1
            continue
        if src.startswith("struct", i):
            sm = re.match(r"struct\s+([A-Za-z_][A-Za-z0-9_]*)\s*", src[i:])
            brace = src.find("{", i)
            end = matching(src, brace, "{", "}")
            items.append({"kind": "struct", "name": sm.group(1), "body": src[brace : end + 1]})
            i = end + 1
            if i < n and src[i] == ";":
                i += 1
            continue
        fm = re.match(
            r"(?:kernel\s+)?(?:inline\s+)?(.+?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            src[i:],
            re.S,
        )
        if not fm:
            i += 1
            continue
        is_kernel = src.startswith("kernel", i)
        brace = src.find("{", i)
        if brace < 0:
            break
        # A false match can land inside a signature. Require the paren to precede the brace
        # and the name to be the last identifier before '('.
        paren = src.find("(", i)
        if paren < 0 or paren > brace:
            i += 1
            continue
        name = fm.group(2)
        # Reject matches where more code sits between the name's paren and this paren.
        if src.rfind(name, i, paren) < 0:
            i += 1
            continue
        ret = src[i:paren]
        ret = ret[: ret.rfind(name)].strip()
        ret = re.sub(r"^(?:kernel\s+)?(?:inline\s+)?", "", ret).strip()
        if ";" in ret or "=" in ret:
            semi = src.find(";", i)
            i = semi + 1 if 0 <= semi < paren else i + 1
            continue
        end = matching(src, brace, "{", "}")
        items.append(
            {
                "kind": "kernel" if is_kernel else "fn",
                "name": name,
                "ret": ret,
                "signature": src[paren:brace].strip(),
                "body": src[brace : end + 1],
            }
        )
        i = end + 1
    return items


def fold_ifs(text: str) -> str:
    while True:
        m = re.search(r"\bif\s*\(", text)
        if not m:
            return text
        # Find a foldable if. Scan each if.
        found = None
        pos = 0
        while True:
            m = re.search(r"\bif\s*\(", text[pos:])
            if not m:
                break
            abs_i = pos + m.start()
            paren_open = text.find("(", abs_i)
            paren_close = matching(text, paren_open, "(", ")")
            cond = text[paren_open + 1 : paren_close].strip()
            verdict = const_cond(cond)
            if verdict is None:
                pos = paren_close + 1
                continue
            found = (abs_i, paren_close + 1, verdict)
            break
        if not found:
            return text
        abs_i, after, verdict = found
        then_stmt, after_then = read_stmt(text, after)
        else_stmt = ""
        t = skip_ws(text, after_then)
        if text.startswith("else", t) and (t + 4 == len(text) or not (text[t + 4].isalnum() or text[t + 4] == "_")):
            else_stmt, after_then = read_stmt(text, t + 4)
        if verdict:
            repl = then_stmt
        else:
            repl = else_stmt
        text = text[:abs_i] + repl + text[after_then:]


def const_cond(cond: str):
    c = cond.strip()
    if c in {"true", "1"}:
        return True
    if c in {"false", "0"}:
        return False
    if c.startswith("false") and "||" not in c:
        return False
    if c.startswith("true") and "&&" not in c:
        return True
    if c.startswith("!") and c[1:].strip() in {"false", "0"}:
        return True
    if c.startswith("!") and c[1:].strip() in {"true", "1"}:
        return False
    return None


class Prog:
    def __init__(self, items):
        self.items = items
        self.templates = {}
        self.high = {}
        self.structs = []
        self.functions = []
        self.kernels = []
        self.concrete = []
        self.done = {}
        self.jt = set()
        for item in items:
            if item["kind"] == "template":
                self.templates.setdefault(item["name"], []).append(item)
            elif item["kind"] == "high":
                key = "true" if item["spec"] else "primary"
                self.high[key] = item
            elif item["kind"] == "struct":
                self.structs.append(item)
            elif item["kind"] == "tstruct":
                self.templates.setdefault(item["name"], []).append(item)
            elif item["kind"] == "fn":
                self.functions.append(item)
            elif item["kind"] == "kernel":
                self.kernels.append(item)

    def substitute(self, text, env):
        for name, value in env.items():
            text = re.sub(rf"\b{name}\b", value, text)
        text = re.sub(
            r"\bjlen\s*\(\s*([^)]+)\)",
            lambda m: const_int(m.group(1)) or m.group(0),
            text,
        )
        text = re.sub(
            r"\bncoeff\s*\(\s*([^)]+)\)",
            lambda m: const_int(m.group(1)) or m.group(0),
            text,
        )

        def jt(m):
            arg = eval_arg(m.group(1))
            if arg.isdigit() or (arg.startswith("-") and arg[1:].isdigit()):
                self.jt.add(int(arg))
                return f"Jt_{arg}"
            return m.group(0)

        text = re.sub(r"\bJt\s*<\s*([^>]+)\s*>", jt, text)
        return text

    def request(self, name, arg_exprs, actuals):
        ev = [eval_arg(a) for a in arg_exprs]
        if name == "high_apply":
            return self.request_high(ev)
        cands = self.templates.get(name, [])
        entry = None
        for cand in cands:
            if cand["kind"] == "template" and len(cand["params"]) == len(ev):
                entry = cand
                break
        if entry is None:
            raise RuntimeError(f"no template {name}<{', '.join(ev)}>")
        env = {pname: value for (_, pname), value in zip(entry["params"], ev)}
        sig = self.substitute(entry["signature"], env)
        body = self.substitute(entry["body"], env)
        ret = self.substitute(entry["ret"], env)
        params = parse_params(sig)
        omit = []
        if actuals is not None:
            for param, actual in zip(params, actuals):
                if actual.strip() == "nullptr" and param["name"] in NULLABLE:
                    omit.append(param["name"])
        body = self.apply_omit(body, omit)
        kept = [p for p in params if p["name"] not in omit]
        sig = "(" + ", ".join(p["raw"] for p in kept) + ")"
        key = (name, tuple(ev), tuple(omit))
        mangled = name + "_" + "_".join(ev)
        if omit:
            mangled += "__" + "_".join(omit)
        if key not in self.done:
            self.done[key] = mangled
            body = self.resolve(body)
            self.concrete.append({"name": mangled, "ret": ret, "signature": sig, "body": body, "kernel": False})
        return mangled

    def request_high(self, ev):
        # ev is [high_flag, ford] from HighStore<(flag), ford>
        high, ford = ev
        mangled = f"high_apply_{high}_{ford}"
        key = ("high_apply", tuple(ev), ())
        if key in self.done:
            return mangled
        self.done[key] = mangled
        spec = self.high["true"]
        # spec params are the primary's or the specialization's. Specialization
        # template <int FORD> struct HighStore<true, FORD>.
        env = {"FORD": ford, "HIGH": high}
        method = extract_apply(spec["body"])
        sig = self.substitute(method["signature"], env)
        if high == "1":
            body = self.apply_omit(self.substitute(method["body"], env), [])
            body = self.resolve(body)
        else:
            body = "{}"
        self.concrete.append(
            {"name": mangled, "ret": "void", "signature": sig, "body": body, "kernel": False}
        )
        return mangled

    def apply_omit(self, body, omit):
        if "coef" in omit or "acc" in omit:
            body = re.sub(r"thread\s+dsacc\s*\*\s*acc_ptr\s*=\s*[^;]+;", "", body)
            if "coef" in omit or "acc" in omit:
                body = re.sub(r"\bacc_ptr\b", "nullptr" if ("acc" in omit or "coef" in omit) else "acc", body)
        else:
            body = re.sub(r"thread\s+dsacc\s*\*\s*acc_ptr\s*=\s*[^;]+;", "", body)
            body = re.sub(r"\bacc_ptr\b", "acc", body)
        # The branch above treats coef-omit and acc-omit together because this
        # shader never contracts with only one of them. When neither is omitted
        # the else arm rewrites acc_ptr to the local accumulator.
        if "coef" in omit:
            body = re.sub(r"\bcoef\s*!=\s*nullptr\b", "false", body)
        else:
            body = re.sub(r"\bcoef\s*!=\s*nullptr\b", "true", body)
        if "saved" in omit:
            body = re.sub(r"\bsaved\s*!=\s*nullptr\b", "false", body)
        else:
            body = re.sub(r"\bsaved\s*!=\s*nullptr\b", "true", body)
        if "acc" in omit:
            body = re.sub(r"\bacc\s*!=\s*nullptr\b", "false", body)
        else:
            body = re.sub(r"\bacc\s*!=\s*nullptr\b", "true", body)
        return fold_ifs(body)

    def resolve(self, text):
        guard = 0
        while guard < 10000:
            guard += 1
            m = re.search(r"HighStore<\(([^)]+)\),\s*([^>]+)>", text)
            if m:
                high = eval_arg(m.group(1))
                ford = eval_arg(m.group(2))
                name = self.request_high([high, ford])
                text = text[: m.start()] + name + text[m.end() :]
                text = text.replace(name + "::apply", name, 1)
                continue
            m = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*<([^>;]+?)>\s*\(", text)
            if not m or m.group(1) not in self.templates:
                break
            call_end = matching(text, m.end() - 1, "(", ")")
            actuals = split_commas(text[m.end() : call_end])
            args = split_commas(m.group(2))
            mangled = self.request(m.group(1), args, actuals)
            # Drop nullptr actuals that the concrete function omitted.
            kept = [a for a in actuals if a.strip() != "nullptr"]
            # Only drop nullptr when the mangled name records an omission.
            if "__" not in mangled:
                kept = actuals
            text = text[: m.start()] + mangled + "(" + ", ".join(kept) + ")" + text[call_end + 1 :]
        return text

    def build(self):
        for item in self.functions:
            item = dict(item)
            item["body"] = self.resolve(item["body"])
            item["kernel"] = False
            self.concrete.append(item)
        entries = []
        for kern in self.kernels:
            kern = dict(kern)
            kern["body"] = self.resolve(kern["body"])
            kern["kernel"] = True
            entries.append(kern)
        return entries


def extract_apply(struct_body: str):
    m = re.search(r"static\s+void\s+apply\s*(\([^)]*\))\s*(\{)", struct_body)
    if not m:
        return {"signature": "()", "body": "{}"}
    brace = m.start(2)
    # m.start(2) is relative to struct_body. The regex is on struct_body.
    brace = struct_body.find("{", m.start())
    end = matching(struct_body, brace, "{", "}")
    return {"signature": m.group(1), "body": struct_body[brace : end + 1]}


def parse_params(signature: str):
    inner = signature.strip()[1:-1].strip()
    if not inner:
        return []
    params = []
    for raw0 in split_commas(inner):
        raw0 = raw0.strip()
        if not raw0:
            continue
        kind = classify(raw0)
        raw = re.sub(r"\[\[.*?\]\]", "", raw0).strip()
        params.append({"raw": raw, "name": param_name(raw), **kind})
    return params


def param_name(raw: str):
    raw = re.sub(r"\[[^\]]*\]", "", raw)
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", raw)
    return m.group(1) if m else "arg"


def classify(raw: str):
    if any(
        key in raw
        for key in (
            "primitive_acceleration",
            "thread_position_in_grid",
            "thread_index_in_threadgroup",
            "thread_index_in_simdgroup",
            "threadgroup_position_in_grid",
            "threads_per_threadgroup",
            "simdgroup_index",
            "simdgroups_per_threadgroup",
        )
    ):
        return {"kind": "builtin", "ty": None, "size": None, "access": None, "builtin": builtin_kind(raw)}
    raw = re.sub(r"\[\[.*?\]\]", "", raw).strip()
    is_constant = bool(re.search(r"\bconstant\b", raw))
    is_device = bool(re.search(r"\bdevice\b", raw))
    is_threadgroup = bool(re.search(r"\bthreadgroup\b", raw))
    is_thread = bool(re.search(r"\bthread\b", raw))
    is_const = bool(re.search(r"\bconst\b", raw))
    if is_threadgroup and "*" in raw:
        ty = "u32" if re.search(r"\buint\b", raw) else "f32"
        return {"kind": "wg", "ty": ty, "size": None, "access": "read_write", "builtin": None}
    if "*" in raw and (is_device or is_constant):
        if "float2" in raw:
            ty = "array<vec2<f32>>"
        elif re.search(r"\bfloat\b", raw):
            ty = "array<f32>"
        elif re.search(r"\buint\b", raw):
            ty = "array<atomic<u32>>" if re.search(r"\bstats\b", raw) else "array<u32>"
        else:
            sm = re.search(r"\b(GpuPoint|GpuRay|GpuKernel|GpuNode)\b", raw)
            ty = f"array<{sm.group(1)}>" if sm else "array<f32>"
        access = "read_write" if is_device and not is_const else "read"
        return {"kind": "storage", "ty": ty, "size": None, "access": access, "builtin": None}
    arr = re.search(r"\[([^\]]+)\]", raw)
    if is_thread and arr:
        ty = value_ty(re.sub(r"\[[^\]]*\]", "", raw))
        size = const_int(arr.group(1)) or arr.group(1)
        return {"kind": "array", "ty": ty, "size": size, "access": None, "builtin": None}
    if is_thread and "*" in raw:
        ty = value_ty(raw.replace("*", " "))
        size = pointer_len(param_name(raw), ty)
        return {"kind": "array", "ty": ty, "size": size, "access": None, "builtin": None}
    if is_thread and "&" in raw and not is_const:
        return {"kind": "mut", "ty": value_ty(raw), "size": None, "access": None, "builtin": None}
    return {"kind": "value", "ty": value_ty(raw), "size": None, "access": None, "builtin": None}


def builtin_kind(raw: str):
    if "uint3" in raw or "uint3" in raw.replace(" ", ""):
        return "gid3"
    if "thread_position_in_grid" in raw:
        return "gid"
    if "thread_index_in_threadgroup" in raw:
        return "lid"
    if "threadgroup_position_in_grid" in raw:
        return "wgid"
    if "threads_per_threadgroup" in raw:
        return "lanes"
    return "drop"


def value_ty(raw: str):
    raw = re.sub(r"\[\[.*?\]\]", "", raw)
    raw = re.sub(r"\b(threadgroup|thread|constant|const|device|kernel|inline)\b", " ", raw)
    raw = raw.replace("&", " ").replace("*", " ")
    raw = re.sub(r"\[[^\]]*\]", " ", raw)
    parts = raw.split()
    if len(parts) >= 2:
        ty = parts[-2]
    else:
        ty = parts[0] if parts else "f32"
    return metal_ty(ty)


def metal_ty(ty: str) -> str:
    return {
        "uint": "u32",
        "int": "i32",
        "float": "f32",
        "bool": "bool",
        "float2": "vec2<f32>",
        "float3": "vec3<f32>",
        "float4": "vec4<f32>",
        "uint3": "vec3<u32>",
        "void": "void",
    }.get(ty, ty)


def wgsl_param(p, wg_size):
    kind, ty, size, access = p["kind"], p["ty"], p["size"], p["access"]
    if kind == "value":
        return f"{p['name']}: {ty}"
    if kind == "mut":
        return f"{p['name']}: ptr<function, {ty}>"
    if kind == "array":
        return f"{p['name']}: ptr<function, array<{ty}, {size}>>"
    if kind == "storage":
        mode = "read" if access == "read" else "read_write"
        return f"{p['name']}: ptr<storage, {ty}, {mode}>"
    if kind == "wg":
        length = wg_size.get(ty, "64")
        return f"{p['name']}: ptr<workgroup, array<{ty}, {length}>>"
    return None


class Lower:
    def __init__(self, structs_src, wg_size):
        self.struct_names = set(re.findall(r"struct\s+([A-Za-z_][A-Za-z0-9_]*)", structs_src))
        self.struct_names.update({"ds", "dsacc", "vec2<f32>", "vec3<f32>", "vec4<f32>", "u32", "i32", "f32", "bool"})
        self.sigs = dict(BUILTIN_SIGS)
        self.wg_size = wg_size
        self.workgroup = []

    def note(self, name, params):
        self.sigs[name] = [(p["name"], p["kind"], p["ty"], p["size"], p["access"]) for p in params if p["kind"] != "builtin"]

    def lower_fn(self, name, ret, params, body, kernel):
        ctx = Ctx()
        bindings = []
        lets = []
        fn_params = []
        builtins = []
        for p in params:
            if p["kind"] == "builtin":
                if p["builtin"] == "gid":
                    lets.append(f"let {p['name']} = gid3.x;")
                    builtins.append("gid")
                elif p["builtin"] == "gid3":
                    lets.append(f"let {p['name']} = gid3;")
                    builtins.append("gid")
                elif p["builtin"] == "lid":
                    lets.append(f"let {p['name']} = lid3.x;")
                    builtins.append("lid")
                elif p["builtin"] == "wgid":
                    lets.append(f"let {p['name']} = wgid3.x;")
                    builtins.append("wgid")
                elif p["builtin"] == "lanes":
                    lets.append(f"let {p['name']} = 64u;")
                continue
            if kernel and p["kind"] == "storage":
                bindings.append(p)
                ctx.globals.add(p["name"])
                continue
            if kernel and p["kind"] == "value" and p["ty"] in HOST_STRUCTS:
                bindings.append(p)
                ctx.globals.add(p["name"])
                continue
            fn_params.append(p)
            if p["kind"] == "storage":
                ctx.storage_ptrs.add(p["name"])
            elif p["kind"] == "mut":
                ctx.mut_ptrs.add(p["name"])
            elif p["kind"] == "array":
                ctx.array_ptrs.add(p["name"])
            elif p["kind"] == "wg":
                ctx.wg_ptrs.add(p["name"])
                p["size"] = self.wg_size.get(p["ty"], "64")
        body = body.strip()
        if body.startswith("{") and body.endswith("}"):
            inner = body[1:-1]
        else:
            inner = body
        if kernel and "simd_max" in inner:
            inner = re.sub(
                r"uint\s+worst\s*=\s*simd_max\s*\([\s\S]*",
                "atomicMax(&stats[0], iters_used);\natomicAdd(&stats[1], unconverged);\natomicAdd(&stats[2], invalid);\n",
                inner,
            )
            lets = [line for line in lets if not any(n in line for n in ("lane", "sg", "tg", "sgs"))]
        ctx.sigs = self.sigs
        ctx.struct_names = self.struct_names
        lowered = self.lower_block(inner, ctx)
        header_params = []
        if kernel:
            if "gid" in builtins:
                header_params.append("@builtin(global_invocation_id) gid3: vec3<u32>")
            if "lid" in builtins:
                header_params.append("@builtin(local_invocation_id) lid3: vec3<u32>")
            if "wgid" in builtins:
                header_params.append("@builtin(workgroup_id) wgid3: vec3<u32>")
        else:
            for p in fn_params:
                text = wgsl_param(p, self.wg_size)
                if text:
                    header_params.append(text)
        ret_ty = metal_ty(ret.replace("inline", "").strip()) if ret else "void"
        ret_ty = ret_ty if ret_ty else "void"
        if ret_ty in {"uint*", "constant uint*", "void"} or ret_ty.endswith("*"):
            # mul_tab is rewritten to return the table offset.
            if name.startswith("mul_tab"):
                ret_ty = "u32"
            else:
                ret_ty = "void"
        attrs = "@compute @workgroup_size(64)\n" if kernel else ""
        prelude = "\n".join("    " + line for line in lets)
        body_text = lowered
        if prelude:
            body_text = prelude + "\n" + body_text
        fn = f"{attrs}fn {name}({', '.join(header_params)}) -> {ret_ty} {{\n{body_text}\n}}\n"
        if ret_ty == "void":
            fn = f"{attrs}fn {name}({', '.join(header_params)}) {{\n{body_text}\n}}\n"
        if name == "dsv":
            # Host structs store float2 as array<f32, 2> so the bytes match repr(C).
            fn = fn.replace("v: vec2<f32>", "v: array<f32, 2>").replace("v.x", "v[0]").replace("v.y", "v[1]")
        return fn, bindings, ctx.wg_decls

    def lower_block(self, inner, ctx):
        lines = []
        for stmt in iter_stmts(inner):
            lines.append(self.lower_stmt(stmt, ctx))
        return "\n".join(lines)

    def lower_stmt(self, stmt, ctx):
        stmt = stmt.strip()
        wg = re.match(
            r"threadgroup\s+(float|uint)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[([^\]]+)\]\s*;",
            stmt,
        )
        if wg:
            ty = "f32" if wg.group(1) == "float" else "u32"
            length = const_int(wg.group(3)) or wg.group(3)
            ctx.wg_decls.append((wg.group(2), ty, length))
            return ""
        stmt = literals(stmt)
        stmt = types(stmt)
        if stmt.startswith("for"):
            return self.lower_for(stmt, ctx)
        if stmt.startswith("if"):
            return self.lower_if(stmt, ctx)
        if stmt.startswith("{"):
            inner = stmt[1:-1]
            return "{\n" + self.lower_block(inner, ctx) + "\n}"
        stmt = self.rewrite_aliases(stmt, ctx)
        stmt = rewrite_indexing(stmt, ctx)
        stmt = rewrite_mut_assign(stmt, ctx)
        stmt = rewrite_compound(stmt)
        stmt = rewrite_ternary(stmt)
        stmt = self.rewrite_acc(stmt)
        if stmt.startswith("return"):
            expr = stmt[len("return") :].strip()
            if expr.endswith(";"):
                expr = expr[:-1].strip()
            if not expr:
                return "return;"
            return "return " + self.rewrite_calls(expr, ctx) + ";"
        decl = self.try_decl(stmt, ctx)
        if decl is not None:
            return decl
        if stmt.endswith(";"):
            expr = stmt[:-1].strip()
            if not expr:
                return ""
            return self.rewrite_calls(expr, ctx) + ";"
        return self.rewrite_calls(stmt, ctx)

    def lower_for(self, stmt, ctx):
        paren = stmt.find("(")
        close = matching(stmt, paren, "(", ")")
        header = stmt[paren + 1 : close]
        rest = stmt[close + 1 :]
        body_stmt, _ = read_stmt(rest, 0)
        parts = split_semis(header)
        if len(parts) != 3:
            body = self.lower_stmt(body_stmt, ctx)
            return f"for ({header}) {body}"
        init, cond, step = parts
        m = re.match(r"(?:u32|i32|uint|int)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", init.strip())
        if not m:
            body = self.lower_stmt(body_stmt, ctx)
            return f"for ({header}) {body}"
        name = m.group(1)
        ty = "u32" if init.strip().startswith(("uint", "u32")) else "i32"
        init_expr = self.rewrite_calls(rewrite_indexing(types(literals(m.group(2))), ctx), ctx)
        cond = self.rewrite_calls(rewrite_indexing(types(literals(cond)), ctx), ctx)
        step = lower_step(name, step.strip(), ty)
        ctx.locals[name] = ("var", ty)
        body = self.lower_stmt(body_stmt, ctx)
        return f"for (var {name}: {ty} = {init_expr}; {cond}; {step}) {body}"

    def lower_if(self, stmt, ctx):
        paren = stmt.find("(")
        close = matching(stmt, paren, "(", ")")
        cond = stmt[paren + 1 : close]
        rest = stmt[close + 1 :]
        then_stmt, used = read_stmt(rest, 0)
        else_txt = rest[used:]
        cond = self.rewrite_calls(rewrite_indexing(types(literals(cond)), ctx), ctx)
        then_l = self.lower_stmt(then_stmt, ctx)
        if else_txt.strip().startswith("else"):
            else_body = else_txt.strip()[4:].strip()
            else_stmt, _ = read_stmt(else_body, 0)
            else_l = self.lower_stmt(else_stmt, ctx)
            return f"if ({cond}) {then_l} else {else_l}"
        return f"if ({cond}) {then_l}"

    def rewrite_aliases(self, stmt, ctx):
        # `T* name = base + expr` or `T* name = mul_tab_x(tables, p)`.
        m = re.match(
            r"(?:let\s+|var\s+)?([A-Za-z0-9_<>, \*]+)\*\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+);",
            stmt.strip(),
        )
        if not m:
            return stmt
        name, expr = m.group(2), m.group(3).strip()
        add = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*\+\s*(.+)", expr)
        if add:
            ctx.aliases[name] = add.group(1)
            return f"let {name}_base = {self.rewrite_calls(add.group(2), ctx)};"
        call = re.match(r"(mul_tab_[A-Za-z0-9_]+)\s*\((.+)\)", expr)
        if call:
            args = split_commas(call.group(2))
            buf = args[0].strip()
            ctx.aliases[name] = buf
            rest = ", ".join(args[1:])
            return f"let {name}_base = {call.group(1)}({rest});"
        return stmt

    def try_decl(self, stmt, ctx):
        raw = stmt.strip()
        if not raw.endswith(";"):
            return None
        raw = raw[:-1].strip()
        if raw.startswith(("return", "if", "for", "while", "break", "continue")):
            return None
        # threadgroup array -> module scope
        m = re.match(r"(?:threadgroup\s+)?(f32|u32|i32)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[([^\]]+)\]$", raw)
        if m and "threadgroup" in stmt:
            length = const_int(m.group(3)) or m.group(3)
            ctx.wg_decls.append((m.group(2), m.group(1), length))
            return ""
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[([^\]]+)\]\s*=\s*\{(.*)\}$", raw, re.S)
        if m:
            ty, name, size, init = m.group(1), m.group(2), m.group(3), m.group(4)
            size_e = array_len(name, ty, const_int(size) or size)
            ctx.locals[name] = ("array", ty, size_e)
            vals = ", ".join(self.rewrite_calls(v.strip(), ctx) for v in split_commas(init))
            return f"var {name}: array<{ty}, {size_e}> = array<{ty}, {size_e}>({vals});"
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[([^\]]+)\]$", raw)
        if m:
            ty, name, size = m.group(1), m.group(2), m.group(3)
            size_e = array_len(name, ty, const_int(size) or size)
            ctx.locals[name] = ("array", ty, size_e)
            return f"var {name}: array<{ty}, {size_e}>;"
        # `ds x, y, z` or `ds x = a, y = b`
        m = re.match(r"((?:vec[234])<f32>|[A-Za-z_][A-Za-z0-9_]*)\s+(.+)$", raw)
        if m and m.group(1) in self.known_types(ctx):
            ty = m.group(1)
            parts = split_commas(m.group(2))
            lines = []
            for part in parts:
                pm = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=\s*(.+))?$", part.strip(), re.S)
                if not pm:
                    return None
                name = pm.group(1)
                init = pm.group(2)
                ctx.locals[name] = ("var", ty)
                if init is None:
                    lines.append(f"var {name}: {ty};")
                else:
                    lines.append(f"var {name}: {ty} = {self.rewrite_calls(init.strip(), ctx)};")
            return "\n".join(lines)
        return None

    def known_types(self, ctx):
        return self.struct_names | {"u32", "i32", "f32", "bool", "vec2<f32>", "vec3<f32>", "vec4<f32>"}

    def rewrite_acc(self, stmt):
        m = re.match(r"(acc_fma|acc_fnma|acc_add)\s*\((.*)\)\s*;?$", stmt.strip(), re.S)
        if not m:
            return stmt
        args = split_commas(m.group(2))
        if not args:
            return stmt
        lhs = args[0]
        return f"{lhs} = {m.group(1)}({', '.join(args)});"

    def rewrite_calls(self, expr, ctx):
        slots = []

        def park(text):
            slots.append(text)
            return f"#S{len(slots) - 1}#"

        guard = 0
        while guard < 10000:
            guard += 1
            found = find_innermost_call(expr)
            if not found:
                break
            i, j, name, args = found
            sig = self.sigs.get(name)
            skip = (
                name in {"u32", "i32", "f32", "bool", "vec2<f32>", "vec3<f32>", "vec4<f32>", "bitcast<f32>", "bitcast<u32>"}
                or name in self.struct_names
                or sig is None
                or len(sig) != len(args)
            )
            if skip:
                expr = expr[:i] + park(expr[i:j]) + expr[j:]
                continue
            lowered = [pass_arg(arg.strip(), kind, ctx) for (_, kind, _, _, _), arg in zip(sig, args)]
            expr = expr[:i] + park(name + "(" + ", ".join(lowered) + ")") + expr[j:]
        for _ in range(len(slots) + 1):
            expr = re.sub(r"#S(\d+)#", lambda m: slots[int(m.group(1))], expr)
        return expr


class Ctx:
    def __init__(self):
        self.globals = set()
        self.storage_ptrs = set()
        self.mut_ptrs = set()
        self.array_ptrs = set()
        self.wg_ptrs = set()
        self.aliases = {}
        self.locals = {}
        self.wg_decls = []
        self.sigs = {}
        self.struct_names = set()


def find_innermost_call(expr: str):
    best = None
    i = 0
    while i < len(expr):
        m = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*|vec2<f32>|vec3<f32>|vec4<f32>|bitcast<f32>|bitcast<u32>)\s*\(", expr[i:])
        if not m:
            break
        name = m.group(1)
        abs_i = i + m.start()
        paren = expr.find("(", abs_i)
        try:
            end = matching(expr, paren, "(", ")")
        except RuntimeError:
            break
        inner = expr[paren + 1 : end]
        if "(" not in inner and name != "\x00":
            args = split_commas(inner) if inner.strip() else []
            # Prefer the leftmost innermost. Return the first.
            return abs_i, end + 1, name, args
        i = paren + 1
    return best


def pass_arg(arg, kind, ctx):
    arg = arg.strip()
    if kind in {"mut", "array", "storage", "wg"}:
        if arg in ctx.storage_ptrs or arg in ctx.mut_ptrs or arg in ctx.array_ptrs or arg in ctx.wg_ptrs:
            return arg
        if arg in ctx.globals or arg in ctx.locals:
            return f"&{arg}"
        if arg.startswith("&"):
            return arg
        return f"&({arg})"
    # value
    if arg in ctx.mut_ptrs:
        return f"*{arg}"
    return arg


def rewrite_indexing(text, ctx):
    names = list(ctx.storage_ptrs | ctx.array_ptrs | ctx.wg_ptrs | ctx.mut_ptrs)
    # Longer names first so `params` doesn't eat a prefix. Word boundaries handle that.
    for name in sorted(names, key=len, reverse=True):
        if name in ctx.aliases:
            continue
        text = re.sub(rf"\b{name}\[", f"(*{name})[", text)
    for name, buf in ctx.aliases.items():
        if buf in ctx.storage_ptrs or buf in ctx.globals:
            dest = f"(*{buf})" if buf in ctx.storage_ptrs else buf
            text = re.sub(rf"\b{name}\[", f"{dest}[{name}_base + ", text)
    for name in list(ctx.mut_ptrs):
        text = re.sub(rf"\b{name}\.", f"(*{name}).", text)
    return text


def rewrite_mut_assign(stmt, ctx):
    m = re.match(r"(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=", stmt)
    if m and m.group(2) in ctx.mut_ptrs:
        return stmt[: m.start(2)] + "*" + stmt[m.start(2) :]
    return stmt


def rewrite_compound(stmt: str) -> str:
    m = re.match(r"(.*)\s*(\+=|-=|\*=|/=|\|=|&=|>>=|<<=)\s*(.+);$", stmt, re.S)
    if not m:
        return stmt
    lhs, op, rhs = m.group(1).strip(), m.group(2), m.group(3).strip()
    if any(tok in lhs for tok in ("==", "!=", "<=", ">=", "&&", "||")):
        return stmt
    sym = op[:-1]
    return f"{lhs} = {lhs} {sym} ({rhs});"


def rewrite_constructors(text: str) -> str:
    out = []
    i = 0
    while i < len(text):
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\{", text[i:])
        if m and m.group(1) not in {"if", "for", "while", "else", "struct", "switch"}:
            brace = i + len(m.group(1))
            try:
                end = matching(text, brace, "{", "}")
            except RuntimeError:
                out.append(text[i])
                i += 1
                continue
            inner = rewrite_constructors(text[brace + 1 : end])
            out.append(m.group(1) + "(" + inner + ")")
            i = end + 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def rewrite_ternary(stmt: str) -> str:
    guard = 0
    while guard < 20:
        guard += 1
        q = find_top(stmt, "?")
        if q < 0:
            return stmt
        c = find_top(stmt, ":", q + 1)
        if c < 0:
            return stmt
        # Condition starts after the previous '=' or 'return' at depth 0.
        start = 0
        depth = 0
        for i, ch in enumerate(stmt[:q]):
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif depth == 0 and ch == "=" and not stmt[i : i + 2] in {"==", "!="} and (i == 0 or stmt[i - 1] not in "!<>="):
                start = i + 1
            elif depth == 0 and stmt.startswith("return", i):
                start = i + len("return")
        cond = stmt[start:q].strip()
        a = stmt[q + 1 : c].strip()
        # b runs until depth-0 comma or end, but for a full statement it's the rest without ';'.
        rest = stmt[c + 1 :]
        semi = ""
        if rest.endswith(";"):
            rest = rest[:-1]
            semi = ";"
        b = rest.strip()
        if not any(op in cond for op in ("==", "!=", "<=", ">=", "<", ">", "&&", "||")):
            cond = f"({cond}) != 0"
        repl = f"select({b}, {a}, {cond})"
        stmt = stmt[:start] + " " + repl + semi
    return stmt


def find_top(text, token, start=0):
    depth = 0
    i = start
    while i < len(text):
        ch = text[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0 and text.startswith(token, i):
            if token == ":" and i + 1 < len(text) and text[i + 1] == ":":
                i += 2
                continue
            return i
        i += 1
    return -1


def lower_step(name, step, ty):
    one = "1u" if ty == "u32" else "1"
    step = step.strip()
    if step in {f"++{name}", f"{name}++"}:
        return f"{name} = {name} + {one}"
    if step in {f"--{name}", f"{name}--"}:
        return f"{name} = {name} - {one}"
    m = re.match(rf"{name}\s*(\+=|-=|\*=|>>=|<<=)\s*(.+)", step)
    if m:
        sym = m.group(1)[:-1]
        return f"{name} = {name} {sym} ({m.group(2).strip()})"
    return step


def literals(text: str) -> str:
    text = re.sub(r"\b(\d+\.\d+(?:[eE][+-]?\d+)?)f\b", r"\1", text)
    text = re.sub(r"\b(\d+)f\b", r"\1.0", text)
    return text


def types(text: str) -> str:
    text = text.replace("threadgroup_barrier(mem_flags::mem_threadgroup)", "workgroupBarrier()")
    text = re.sub(r"\bthreadgroup\s+", "", text)
    text = re.sub(r"\bthread\s+", "", text)
    text = re.sub(r"\bdevice\s+", "", text)
    text = re.sub(r"\bconstant\s+", "", text)
    text = re.sub(r"\bkernel\s+", "", text)
    text = re.sub(r"\binline\s+", "", text)
    text = re.sub(r"\bconst\s+", "", text)
    text = text.replace("float4", "vec4<f32>")
    text = text.replace("float3", "vec3<f32>")
    text = text.replace("float2", "vec2<f32>")
    text = re.sub(r"\buint3\b", "vec3<u32>", text)
    text = re.sub(r"\buint\b", "u32", text)
    text = re.sub(r"\bint\b", "i32", text)
    text = re.sub(r"\bfloat\b", "f32", text)
    text = re.sub(r"\bisfinite\s*\(", "isfinite_f(", text)
    text = re.sub(r"\bfabs\s*\(", "abs(", text)
    text = re.sub(r"\brint\s*\(", "round(", text)
    text = rewrite_constructors(text)
    # frexp(x, e) -> the WGSL struct. Call sites that need e are rewritten below.
    text = re.sub(
        r"f32\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*frexp\s*\(\s*([^,]+)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*;",
        r"let \1_fr = frexp(\2); let \1 = \1_fr.fract; let \3 = \1_fr.exp;",
        text,
    )
    return text


def emit_struct(name, body, jt=False):
    fields = []
    inner = body.strip()[1:-1]
    for stmt in iter_stmts(inner + (";" if not inner.strip().endswith(";") else "")):
        # struct bodies are declarations without always using the statement reader well.
        pass
    chunks = split_semis(inner)
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or chunk.startswith("static"):
            continue
        chunk = literals(chunk)
        # float2 name
        m = re.match(r"float2\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s*,\s*([A-Za-z_][A-Za-z0-9_]*))*", chunk)
        if m and name in HOST_STRUCTS:
            for part in split_commas(chunk.replace("float2", "", 1)):
                fields.append(f"    {part.strip()}: array<f32, 2>,")
            continue
        chunk_ty = types(chunk)
        m = re.match(r"([A-Za-z0-9_<>, ]+?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[([^\]]+)\]$", chunk_ty.strip())
        if m:
            size = const_int(m.group(3)) or m.group(3)
            fields.append(f"    {m.group(2)}: array<{m.group(1).strip()}, {size}>,")
            continue
        m = re.match(r"([A-Za-z0-9_<>, ]+?)\s+(.+)$", chunk_ty.strip())
        if not m:
            continue
        ty = m.group(1).strip()
        if name in HOST_STRUCTS and ty == "vec2<f32>":
            ty = "array<f32, 2>"
        for part in split_commas(m.group(2)):
            fname = part.strip().split("=")[0].strip()
            if fname:
                fields.append(f"    {fname}: {ty},")
    return "struct " + name + " {\n" + "\n".join(fields) + "\n}\n"


def emit_jt():
    out = []
    # Filled later.
    return out


def discover_wg(src: str):
    sizes = {"f32": "64", "u32": "64"}
    for m in re.finditer(r"threadgroup\s+float\s+[A-Za-z0-9_]+\s*\[([^\]]+)\]", src):
        sizes["f32"] = const_int(m.group(1)) or m.group(1)
    for m in re.finditer(r"threadgroup\s+uint\s+[A-Za-z0-9_]+\s*\[([^\]]+)\]", src):
        sizes["u32"] = const_int(m.group(1)) or m.group(1)
    return sizes


def binding_line(index, p):
    if p["kind"] == "value":
        return f"@group(0) @binding({index}) var<storage, read> {p['name']}: {p['ty']};"
    mode = "read" if p["access"] == "read" else "read_write"
    return f"@group(0) @binding({index}) var<storage, {mode}> {p['name']}: {p['ty']};"


def hoist_uint_consts(src: str):
    found = re.findall(r"constant\s+uint\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\d+)\s*;", src)
    src = re.sub(r"constant\s+uint\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*\d+\s*;", "", src)
    lines = "\n".join(f"const {name}: u32 = {val}u;" for name, val in found)
    return src, lines


def translate(src: str, tg: int | None) -> str:
    src = preprocess(src, tg)
    src, consts = hoist_uint_consts(src)
    wg_size = discover_wg(src)
    items = parse_items(src)
    prog = Prog(items)
    entries = prog.build()
    # Jt structs discovered during substitution.
    struct_text = []
    for n in sorted(prog.jt):
        struct_text.append(f"struct Jt_{n} {{\n    c: array<ds, {jlen(n)}>,\n}}\n")
    for item in prog.structs:
        struct_text.append(emit_struct(item["name"], item["body"]))
    structs_src = "\n".join(struct_text)
    lower = Lower(structs_src, wg_size)
    # Note signatures before bodies so calls resolve. Concrete list includes helpers.
    prepared = []
    for fn in prog.concrete:
        params = parse_params(fn["signature"])
        # mul_tab returns an offset and does not take the table pointer.
        if fn["name"].startswith("mul_tab"):
            params = [p for p in params if p["name"] != "tables"]
            fn = dict(fn)
            fn["ret"] = "u32"
        lower.note(fn["name"], params)
        prepared.append((fn, params))
    for fn in entries:
        params = parse_params(fn["signature"])
        lower.note(fn["name"], params)
    shared = [structs_src]
    entry_text = []
    for fn, params in prepared:
        text, _, _ = lower.lower_fn(fn["name"], fn["ret"], params, fn["body"], False)
        shared.append(text)
    for fn in entries:
        params = parse_params(fn["signature"])
        text, bindings, wg_decls = lower.lower_fn(fn["name"], fn["ret"], params, fn["body"], True)
        lines = [f"// ENTRY {fn['name']}"]
        seen = set()
        for nm, ty, length in wg_decls:
            if nm in seen:
                continue
            seen.add(nm)
            lines.append(f"var<workgroup> {nm}: array<{ty}, {length}>;")
        for index, p in enumerate(bindings):
            lines.append(binding_line(index, p))
        lines.append(text)
        lines.append(f"// END ENTRY {fn['name']}")
        entry_text.append("\n".join(lines))
    head = "// SHARED\n" + (consts + "\n" if consts else "")
    return head + "\n".join(shared) + "\n" + "\n".join(entry_text) + "\n"


def main():
    if len(sys.argv) < 3:
        print("usage: msl_to_wesl.py input.metal output.wesl [--tg N]", file=sys.stderr)
        return 2
    tg = None
    if "--tg" in sys.argv:
        tg = int(sys.argv[sys.argv.index("--tg") + 1])
    Path(sys.argv[2]).write_text(translate(Path(sys.argv[1]).read_text(), tg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
