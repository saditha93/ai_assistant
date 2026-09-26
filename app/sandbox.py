import ast
import statistics
import sys
import time
from collections import Counter, defaultdict

ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Name, ast.Load, ast.Store,
    ast.Call, ast.keyword, ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Starred,
    ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp, ast.comprehension,
    ast.For, ast.While, ast.If, ast.IfExp, ast.Break, ast.Continue, ast.Pass,
    ast.Compare, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Subscript, ast.Slice, ast.Attribute,
    ast.JoinedStr, ast.FormattedValue, ast.Lambda, ast.arguments, ast.arg,
    ast.FunctionDef, ast.Return,
    ast.operator, ast.cmpop, ast.boolop, ast.unaryop,
)

SAFE_BUILTINS = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "len", "range", "enumerate", "zip", "sorted", "reversed", "min", "max", "sum", "abs", "round",
        "any", "all", "list", "dict", "set", "tuple", "str", "int", "float", "bool", "isinstance", "map",
        "filter", "next", "iter", "chr", "ord", "repr", "divmod", "True", "False", "None",
    )
}
SAFE_BUILTINS.update(Counter=Counter, defaultdict=defaultdict, mean=statistics.mean, median=statistics.median)

BLOCKED_ATTRS = {"format", "format_map"}

MAX_OUTPUT_CHARS = 2000


class SandboxError(Exception):
    pass


def check_code(code: str) -> ast.Module:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SandboxError(f"Syntax error: {exc.msg} (line {exc.lineno})") from exc
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise SandboxError(f"'{type(node).__name__.lower()}' statements/expressions are not allowed here")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in BLOCKED_ATTRS):
            raise SandboxError(f"Access to '{node.attr}' is not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise SandboxError(f"Name '{node.id}' is not allowed")
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            raise SandboxError("Decorators are not allowed")
    return tree


def run_code(code: str, variables: dict, timeout_s: float = 20) -> dict:
    """Execute `code` with `variables` as its globals. The dict is updated in place, so a
    caller can run several cells that share state, like a notebook.

    Returns {"ok", "output", "error"}. Must be called from the thread that should run it
    (we call it via asyncio.to_thread), because the deadline uses sys.settrace.
    """
    output: list[str] = []

    def _print(*args, **_):
        output.append(" ".join(str(a) for a in args))

    try:
        tree = check_code(code)
    except SandboxError as exc:
        return {"ok": False, "output": "", "error": str(exc)}

    deadline = time.monotonic() + timeout_s

    def tracer(frame, event, arg):
        if time.monotonic() > deadline:
            raise TimeoutError(f"code ran longer than {timeout_s:.0f}s")
        return tracer

    variables["__builtins__"] = {**SAFE_BUILTINS, "print": _print}
    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        exec(compile(tree, "<sandbox>", "exec"), variables)
        error = None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        sys.settrace(previous)
        variables.pop("__builtins__", None)

    text = "\n".join(output)
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS] + f"\n... [output truncated, {len(text)} chars total]"
    return {"ok": error is None, "output": text, "error": error}
