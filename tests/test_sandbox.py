from app.sandbox import run_code


def test_runs_simple_analysis_and_keeps_state():
    env = {"data": [{"cause": "db"}, {"cause": "tls"}, {"cause": "db"}]}
    first = run_code("counts = Counter(d['cause'] for d in data)\nprint(counts.most_common(1))", env)
    assert first["ok"] and "('db', 2)" in first["output"]
    second = run_code("print(sum(counts.values()))", env)  # variables survive between cells
    assert second["output"] == "3"


def test_helpers_are_callable():
    result = run_code("print(double(21))", {"double": lambda x: x * 2})
    assert result["output"] == "42"


def test_escapes_are_blocked():
    attempts = [
        "import os",
        "from subprocess import run",
        "open('/etc/passwd').read()",
        "().__class__.__bases__[0].__subclasses__()",
        "eval('1+1')",
        "getattr(print, '__globals__')",
        "__import__('os')",
        "class X: pass",
        "with x: pass",
        "print('{0.__globals__}'.format(print))",
    ]
    for code in attempts:
        result = run_code(code, {})
        assert not result["ok"], code


def test_infinite_loop_is_stopped():
    result = run_code("while True:\n    pass", {}, timeout_s=0.3)
    assert not result["ok"] and "TimeoutError" in result["error"]


def test_output_is_truncated():
    result = run_code("print('x' * 5000)", {})
    assert "truncated" in result["output"] and len(result["output"]) < 2100
