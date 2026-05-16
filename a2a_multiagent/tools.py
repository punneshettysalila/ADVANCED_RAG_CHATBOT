"""
Tool layer for agentic execution.

Includes:
- calculator tool
- web search tool
- Python code execution tool
- light answer assembly helpers
"""

from __future__ import annotations

import ast
import io
import json
import math
import os
import re
import subprocess
import sys
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "data": self.data or {},
            "error": self.error,
        }


# ----------------------------------------------------------------------
# Calculator
# ----------------------------------------------------------------------

_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
}

_ALLOWED_NAMES = {
    "pi": math.pi,
    "e": math.e,
}

_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


def _safe_eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"Unknown name: {node.id}")
    if isinstance(node, ast.BinOp):
        op = type(node.op)
        if op not in _ALLOWED_BINOPS:
            raise ValueError("Operator not allowed")
        return _ALLOWED_BINOPS[op](_safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = type(node.op)
        if op not in _ALLOWED_UNARYOPS:
            raise ValueError("Unary operator not allowed")
        return _ALLOWED_UNARYOPS[op](_safe_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed")
        fn_name = node.func.id
        if fn_name not in _ALLOWED_FUNCTIONS:
            raise ValueError(f"Function not allowed: {fn_name}")
        args = [_safe_eval_node(arg) for arg in node.args]
        return _ALLOWED_FUNCTIONS[fn_name](*args)
    raise ValueError(f"Unsupported expression: {ast.dump(node, include_attributes=False)}")


def calculator(expression: str) -> ToolResult:
    expression = (expression or "").strip()
    if not expression:
        return ToolResult("calculator", False, "", error="Empty expression")
    try:
        cleaned = expression.replace("^", "**")
        parsed = ast.parse(cleaned, mode="eval")
        value = _safe_eval_node(parsed)
        return ToolResult(
            "calculator",
            True,
            str(value),
            data={"expression": expression, "result": value},
        )
    except Exception as exc:
        return ToolResult("calculator", False, "", error=str(exc))


# ----------------------------------------------------------------------
# Web search
# ----------------------------------------------------------------------

def web_search(query: str, limit: int = 5) -> ToolResult:
    query = (query or "").strip()
    if not query:
        return ToolResult("web_search", False, "", error="Empty query")

    try:
        response = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        response.raise_for_status()
        html = response.text
        pattern = re.compile(
            r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>.*?'
            r'<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        results: List[Dict[str, str]] = []
        for match in pattern.finditer(html):
            title = re.sub(r"<.*?>", "", match.group("title"))
            snippet = re.sub(r"<.*?>", "", match.group("snippet"))
            href = match.group("href")
            results.append({"title": title.strip(), "snippet": snippet.strip(), "url": href.strip()})
            if len(results) >= limit:
                break
        if not results:
            return ToolResult("web_search", False, "", error="No search results parsed")
        lines = []
        for idx, item in enumerate(results, start=1):
            lines.append(f"{idx}. {item['title']}\n   {item['snippet']}\n   {item['url']}")
        return ToolResult("web_search", True, "\n".join(lines), data={"results": results})
    except Exception as exc:
        return ToolResult("web_search", False, "", error=str(exc))


# ----------------------------------------------------------------------
# Code execution
# ----------------------------------------------------------------------

def execute_python_code(code: str, timeout: int = 8) -> ToolResult:
    code = (code or "").strip()
    if not code:
        return ToolResult("code_execution", False, "", error="Empty code")
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        return ToolResult(
            "code_execution",
            completed.returncode == 0,
            output.strip(),
            data={"returncode": completed.returncode},
            error=None if completed.returncode == 0 else f"Process exited with {completed.returncode}",
        )
    except subprocess.TimeoutExpired:
        return ToolResult("code_execution", False, "", error=f"Code execution timed out after {timeout} seconds")
    except Exception as exc:
        return ToolResult("code_execution", False, "", error=str(exc))


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def format_tool_context(results: List[ToolResult]) -> str:
    if not results:
        return ""
    lines = ["Tool results:"]
    for result in results:
        if result.success:
            lines.append(f"- {result.tool_name}: {result.output}")
        else:
            lines.append(f"- {result.tool_name}: ERROR {result.error or 'unknown'}")
    return "\n".join(lines)


def extract_code_block(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()
