"""assertsql: counterexample-guided text-to-SQL repair with synthesized semantic assertions.

Test-driven development for SQL: synthesize checkable assertions from the
natural-language question, execute candidates in a read-only sandbox, and
repair them using violated assertions as counterexamples. No LLM anywhere.
"""

from .ir import QueryIR, SelectItem, Condition, Join, render
from .schema import Schema, Table, Column, load_schema
from .assertions import Assertion, Violation, synthesize, assertions_from_gold
from .generator import generate
from .sandbox import Sandbox, SandboxResult
from .repair import repair, RepairTrace, RepairStep
from .plandiff import diff_plans

__version__ = "0.1.0"
__all__ = [
    "QueryIR", "SelectItem", "Condition", "Join", "render",
    "Schema", "Table", "Column", "load_schema",
    "Assertion", "Violation", "synthesize", "assertions_from_gold",
    "generate", "Sandbox", "SandboxResult",
    "repair", "RepairTrace", "RepairStep", "diff_plans",
]
