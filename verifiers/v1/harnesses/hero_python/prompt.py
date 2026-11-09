"""Fixed tool-usage preamble for the HERO Python harness.

Task-level solving strategy belongs in the task system prompt (the GEPA-evolved part),
so this preamble only explains the tool.
"""

HERO_PYTHON_SYSTEM_PROMPT = """You have one tool, python_repl, which runs Python in a persistent kernel.
Variables, imports, and functions persist between calls.Only printed output and the final expression are shown, so print what you need.
Run shell commands with print(bash(command)).Use Python to calculate or verify the problem when useful, and the feasibility of a computed solution before reporting it.
If code raises an exception, fix it; earlier state remains available.
"""
