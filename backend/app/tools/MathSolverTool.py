import re
from sympy import sympify, solve, simplify

class MathSolverTool:
    VALID_CHARS = r'^[0-9a-zA-Z+\-*/^().=; ]+$'

    def __init__(self, safe_mode: bool = True):
        self.safe_mode = safe_mode

    def _validate_input(self, equation: str) -> None:
        if self.safe_mode and not re.match(self.VALID_CHARS, equation):
            raise ValueError("Invalid characters in expression.")
        
    def parse_equation(self,equation:str):
        parts = [p.strip() for p in equation.split(';')]
        exprs = []
        for p in parts:
            if '=' in p:
                lhs, rhs = p.split('=', 1)
                exprs.append(sympify(lhs) - sympify(rhs))
            else:
                exprs.append(sympify(p))
        return exprs
    
    def solve(self,equation:str):
        try:
            self._validate_input(equation)
            exprs = self.parse_equation(equation)
            if len(exprs) == 1:
                expr = exprs[0]
                if not expr.free_symbols and "=" not in equation:
                    return {"type": "simplified", "result": str(simplify(expr))}
            all_symbols = sorted(set().union(*[e.free_symbols for e in exprs]))
            result = solve(exprs, list(all_symbols) or None)
            return {
                "type": "solution",
                "variables": [str(s) for s in all_symbols],
                "result": str(result),
            }
        except Exception as e:
            return {"type": "error", "message": str(e)}
