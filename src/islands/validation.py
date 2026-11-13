"""Deterministic checks before expensive model training."""

import ast

from .models import Candidate, ValidationResult


class CandidateValidator:
    """Validate generated Python without executing candidate code."""

    def validate(self, candidate: Candidate) -> ValidationResult:
        checks: dict[str, bool] = {}
        errors: list[str] = []

        try:
            tree = ast.parse(candidate.code, filename=f"{candidate.gene_id}.py")
            checks["syntax"] = True
        except SyntaxError as exc:
            checks["syntax"] = False
            errors.append(f"syntax error: {exc.msg} (line {exc.lineno})")
            return ValidationResult(False, checks, errors)

        classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        checks["model_class"] = "ExquisiteNetV2" in classes
        checks["optimizer_function"] = "get_optimizer" in functions
        checks["option_markers"] = "# --OPTION--" in candidate.code

        if not checks["model_class"]:
            errors.append("missing ExquisiteNetV2 class")
        if not checks["optimizer_function"]:
            errors.append("missing get_optimizer function")
        if not checks["option_markers"]:
            errors.append("missing # --OPTION-- markers")

        return ValidationResult(not errors, checks, errors)
