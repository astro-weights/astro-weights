"""A toy stand-in for BFCL's MathAPI (stateless), for tests only."""


class MathAPI:
    def add(self, a: float, b: float) -> dict[str, float]:
        """Add two numbers."""
        return {"result": a + b}
