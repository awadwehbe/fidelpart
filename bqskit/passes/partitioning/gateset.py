# gateset.py
from bqskit.ir.gate import Gate
from typing import Iterator, Optional

class CustomGateSet:
    """A custom gate set class to mimic BQSKit's internal GateSet."""
    def __init__(self, gates: Optional[set[Gate]] = None):
        """
        Initialize the gate set with a set of gate instances.

        Args:
            gates (set[Gate], optional): Set of gate instances. Defaults to empty set.
        """
        self.gates = gates if gates is not None else set()

    def __iter__(self) -> Iterator[Gate]:
        """Make the gate set iterable, as BQSKit might expect."""
        return iter(self.gates)

    def __contains__(self, gate: Gate) -> bool:
        """Check if a gate is in the set."""
        return gate in self.gates

    def add(self, gate: Gate) -> None:
        """Add a gate to the set."""
        self.gates.add(gate)

    def __len__(self) -> int:
        """Return the number of gates."""
        return len(self.gates)