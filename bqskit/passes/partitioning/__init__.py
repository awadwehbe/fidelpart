"""
This package implements partitioning passes.

Partitioning passes group together adjacent gates into CircuitGates objects.
"""
from __future__ import annotations

from bqskit.passes.partitioning.cluster import ClusteringPartitioner
from bqskit.passes.partitioning.greedy import GreedyPartitioner
from bqskit.passes.partitioning.quick import QuickPartitioner
from bqskit.passes.partitioning.scan import ScanPartitioner
from bqskit.passes.partitioning.single import GroupSingleQuditGatePass
#from .hypergraphpartition import EnhancedHypergraphPartitionPass
from bqskit.passes.partitioning.hyp import EnhancedHypergraphPartitionPass

__all__ = [
    'ClusteringPartitioner',
    'GreedyPartitioner',
    'ScanPartitioner',
    'QuickPartitioner',
    'GroupSingleQuditGatePass',
    #
    'EnhancedHypergraphPartitionPass'
]
