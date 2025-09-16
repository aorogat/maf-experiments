"""
tasks.py
--------
Defines the mapping between task names (CLI arguments) and their
corresponding LiteralMessagePassing subclasses.
"""

from multi_agent.agentsNetOriginalCode import LiteralMessagePassing as lmp

# Map from string name -> task class
TASKS = {
    "matching": lmp.Matching,
    "consensus": lmp.Consensus,
    "coloring": lmp.Coloring,
    "leader_election": lmp.LeaderElection,
    "vertex_cover": lmp.VertexCover,
}
