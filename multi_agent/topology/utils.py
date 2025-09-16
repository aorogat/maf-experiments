"""
utils.py
--------
General helper utilities for the topology multi-agent experiments.
"""

import subprocess
import time
import contextlib

import networkx as nx




def determine_rounds(task: str, graph: nx.Graph, num_sample: int, num_samples: int, rounds: int) -> int:
    """
    Determine how many rounds to run based on the task and graph size.
    - For consensus/leader election or large graphs: 2 * diameter + 1
    - Otherwise, return the provided rounds
    """
    if task in ["consensus", "leader_election"] or graph.number_of_nodes() > 16:
        return 2 * nx.diameter(graph) + 1
    else:
        return rounds



def get_git_commit_hash() -> str:
    """
    Get the current git commit hash of the repository.

    Returns
    -------
    str
        The commit hash, or "None" if git is unavailable.
    """
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        return commit_hash
    except Exception:
        return "None"



@contextlib.contextmanager
def timer():
    """
    Context manager to measure elapsed time.

    Example
    -------
    >>> with timer() as t:
    ...     do_something()
    >>> print(t.elapsed)
    0.532  # seconds
    """
    start = time.time()
    class _Timer:
        elapsed = 0
    t = _Timer()
    try:
        yield t
    finally:
        t.elapsed = time.time() - start


def format_seconds(seconds: float) -> str:
    """
    Nicely format a runtime in seconds into human-readable string.

    Parameters
    ----------
    seconds : float
        Runtime in seconds.

    Returns
    -------
    str
        Formatted string like '12.3s' or '1m 2.4s'.
    """
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, sec = divmod(seconds, 60)
    return f"{int(minutes)}m {sec:.1f}s"
