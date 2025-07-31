"""
Traditional Deadlock Detection and Prevention Algorithms
Implementation of Banker's algorithm, wait-for graphs, and other classic methods.
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, deque
import logging

class BankersAlgorithm:
    """
    Implementation of Banker's Algorithm for deadlock avoidance.
    Determines if a resource allocation request can be safely granted.
    """

    def __init__(self, num_processes: int, num_resources: int):
        self.num_processes = num_processes
        self.num_resources = num_resources
        self.logger = logging.getLogger(__name__)

        # System state matrices
        self.allocation = np.zeros((num_processes, num_resources), dtype=int)
        self.max_need = np.zeros((num_processes, num_resources), dtype=int)
        self.available = np.zeros(num_resources, dtype=int)

    def set_allocation_matrix(self, allocation: List[List[int]]):
        """Set current allocation matrix."""
        self.allocation = np.array(allocation)

    def set_max_matrix(self, max_need: List[List[int]]):
        """Set maximum need matrix."""
        self.max_need = np.array(max_need)

    def set_available_resources(self, available: List[int]):
        """Set available resources vector."""
        self.available = np.array(available)

    def calculate_need_matrix(self) -> np.ndarray:
        """Calculate the need matrix (Max - Allocation)."""
        return self.max_need - self.allocation

    def is_safe_state(self) -> Tuple[bool, List[int]]:
        """
        Check if the current state is safe using Banker's algorithm.

        Returns:
            Tuple of (is_safe, safe_sequence)
        """
        need = self.calculate_need_matrix()
        work = self.available.copy()
        finish = np.zeros(self.num_processes, dtype=bool)
        safe_sequence = []

        # Find processes that can complete
        while len(safe_sequence) < self.num_processes:
            found = False

            for i in range(self.num_processes):
                if not finish[i] and np.all(need[i] <= work):
                    # Process i can be allocated resources
                    work += self.allocation[i]
                    finish[i] = True
                    safe_sequence.append(i)
                    found = True
                    break

            if not found:
                # No process can proceed - unsafe state
                return False, []

        return True, safe_sequence

    def request_resources(self, process_id: int, request: List[int]) -> bool:
        """
        Process a resource request using Banker's algorithm.

        Args:
            process_id: ID of requesting process
            request: List of requested resources

        Returns:
            True if request can be safely granted, False otherwise
        """
        request = np.array(request)
        need = self.calculate_need_matrix()

        # Check if request is valid
        if not np.all(request <= need[process_id]):
            self.logger.warning(f"Process {process_id} request exceeds need")
            return False

        if not np.all(request <= self.available):
            self.logger.warning(f"Process {process_id} request exceeds available")
            return False

        # Temporarily allocate resources
        old_allocation = self.allocation[process_id].copy()
        old_available = self.available.copy()

        self.allocation[process_id] += request
        self.available -= request

        # Check if resulting state is safe
        is_safe, safe_seq = self.is_safe_state()

        if is_safe:
            self.logger.info(f"Request granted for process {process_id}")
            return True
        else:
            # Rollback allocation
            self.allocation[process_id] = old_allocation
            self.available = old_available
            self.logger.warning(f"Request denied for process {process_id} - would cause unsafe state")
            return False

    def release_resources(self, process_id: int, release: List[int]):
        """Release resources from a process."""
        release = np.array(release)

        if np.all(release <= self.allocation[process_id]):
            self.allocation[process_id] -= release
            self.available += release
            self.logger.info(f"Resources released by process {process_id}")
        else:
            self.logger.error(f"Process {process_id} trying to release more than allocated")

    def get_system_state(self) -> Dict:
        """Get current system state for visualization."""
        need = self.calculate_need_matrix()
        is_safe, safe_seq = self.is_safe_state()

        return {
            'allocation': self.allocation.tolist(),
            'max_need': self.max_need.tolist(),
            'need': need.tolist(),
            'available': self.available.tolist(),
            'is_safe': is_safe,
            'safe_sequence': safe_seq
        }

class WaitForGraph:
    """
    Wait-for graph implementation for deadlock detection.
    Maintains and analyzes the wait-for relationships between processes.
    """

    def __init__(self):
        self.graph = defaultdict(set)
        self.logger = logging.getLogger(__name__)

    def add_edge(self, from_process: int, to_process: int):
        """Add wait-for edge from one process to another."""
        self.graph[from_process].add(to_process)
        self.logger.debug(f"Added wait-for edge: P{from_process} -> P{to_process}")

    def remove_edge(self, from_process: int, to_process: int):
        """Remove wait-for edge."""
        if to_process in self.graph[from_process]:
            self.graph[from_process].remove(to_process)
            self.logger.debug(f"Removed wait-for edge: P{from_process} -> P{to_process}")

    def remove_process(self, process_id: int):
        """Remove all edges involving a process."""
        # Remove outgoing edges
        if process_id in self.graph:
            del self.graph[process_id]

        # Remove incoming edges
        for source in list(self.graph.keys()):
            self.graph[source].discard(process_id)

    def detect_cycle(self) -> Optional[List[int]]:
        """
        Detect cycles in the wait-for graph using DFS.

        Returns:
            List of process IDs forming a cycle, or None if no cycle
        """
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: int) -> Optional[List[int]]:
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]

            if node in visited:
                return None

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.graph[node]:
                cycle = dfs(neighbor)
                if cycle:
                    return cycle

            rec_stack.remove(node)
            path.pop()
            return None

        # Check each node
        for node in self.graph:
            if node not in visited:
                cycle = dfs(node)
                if cycle:
                    self.logger.warning(f"Cycle detected: {cycle}")
                    return cycle[:-1]  # Remove duplicate last element

        return None

    def has_deadlock(self) -> bool:
        """Check if the graph contains a deadlock (cycle)."""
        return self.detect_cycle() is not None

    def get_strongly_connected_components(self) -> List[List[int]]:
        """Find strongly connected components using Tarjan's algorithm."""
        index_map = {}
        lowlink_map = {}
        index = [0]  # Use list for mutable reference
        stack = []
        on_stack = set()
        components = []

        def strongconnect(node: int):
            index_map[node] = index[0]
            lowlink_map[node] = index[0]
            index[0] += 1
            stack.append(node)
            on_stack.add(node)

            for neighbor in self.graph[node]:
                if neighbor not in index_map:
                    strongconnect(neighbor)
                    lowlink_map[node] = min(lowlink_map[node], lowlink_map[neighbor])
                elif neighbor in on_stack:
                    lowlink_map[node] = min(lowlink_map[node], index_map[neighbor])

            if lowlink_map[node] == index_map[node]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    component.append(w)
                    if w == node:
                        break
                components.append(component)

        for node in self.graph:
            if node not in index_map:
                strongconnect(node)

        return components

    def visualize_graph(self) -> str:
        """Generate string representation of the wait-for graph."""
        if not self.graph:
            return "Empty wait-for graph"

        lines = ["Wait-for Graph:"]
        for source, targets in self.graph.items():
            if targets:
                for target in targets:
                    lines.append(f"  P{source} -> P{target}")
            else:
                lines.append(f"  P{source} (no outgoing edges)")

        return "\n".join(lines)

    def get_graph_dict(self) -> Dict[int, List[int]]:
        """Get graph as dictionary for serialization."""
        return {k: list(v) for k, v in self.graph.items()}

def demonstrate_bankers_algorithm():
    """Demonstrate Banker's algorithm with example data."""
    print("=== Banker's Algorithm Demonstration ===")

    # Example system with 3 processes and 3 resource types
    banker = BankersAlgorithm(3, 3)

    # Set system state
    allocation = [[0, 1, 0], [2, 0, 0], [3, 0, 2]]
    max_need = [[7, 5, 3], [3, 2, 2], [9, 0, 2]]
    available = [3, 3, 2]

    banker.set_allocation_matrix(allocation)
    banker.set_max_matrix(max_need)
    banker.set_available_resources(available)

    # Check initial state
    is_safe, safe_seq = banker.is_safe_state()
    print(f"Initial state is safe: {is_safe}")
    if is_safe:
        print(f"Safe sequence: {safe_seq}")

    # Test resource request
    print("\nTesting resource request...")
    request = [1, 0, 2]
    can_grant = banker.request_resources(0, request)
    print(f"Can grant request {request} to process 0: {can_grant}")

def demonstrate_wait_for_graph():
    """Demonstrate wait-for graph cycle detection."""
    print("\n=== Wait-For Graph Demonstration ===")

    wfg = WaitForGraph()

    # Create a simple cycle: P0 -> P1 -> P2 -> P0
    wfg.add_edge(0, 1)
    wfg.add_edge(1, 2)
    wfg.add_edge(2, 0)

    print(wfg.visualize_graph())

    cycle = wfg.detect_cycle()
    if cycle:
        print(f"Cycle detected: {cycle}")
    else:
        print("No cycle detected")

if __name__ == "__main__":
    demonstrate_bankers_algorithm()
    demonstrate_wait_for_graph()
