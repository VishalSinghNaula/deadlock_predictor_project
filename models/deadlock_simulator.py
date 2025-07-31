"""
Real-time Deadlock Simulation Engine
Simulates OS-level thread execution with resource allocation and deadlock detection.
"""

import threading
import time
import random
import logging
from typing import List, Dict, Set, Optional, Callable, Any
from enum import Enum
from collections import defaultdict, deque
import queue

class ThreadState(Enum):
    """Thread execution states."""
    READY = "READY"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    TERMINATED = "TERMINATED"

class ResourceType(Enum):
    """Types of system resources."""
    MUTEX = "MUTEX"
    SEMAPHORE = "SEMAPHORE"
    FILE = "FILE"
    NETWORK = "NETWORK"
    MEMORY = "MEMORY"

class SimulatedThread:
    """Represents a simulated thread in the system."""

    def __init__(self, thread_id: int, cpu_burst: int):
        self.thread_id = thread_id
        self.cpu_burst = cpu_burst  # milliseconds
        self.state = ThreadState.READY
        self.held_resources = set()
        self.waiting_for = None
        self.start_time = None
        self.end_time = None
        self.total_wait_time = 0
        self.last_state_change = time.time()

    def change_state(self, new_state: ThreadState):
        """Change thread state and update timing."""
        current_time = time.time()
        if self.state == ThreadState.WAITING:
            self.total_wait_time += current_time - self.last_state_change

        self.state = new_state
        self.last_state_change = current_time

    def __str__(self):
        return f"T{self.thread_id}[{self.state.value}]"

    def __repr__(self):
        return self.__str__()

class Resource:
    """Represents a system resource that can cause contention."""

    def __init__(self, resource_id: int, resource_type: ResourceType):
        self.resource_id = resource_id
        self.resource_type = resource_type
        self.owner = None
        self.waiting_queue = deque()
        self.lock = threading.Lock()

    def request(self, thread: SimulatedThread) -> bool:
        """
        Request resource access.

        Returns:
            True if resource acquired, False if must wait
        """
        with self.lock:
            if self.owner is None:
                self.owner = thread
                thread.held_resources.add(self)
                return True
            else:
                self.waiting_queue.append(thread)
                thread.waiting_for = self
                return False

    def release(self, thread: SimulatedThread) -> Optional[SimulatedThread]:
        """
        Release resource and wake next waiting thread.

        Returns:
            Next thread to be woken up, if any
        """
        with self.lock:
            if self.owner == thread:
                thread.held_resources.discard(self)
                self.owner = None

                if self.waiting_queue:
                    next_thread = self.waiting_queue.popleft()
                    next_thread.waiting_for = None
                    self.owner = next_thread
                    next_thread.held_resources.add(self)
                    return next_thread

        return None

    def __str__(self):
        return f"R{self.resource_id}[{self.resource_type.value}]"

    def __repr__(self):
        return self.__str__()

class DeadlockDetector:
    """Detects deadlocks using wait-for graph analysis."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def build_wait_for_graph(self, threads: List[SimulatedThread]) -> Dict[int, Set[int]]:
        """
        Build wait-for graph from current thread states.

        Returns:
            Dictionary mapping thread_id -> set of thread_ids it's waiting for
        """
        wait_graph = defaultdict(set)

        for thread in threads:
            if thread.state == ThreadState.WAITING and thread.waiting_for:
                resource = thread.waiting_for
                if resource.owner:
                    wait_graph[thread.thread_id].add(resource.owner.thread_id)

        return dict(wait_graph)

    def detect_cycle(self, wait_graph: Dict[int, Set[int]]) -> Optional[List[int]]:
        """
        Detect cycles in wait-for graph using DFS.

        Returns:
            List of thread IDs forming the cycle, or None if no cycle
        """
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: int) -> Optional[List[int]]:
            if node in rec_stack:
                # Found cycle - extract it
                cycle_start = path.index(node)
                return path[cycle_start:] + [node]

            if node in visited:
                return None

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in wait_graph.get(node, set()):
                cycle = dfs(neighbor)
                if cycle:
                    return cycle

            rec_stack.remove(node)
            path.pop()
            return None

        for node in wait_graph:
            if node not in visited:
                cycle = dfs(node)
                if cycle:
                    return cycle

        return None

    def check_deadlock(self, threads: List[SimulatedThread]) -> Optional[Dict[str, Any]]:
        """
        Check for deadlock in current system state.

        Returns:
            Deadlock information if detected, None otherwise
        """
        wait_graph = self.build_wait_for_graph(threads)

        if not wait_graph:
            return None

        cycle = self.detect_cycle(wait_graph)

        if cycle:
            # Extract deadlocked threads and resources
            deadlocked_threads = []
            involved_resources = set()

            for tid in cycle[:-1]:  # Remove duplicate last element
                thread = next(t for t in threads if t.thread_id == tid)
                deadlocked_threads.append(thread)
                if thread.waiting_for:
                    involved_resources.add(thread.waiting_for)

            return {
                'cycle': cycle[:-1],
                'threads': deadlocked_threads,
                'resources': list(involved_resources),
                'wait_graph': wait_graph
            }

        return None

class DeadlockSimulator:
    """
    Main deadlock simulation engine.
    Simulates concurrent thread execution with realistic resource contention.
    """

    def __init__(self, thread_count: int, burst_times: List[int]):
        self.thread_count = thread_count
        self.burst_times = burst_times

        self.logger = logging.getLogger(__name__)
        self.detector = DeadlockDetector()

        # Simulation state
        self.threads = []
        self.resources = []
        self.running = False
        self.start_time = None
        self.deadlock_detected = False
        self.deadlock_info = None

        # Configuration
        self.num_resources = 5
        self.time_quantum = 0.001  # 1ms time slices
        self.max_simulation_time = 10.0  # 10 seconds max

        # Initialize components
        self._create_threads()
        self._create_resources()

    def _create_threads(self):
        """Create simulated threads with specified burst times."""
        for i in range(self.thread_count):
            burst_time = self.burst_times[i] if i < len(self.burst_times) else 100
            thread = SimulatedThread(i, burst_time)
            self.threads.append(thread)

        self.logger.info(f"Created {len(self.threads)} threads")

    def _create_resources(self):
        """Create system resources that threads will contend for."""
        resource_types = list(ResourceType)

        for i in range(self.num_resources):
            resource_type = resource_types[i % len(resource_types)]
            resource = Resource(i, resource_type)
            self.resources.append(resource)

        self.logger.info(f"Created {len(self.resources)} resources")

    def _simulate_thread_execution(self, thread: SimulatedThread, 
                                 progress_callback: Optional[Callable] = None):
        """
        Simulate individual thread execution with resource requests.

        Args:
            thread: Thread to simulate
            progress_callback: Optional callback for progress updates
        """
        try:
            thread.start_time = time.time()
            thread.change_state(ThreadState.RUNNING)

            if progress_callback:
                progress_callback(f"Thread T{thread.thread_id} started")

            execution_time = 0
            burst_time_ms = thread.cpu_burst

            while execution_time < burst_time_ms and self.running:
                # Simulate work in small time slices
                slice_time = min(self.time_quantum * 1000, burst_time_ms - execution_time)
                time.sleep(slice_time / 1000.0)  # Convert to seconds
                execution_time += slice_time

                # Randomly request resources during execution
                if random.random() < 0.3:  # 30% chance per time slice
                    self._request_random_resources(thread, progress_callback)

                # Check if we need to wait for resources
                if thread.state == ThreadState.WAITING:
                    while thread.state == ThreadState.WAITING and self.running:
                        time.sleep(self.time_quantum)

                        # Check for deadlock periodically
                        if random.random() < 0.1:  # 10% chance per time slice
                            deadlock = self.detector.check_deadlock(self.threads)
                            if deadlock:
                                self.deadlock_detected = True
                                self.deadlock_info = deadlock
                                if progress_callback:
                                    progress_callback(f"🚨 DEADLOCK DETECTED involving threads: {[t.thread_id for t in deadlock['threads']]}")
                                return

                    # Resume running after wait
                    if self.running:
                        thread.change_state(ThreadState.RUNNING)

                # Randomly release some resources
                if random.random() < 0.2:  # 20% chance per time slice
                    self._release_random_resources(thread, progress_callback)

            # Thread finished execution
            self._release_all_resources(thread)
            thread.change_state(ThreadState.TERMINATED)
            thread.end_time = time.time()

            if progress_callback:
                progress_time = thread.end_time - thread.start_time
                progress_callback(f"Thread T{thread.thread_id} completed in {progress_time:.2f}s")

        except Exception as e:
            self.logger.error(f"Thread T{thread.thread_id} simulation error: {str(e)}")
            thread.change_state(ThreadState.TERMINATED)

    def _request_random_resources(self, thread: SimulatedThread, 
                                progress_callback: Optional[Callable] = None):
        """Request random resources for a thread."""
        if len(thread.held_resources) >= 3:  # Limit resource holding
            return

        available_resources = [r for r in self.resources if r not in thread.held_resources]
        if not available_resources:
            return

        resource = random.choice(available_resources)
        acquired = resource.request(thread)

        if acquired:
            if progress_callback:
                progress_callback(f"T{thread.thread_id} acquired {resource}")
        else:
            thread.change_state(ThreadState.WAITING)
            if progress_callback:
                progress_callback(f"T{thread.thread_id} waiting for {resource} (held by T{resource.owner.thread_id})")

    def _release_random_resources(self, thread: SimulatedThread,
                                progress_callback: Optional[Callable] = None):
        """Release random resources held by a thread."""
        if not thread.held_resources:
            return

        resource = random.choice(list(thread.held_resources))
        woken_thread = resource.release(thread)

        if progress_callback:
            progress_callback(f"T{thread.thread_id} released {resource}")
            if woken_thread:
                progress_callback(f"T{woken_thread.thread_id} woken up")
                woken_thread.change_state(ThreadState.READY)

    def _release_all_resources(self, thread: SimulatedThread):
        """Release all resources held by a thread."""
        for resource in list(thread.held_resources):
            woken_thread = resource.release(thread)
            if woken_thread:
                woken_thread.change_state(ThreadState.READY)

    def run_simulation(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run the complete deadlock simulation.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            Simulation results dictionary
        """
        self.running = True
        self.start_time = time.time()
        self.deadlock_detected = False
        self.deadlock_info = None

        if progress_callback:
            progress_callback(f"Starting simulation with {self.thread_count} threads")

        # Create and start thread simulators
        thread_simulators = []
        for thread in self.threads:
            simulator = threading.Thread(
                target=self._simulate_thread_execution,
                args=(thread, progress_callback),
                daemon=True
            )
            thread_simulators.append(simulator)
            simulator.start()

        # Monitor simulation progress
        monitor_thread = threading.Thread(
            target=self._monitor_simulation,
            args=(progress_callback,),
            daemon=True
        )
        monitor_thread.start()

        # Wait for completion or timeout
        start_wait = time.time()
        while self.running and (time.time() - start_wait) < self.max_simulation_time:
            time.sleep(0.1)

            # Check if all threads completed
            if all(t.state == ThreadState.TERMINATED for t in self.threads):
                break

            # Check for deadlock
            if self.deadlock_detected:
                break

        # Stop simulation
        self.running = False

        # Wait for threads to finish
        for simulator in thread_simulators:
            simulator.join(timeout=1.0)

        # Compile results
        return self._compile_results()

    def _monitor_simulation(self, progress_callback: Optional[Callable] = None):
        """Monitor simulation and detect deadlocks."""
        last_check = time.time()

        while self.running:
            time.sleep(0.1)

            current_time = time.time()
            if current_time - last_check >= 0.5:  # Check every 500ms
                deadlock = self.detector.check_deadlock(self.threads)
                if deadlock:
                    self.deadlock_detected = True
                    self.deadlock_info = deadlock
                    if progress_callback:
                        progress_callback("🚨 DEADLOCK DETECTED by monitor!")
                    break

                last_check = current_time

    def _compile_results(self) -> Dict[str, Any]:
        """Compile simulation results."""
        end_time = time.time()
        total_time = end_time - self.start_time if self.start_time else 0

        results = {
            'total_simulation_time': total_time,
            'deadlock_detected': self.deadlock_detected,
            'deadlock_time': None,
            'deadlocked_threads': [],
            'thread_stats': [],
            'resource_stats': []
        }

        if self.deadlock_detected and self.deadlock_info:
            results['deadlock_time'] = total_time
            results['deadlocked_threads'] = [f"T{t.thread_id}" for t in self.deadlock_info['threads']]

        # Thread statistics
        for thread in self.threads:
            thread_stat = {
                'thread_id': thread.thread_id,
                'state': thread.state.value,
                'cpu_burst': thread.cpu_burst,
                'total_wait_time': thread.total_wait_time,
                'completed': thread.state == ThreadState.TERMINATED
            }
            results['thread_stats'].append(thread_stat)

        # Resource statistics
        for resource in self.resources:
            resource_stat = {
                'resource_id': resource.resource_id,
                'type': resource.resource_type.value,
                'owner': resource.owner.thread_id if resource.owner else None,
                'waiting_count': len(resource.waiting_queue)
            }
            results['resource_stats'].append(resource_stat)

        return results

    def stop(self):
        """Stop the running simulation."""
        self.running = False
        self.logger.info("Simulation stopped by user")

    def get_current_state(self) -> Dict[str, Any]:
        """Get current simulation state for visualization."""
        return {
            'threads': [(t.thread_id, t.state.value, len(t.held_resources)) for t in self.threads],
            'resources': [(r.resource_id, r.owner.thread_id if r.owner else None, len(r.waiting_queue)) for r in self.resources],
            'wait_graph': self.detector.build_wait_for_graph(self.threads) if self.threads else {}
        }

def main():
    """Main function for testing the simulator."""
    import argparse

    parser = argparse.ArgumentParser(description='Deadlock Simulator')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads')
    parser.add_argument('--bursts', type=str, default='100 150 200 120', 
                       help='CPU burst times (ms)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

    # Parse burst times
    burst_times = list(map(int, args.bursts.split()))

    print(f"\nDeadlock Simulation Test")
    print(f"Threads: {args.threads}")
    print(f"Burst times: {burst_times}")
    print("-" * 50)

    # Create and run simulator
    simulator = DeadlockSimulator(args.threads, burst_times)

    def progress_callback(message):
        print(f"[SIM] {message}")

    results = simulator.run_simulation(progress_callback)

    # Display results
    print("\n" + "=" * 50)
    print("SIMULATION RESULTS")
    print("=" * 50)

    print(f"Total time: {results['total_simulation_time']:.2f}s")
    print(f"Deadlock detected: {results['deadlock_detected']}")

    if results['deadlock_detected']:
        print(f"Deadlock time: {results['deadlock_time']:.2f}s")
        print(f"Deadlocked threads: {', '.join(results['deadlocked_threads'])}")

    print("\nThread Statistics:")
    for stat in results['thread_stats']:
        print(f"  T{stat['thread_id']}: {stat['state']} - Wait: {stat['total_wait_time']:.2f}s")

    print("\nResource Statistics:")
    for stat in results['resource_stats']:
        owner = f"T{stat['owner']}" if stat['owner'] is not None else "None"
        print(f"  R{stat['resource_id']} ({stat['type']}): Owner={owner}, Waiting={stat['waiting_count']}")

if __name__ == "__main__":
    main()
