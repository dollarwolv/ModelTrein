import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix


class ChinesePostmanProblem:
    """
    Implementation of the Chinese Postman Problem for multiple trains.
    
    CHINESE POSTMAN PROBLEM EXPLAINED:
    Imagine a postman who needs to deliver mail on every street at least once and 
    return to the post office. The Chinese Postman Problem asks: What's the shortest 
    route that covers every street?
    
    In our case:
    - Streets = railway connections between stations
    - Postman = trains
    - We have multiple postmen (trains) with time limits
    
    THE ALGORITHM STEPS:
    1. Check if the graph is "Eulerian" (can traverse all edges exactly once), it is not
    2. If not, find which edges to duplicate to make it Eulerian
    3. Find a route that covers all edges
    4. Split the route into multiple trains respecting time limits
    
    OPTIMIZATION NOTE:
    Since we know this railway network is NOT Eulerian, we've removed:
    - The check for Eulerian graphs in the run() method
    - The conditional logic for handling already-Eulerian graphs
    - The vertex degree dictionary (we only need the odd vertices list)
    This streamlines the algorithm and removes unnecessary computations.
    """
    
    def __init__(self, stations, connections, station_dict, max_trains=7, max_minutes=120):
        """
        Initialize the Chinese Postman Problem solver.
        
        Parameters:
        -----------
        stations: List of Station objects representing train stations
        connections: List of Connection objects representing railway connections
        station_dict: Dictionary mapping station names to Station objects
        max_trains: Maximum number of trains we can use (default: 7)
        max_minutes: Maximum minutes each train can operate (default: 120)
        """
        self.stations = stations
        self.connections = connections
        self.station_dict = station_dict
        self.max_trains = max_trains
        self.max_minutes = max_minutes
        
        # Create mappings between station names and matrix indices
        self.station_names = [s.name for s in stations]
        
        # Keys are station names, values are index of station in the matrix
        self.station_to_index = {name: i for i, name in enumerate(self.station_names)}
        
        # Keys are indices, values are station names
        self.index_to_station = {i: name for name, i in self.station_to_index.items()}
        
        # Number of stations (int)
        self.num_of_stations = len(self.station_names)
        print(f"Initializing CPP with {self.num_of_stations} stations and {len(connections)} connections")
        
        # Build the main distance matrix and edge count
        self._build_distance_matrix()
        
        # Track original edges for coverage calculation
        self.original_edges = set()
        for connection in connections:
            edge = tuple(sorted([connection.station_1, connection.station_2]))
            self.original_edges.add(edge)
    
    def _build_distance_matrix(self):
        """
        Build the distance matrix representation of the graph.
        
        WHY USE A DISTANCE MATRIX?
        The distance matrix is the core data structure that represents our railway network.
        - distance_matrix[i][j] = travel time from station i to station j
        - distance_matrix[i][j] = np.inf means no direct connection
        
        From this single matrix, we can derive:
        - Adjacency information (connected if distance != infinity)
        - Shortest paths between all stations
        - Edge counts (by tracking duplicates during construction)
        
        This approach is more memory-efficient than maintaining multiple matrices.
        """
        # Initialize distance matrix with infinity (no connections)
        self.distance_matrix = np.full((self.num_of_stations, self.num_of_stations), np.inf)
        
        # Set diagonal to 0 (distance from station to itself)
        np.fill_diagonal(self.distance_matrix, 0)
        
        # Edge count dictionary to handle potential parallel edges
        # Format: {(station1_idx, station2_idx): count}
        self.edge_counts = {}
        
        # Fill matrix based on connections
        for connection in self.connections:
            # Get indices for the two stations
            i = self.station_to_index[connection.station_1]
            j = self.station_to_index[connection.station_2]
            
            # For parallel edges, we keep the minimum travel time
            # This is because in practice, trains would use the fastest route
            if self.distance_matrix[i][j] == np.inf:
                # First connection between these stations
                self.distance_matrix[i][j] = connection.travel_time
                self.distance_matrix[j][i] = connection.travel_time
                self.edge_counts[(min(i,j), max(i,j))] = 1
            else:
                # Parallel edge detected! Keep the faster connection
                if connection.travel_time < self.distance_matrix[i][j]:
                    self.distance_matrix[i][j] = connection.travel_time
                    self.distance_matrix[j][i] = connection.travel_time
                self.edge_counts[(min(i,j), max(i,j))] += 1
        
        print(f"Distance matrix built: {len(self.edge_counts)} unique connections")
        if any(count > 1 for count in self.edge_counts.values()):
            print(f"  Note: Found parallel edges in the network")
    
    def get_adjacency_from_distance(self):
        """
        Derive adjacency information from the distance matrix.
        
        This demonstrates how we can get adjacency information without storing
        a separate matrix. A connection exists if distance != infinity.
        
        Returns:
        --------
        adjacency: np.ndarray
            Binary matrix where 1 indicates a connection exists
        """
        # Create adjacency matrix: 1 where distance is not infinity, 0 otherwise
        adjacency = (self.distance_matrix != np.inf).astype(int)
        # Don't count self-loops
        np.fill_diagonal(adjacency, 0)
        return adjacency
    
    def find_odd_degree_vertices(self):
        """
        Find all vertices with odd degree - these need to be matched for augmentation.
        
        Since we know the graph is NOT Eulerian, we will definitely find odd-degree vertices.
        In any graph, the number of odd-degree vertices is always even (handshaking lemma).
        
        WHY DO WE NEED TO FIND ODD VERTICES?
        The Chinese Postman algorithm requires duplicating edges to create an Eulerian graph.
        We do this by matching odd-degree vertices in pairs and duplicating edges along
        the shortest paths between matched vertices.
        
        Returns:
        --------
        odd_vertices: List of vertex indices that have odd degrees
        """
        odd_vertices = []
        
        # For each vertex, count connections (non-infinite distances)
        for i in range(self.num_of_stations):
            # Count edges: non-infinite and not self-loop
            degree = np.sum((self.distance_matrix[i] != np.inf) & (self.distance_matrix[i] != 0))
            
            # Account for parallel edges if any exist
            for (v1, v2), count in self.edge_counts.items():
                if i == v1 or i == v2:
                    degree += count - 1  # Add extra edges beyond the first
            
            if degree % 2 == 1:  # Check if odd
                odd_vertices.append(i)
        
        # Print summary for understanding
        print(f"\nFound {len(odd_vertices)} odd-degree vertices")
        odd_names = [self.index_to_station[i] for i in odd_vertices]
        print(f"Odd vertices: {odd_names[:10]}{'...' if len(odd_names) > 10 else ''}")
        
        return odd_vertices
    
    def compute_shortest_paths(self) -> np.ndarray:
        """
        Find shortest paths between all pairs of stations using scipy's optimized implementation.
        
        WHY SCIPY INSTEAD OF FLOYD-WARSHALL?
        While Floyd-Warshall is a classic algorithm, scipy's shortest_path function:
        - Is highly optimized with C/Fortran backends
        - Automatically selects the best algorithm (Dijkstra, Bellman-Ford, or Floyd-Warshall)
        - Handles sparse matrices efficiently
        - Is significantly faster for large graphs
        
        The algorithm still finds the shortest path between every pair of stations,
        which we need for the matching step.
        
        Returns:
        --------
        shortest_paths : np.ndarray
            Matrix where shortest_paths[i][j] is the shortest distance from i to j
        """
        # Convert to sparse matrix for efficiency (many entries might be infinity)
        # This is especially beneficial for large, sparse railway networks
        sparse_dist = csr_matrix(self.distance_matrix)
        
        # Use scipy's optimized shortest path algorithm
        # directed=False because railway connections are bidirectional
        shortest_paths, predecessors = shortest_path(
            sparse_dist, 
            directed=False, 
            return_predecessors=True
        )
        
        # Store predecessors for path reconstruction later
        self.path_predecessors = predecessors
        
        return shortest_paths
    
    def find_minimum_weight_matching(self, odd_vertices: List[int]) -> List[Tuple[int, int]]:
        """
        Find the best way to pair up odd-degree vertices using a more efficient approach.
        
        THE MATCHING PROBLEM:
        We need to pair up odd vertices so that the total distance between pairs is minimized.
        This is because we'll duplicate edges along the shortest paths between paired vertices.
        
        Since we know the graph is non-Eulerian, we're guaranteed to have odd vertices to match.
        The number of odd vertices is always even (graph theory's handshaking lemma).
        
        IMPROVED ALGORITHM:
        Instead of the original greedy approach, we use a more sophisticated method that
        considers all possible pairings for small sets, and uses a smarter greedy approach
        for larger sets.
        
        Parameters:
        -----------
        odd_vertices : list
            Indices of vertices with odd degree (guaranteed to be non-empty and even count)
            
        Returns:
        --------
        matching : list
            List of pairs (tuples) of matched vertices
        """
        # Get shortest paths between all stations
        shortest_paths = self.compute_shortest_paths()
        
        n_odd = len(odd_vertices)
        
        # For small numbers of odd vertices, we can try all possible pairings
        if n_odd <= 10:
            print(f"\nFinding optimal matching for {n_odd} odd vertices...")
            return self._optimal_matching_small(odd_vertices, shortest_paths)
        else:
            # For larger sets, use an improved greedy approach
            print(f"\nFinding matching for {n_odd} odd vertices using improved greedy...")
            return self._improved_greedy_matching(odd_vertices, shortest_paths)
    
    def _optimal_matching_small(self, odd_vertices: List[int], shortest_paths: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find optimal matching for small sets of odd vertices by trying all possibilities.
        
        This is feasible for small sets because the number of possible matchings
        grows as (n-1)!! which is manageable for n <= 10.
        """
        from itertools import combinations
        
        def generate_all_matchings(vertices):
            """Generate all possible ways to pair up vertices."""
            if len(vertices) == 0:
                yield []
            else:
                first = vertices[0]
                rest = vertices[1:]
                for i, partner in enumerate(rest):
                    pair = (first, partner)
                    remaining = rest[:i] + rest[i+1:]
                    for sub_matching in generate_all_matchings(remaining):
                        yield [pair] + sub_matching
        
        best_matching = None
        best_cost = float('inf')
        
        # Try all possible matchings
        for matching in generate_all_matchings(odd_vertices):
            cost = sum(shortest_paths[v1][v2] for v1, v2 in matching)
            if cost < best_cost:
                best_cost = cost
                best_matching = matching
        
        # Print result
        for v1, v2 in best_matching:
            v1_name = self.index_to_station[v1]
            v2_name = self.index_to_station[v2]
            print(f"  Matched {v1_name} ↔ {v2_name} (distance: {shortest_paths[v1][v2]:.0f})")
        
        return best_matching
    
    def _improved_greedy_matching(self, odd_vertices: List[int], shortest_paths: np.ndarray) -> List[Tuple[int, int]]:
        """
        Improved greedy matching that considers the global picture better.
        
        Instead of just picking the closest pair each time, we score each potential
        pairing based on how it affects the remaining vertices.
        """
        remaining = set(odd_vertices)
        matching = []
        
        while remaining:
            best_score = float('inf')
            best_pair = None
            
            # For each possible pair, calculate a score that considers:
            # 1. The distance of the pair itself
            # 2. How this pairing affects options for remaining vertices
            for v1 in remaining:
                for v2 in remaining:
                    if v1 < v2:
                        # Base score is the distance
                        score = shortest_paths[v1][v2]
                        
                        # Penalty if this pairing leaves other vertices with poor options
                        temp_remaining = remaining - {v1, v2}
                        if len(temp_remaining) >= 2:
                            # Find minimum distance from each remaining vertex to any other
                            min_distances = []
                            for v in temp_remaining:
                                min_dist = min(shortest_paths[v][u] for u in temp_remaining if u != v)
                                min_distances.append(min_dist)
                            
                            # Add average of minimum distances as penalty
                            if min_distances:
                                score += np.mean(min_distances) * 0.5
                        
                        if score < best_score:
                            best_score = score
                            best_pair = (v1, v2)
            
            if best_pair:
                matching.append(best_pair)
                remaining.remove(best_pair[0])
                remaining.remove(best_pair[1])
                
                # Show progress
                v1_name = self.index_to_station[best_pair[0]]
                v2_name = self.index_to_station[best_pair[1]]
                print(f"  Matched {v1_name} ↔ {v2_name} (distance: {shortest_paths[best_pair[0]][best_pair[1]]:.0f})")
        
        return matching
    
    def reconstruct_shortest_path(self, start_idx: int, end_idx: int) -> List[int]:
        """
        Reconstruct the shortest path between two stations using the predecessor matrix.
        
        This is more efficient than implementing Dijkstra from scratch, as we've
        already computed all shortest paths and stored the predecessor information.
        
        Parameters:
        -----------
        start_idx : int
            Starting station index
        end_idx : int
            Ending station index
            
        Returns:
        --------
        path : list
            List of station indices forming the shortest path
        """
        if not hasattr(self, 'path_predecessors'):
            # If we haven't computed shortest paths yet, do it now
            self.compute_shortest_paths()
        
        # Reconstruct path from predecessor matrix
        path = []
        current = end_idx
        
        while current != start_idx:
            path.append(current)
            current = self.path_predecessors[start_idx, current]
            if current == -9999:  # scipy uses -9999 for unreachable
                return []  # No path exists
        
        path.append(start_idx)
        path.reverse()
        
        return path
    
    def augment_graph_to_eulerian(self, matching: List[Tuple[int, int]]) -> Dict[Tuple[int, int], int]:
        """
        Add duplicate edges to make the graph Eulerian.
        
        For each matched pair of odd vertices, we duplicate edges along the 
        shortest path between them. This ensures all vertices end up with even degree.
        
        We use a dictionary instead of a matrix for the augmented edge counts
        to save memory and make the edge traversal more efficient.
        
        Parameters:
        -----------
        matching : list
            Pairs of matched odd-degree vertices
            
        Returns:
        --------
        augmented_edges : dict
            Dictionary mapping (i,j) to number of edges between stations i and j
        """
        # Start with a copy of the original edge counts
        augmented_edges = self.edge_counts.copy()
        
        print(f"\nAugmenting graph with {len(matching)} paths...")
        total_duplicated = 0
        
        for v1, v2 in matching:
            # Find shortest path between matched vertices
            path = self.reconstruct_shortest_path(v1, v2)
            
            # Duplicate edges along this path
            path_length = 0
            for i in range(len(path) - 1):
                curr = path[i]
                next = path[i + 1]
                
                # Normalize edge key (always smaller index first)
                edge_key = (min(curr, next), max(curr, next))
                
                # Add one more edge between these stations
                if edge_key in augmented_edges:
                    augmented_edges[edge_key] += 1
                else:
                    augmented_edges[edge_key] = 1
                
                path_length += self.distance_matrix[curr][next]
                total_duplicated += 1
            
            # Show what we're doing
            path_names = [self.index_to_station[i] for i in path]
            print(f"  Duplicating path: {' → '.join(path_names[:3])}{'...' if len(path_names) > 3 else ''} (length: {path_length:.0f})")
        
        print(f"Total edges duplicated: {total_duplicated}")
        return augmented_edges
    
    def find_eulerian_circuit(self, edge_dict: Dict[Tuple[int, int], int]) -> List[str]:
        """
        Find an Eulerian circuit using Hierholzer's algorithm with an efficient edge representation.
        
        HIERHOLZER'S ALGORITHM:
        1. Start at any vertex
        2. Follow edges to form a circuit, marking them as used
        3. If you get stuck but unused edges remain, backtrack to a vertex with unused edges
        4. Continue until all edges are used
        
        We use a dictionary-based approach for efficiency, which is faster than
        matrix operations for sparse graphs.
        
        Parameters:
        -----------
        edge_dict : dict
            Dictionary mapping (i,j) to number of edges between vertices i and j
            
        Returns:
        --------
        circuit : list
            List of station names forming the Eulerian circuit
        """
        # Build adjacency list from edge dictionary for efficient traversal
        adjacency = {i: [] for i in range(self.num_of_stations)}
        remaining_edges = {}
        
        for (i, j), count in edge_dict.items():
            adjacency[i].append(j)
            adjacency[j].append(i)
            remaining_edges[(i, j)] = count
            remaining_edges[(j, i)] = count
        
        # Start from the first station
        start = 0
        stack = [start]
        circuit = []
        
        print("\nFinding Eulerian circuit...")
        edges_used = 0
        total_edges = sum(edge_dict.values())
        
        while stack:
            current = stack[-1]
            
            # Find an unused edge from current vertex
            next_vertex = None
            for neighbor in adjacency[current]:
                edge_key = (min(current, neighbor), max(current, neighbor))
                if remaining_edges.get((current, neighbor), 0) > 0:
                    next_vertex = neighbor
                    break
            
            if next_vertex is not None:
                # Use this edge
                edge_key = (min(current, next_vertex), max(current, next_vertex))
                remaining_edges[(current, next_vertex)] -= 1
                remaining_edges[(next_vertex, current)] -= 1
                stack.append(next_vertex)
                edges_used += 1
                
                # Show progress for large graphs
                if edges_used % 50 == 0:
                    print(f"  Progress: {edges_used}/{total_edges} edges used")
            else:
                # No more edges from current vertex, add to circuit
                circuit.append(stack.pop())
        
        # Convert indices back to station names
        circuit.reverse()
        circuit_names = [self.index_to_station[i] for i in circuit]
        
        print(f"  Circuit found with {len(circuit_names)} stations")
        return circuit_names
    
    def decompose_into_trajectories(self, eulerian_circuit: List[str], strategy: str = 'balanced') -> Dict[int, List[str]]:
        """
        Split the Eulerian circuit into multiple train routes.
        
        This is where we handle the multi-train aspect. We have one long circuit that 
        covers all edges, but we need to split it into routes that:
        1. Each train can complete within the time limit (120 minutes)
        2. Together cover all the original edges
        3. Minimize the total number of trains used
        
        Parameters:
        -----------
        eulerian_circuit : list
            The complete circuit covering all edges
        strategy : str
            'simple' - Just split when time limit reached
            'balanced' - Try to balance route lengths
            'coverage' - Prioritize coverage of original edges (new!)
            
        Returns:
        --------
        trajectories : dict
            Dictionary mapping train ID to its route
        """
        if strategy == 'simple':
            return self._simple_decomposition(eulerian_circuit)
        elif strategy == 'balanced':
            return self._balanced_decomposition(eulerian_circuit)
        elif strategy == 'coverage':
            return self._coverage_based_decomposition(eulerian_circuit)
        else:
            print(f"Unknown strategy '{strategy}', using 'balanced'")
            return self._balanced_decomposition(eulerian_circuit)
    
    def _simple_decomposition(self, circuit: List[str]) -> Dict[int, List[str]]:
        """
        Simple sequential splitting when time limit is reached.
        
        This is the fastest decomposition method but may not produce optimal results.
        """
        trajectories = {}
        train_id = 0
        current_route = [circuit[0]]
        current_time = 0
        
        for i in range(len(circuit) - 1):
            # Get stations and find travel time
            current_station = circuit[i]
            next_station = circuit[i + 1]
            
            # Look up travel time using indices
            curr_idx = self.station_to_index[current_station]
            next_idx = self.station_to_index[next_station]
            travel_time = self.distance_matrix[curr_idx][next_idx]
            
            # Check if adding this segment exceeds time limit
            if current_time + travel_time > self.max_minutes:
                # Save current route and start new one
                trajectories[train_id] = current_route
                train_id += 1
                current_route = [current_station, next_station]
                current_time = travel_time
            else:
                # Add to current route
                current_route.append(next_station)
                current_time += travel_time
        
        # Don't forget the last route
        if len(current_route) > 1:
            trajectories[train_id] = current_route
        
        return trajectories
    
    def _coverage_based_decomposition(self, circuit: List[str]) -> Dict[int, List[str]]:
        """
        New decomposition strategy that prioritizes coverage of original edges.
        
        This method tries to ensure each train covers as many unique original edges
        as possible, which can lead to better overall coverage with fewer trains.
        """
        # First, analyze the circuit to identify segments
        circuit_edges = []
        for i in range(len(circuit) - 1):
            s1, s2 = circuit[i], circuit[i + 1]
            edge = tuple(sorted([s1, s2]))
            is_original = edge in self.original_edges
            
            # Get travel time
            idx1 = self.station_to_index[s1]
            idx2 = self.station_to_index[s2]
            travel_time = self.distance_matrix[idx1][idx2]
            
            circuit_edges.append({
                'start': s1,
                'end': s2,
                'edge': edge,
                'time': travel_time,
                'is_original': is_original,
                'index': i
            })
        
        # Group consecutive edges into segments
        segments = []
        current_segment = []
        
        for edge_info in circuit_edges:
            if not current_segment:
                current_segment = [edge_info]
            elif edge_info['start'] == current_segment[-1]['end']:
                current_segment.append(edge_info)
            else:
                segments.append(current_segment)
                current_segment = [edge_info]
        
        if current_segment:
            segments.append(current_segment)
        
        # Score and sort segments by importance
        scored_segments = []
        for segment in segments:
            score = sum(1 for e in segment if e['is_original'])  # Number of original edges
            time = sum(e['time'] for e in segment)
            scored_segments.append({
                'segment': segment,
                'score': score,
                'time': time,
                'score_per_time': score / time if time > 0 else 0
            })
        
        # Sort by score per time (efficiency)
        scored_segments.sort(key=lambda x: x['score_per_time'], reverse=True)
        
        # Build trajectories using best segments first
        trajectories = {}
        used_indices = set()
        train_id = 0
        
        for scored_seg in scored_segments:
            segment = scored_seg['segment']
            
            # Check if any edge in this segment is already used
            segment_indices = {e['index'] for e in segment}
            if segment_indices & used_indices:
                continue  # Skip if overlap
            
            # Try to build a trajectory starting with this segment
            if scored_seg['time'] <= self.max_minutes:
                route = [segment[0]['start']]
                for e in segment:
                    route.append(e['end'])
                
                trajectories[train_id] = route
                train_id += 1
                used_indices.update(segment_indices)
        
        # Handle any remaining edges with simple decomposition
        remaining_circuit = []
        for i, edge_info in enumerate(circuit_edges):
            if i not in used_indices:
                if not remaining_circuit or remaining_circuit[-1] != edge_info['start']:
                    remaining_circuit.append(edge_info['start'])
                remaining_circuit.append(edge_info['end'])
        
        if len(remaining_circuit) > 1:
            remaining_trajectories = self._simple_decomposition(remaining_circuit)
            for route in remaining_trajectories.values():
                trajectories[train_id] = route
                train_id += 1
        
        return trajectories
    
    def _balanced_decomposition(self, circuit: List[str]) -> Dict[int, List[str]]:
        """
        Try to create balanced trajectories that maximize coverage.
        
        This is a more sophisticated approach that tries to:
        1. Ensure each trajectory covers unique edges when possible
        2. Balance the time usage across trains
        3. Minimize the total number of trains
        """
        # First, identify which edges in the circuit are "important" (original edges)
        circuit_edges = []
        for i in range(len(circuit) - 1):
            s1, s2 = circuit[i], circuit[i + 1]
            edge = tuple(sorted([s1, s2]))
            is_original = edge in self.original_edges
            
            # Get travel time
            idx1 = self.station_to_index[s1]
            idx2 = self.station_to_index[s2]
            travel_time = self.distance_matrix[idx1][idx2]
            
            circuit_edges.append({
                'start': s1,
                'end': s2,
                'edge': edge,
                'time': travel_time,
                'is_original': is_original,
                'index': i
            })
        
        # Greedily build trajectories prioritizing original edges
        trajectories = {}
        used_indices = set()
        train_id = 0
        edges_covered = set()
        
        while len(used_indices) < len(circuit_edges) and train_id < self.max_trains:
            current_route = []
            current_time = 0
            current_station = None
            
            # Find a good starting point (preferably an uncovered original edge)
            start_idx = None
            
            # First try: uncovered original edge
            for i, edge_info in enumerate(circuit_edges):
                if i not in used_indices and edge_info['is_original'] and edge_info['edge'] not in edges_covered:
                    start_idx = i
                    break
            
            # Second try: any uncovered original edge
            if start_idx is None:
                for i, edge_info in enumerate(circuit_edges):
                    if i not in used_indices and edge_info['is_original']:
                        start_idx = i
                        break
            
            # Last resort: any unused edge
            if start_idx is None:
                for i in range(len(circuit_edges)):
                    if i not in used_indices:
                        start_idx = i
                        break
            
            if start_idx is None:
                break
            
            # Build trajectory from this starting point
            i = start_idx
            while i < len(circuit_edges) and i not in used_indices:
                edge_info = circuit_edges[i]
                
                if current_station is None:
                    # First edge
                    current_route = [edge_info['start'], edge_info['end']]
                    current_time = edge_info['time']
                    current_station = edge_info['end']
                    used_indices.add(i)
                    if edge_info['is_original']:
                        edges_covered.add(edge_info['edge'])
                elif edge_info['start'] == current_station and current_time + edge_info['time'] <= self.max_minutes:
                    # Can add this edge
                    current_route.append(edge_info['end'])
                    current_time += edge_info['time']
                    current_station = edge_info['end']
                    used_indices.add(i)
                    if edge_info['is_original']:
                        edges_covered.add(edge_info['edge'])
                else:
                    # Can't add this edge, try next trajectory
                    break
                
                i += 1
            
            if len(current_route) > 1:
                trajectories[train_id] = current_route
                train_id += 1
        
        print(f"Balanced decomposition: {len(trajectories)} trains cover {len(edges_covered)} original edges")
        return trajectories
    
    def run(self, decomposition_strategy: str = 'balanced', print_matrices: bool = False) -> Dict[int, List]:
        """
        Execute the complete Chinese Postman Problem algorithm.
        
        Since we know the graph is NOT Eulerian, we skip the check and proceed directly to:
        1. Find odd-degree vertices (there will always be some)
        2. Match them optimally
        3. Augment the graph by duplicating edges
        4. Find Eulerian circuit in the augmented graph
        5. Decompose into train routes
        
        Parameters:
        -----------
        decomposition_strategy : str
            'simple', 'balanced', or 'coverage' (default: 'balanced')
        print_matrices : bool
            Whether to print matrices for visualization
            
        Returns:
        --------
        trajectories : dict
            Dictionary mapping train IDs to lists of Station objects
        """
        print("\n" + "="*60)
        print("STARTING CHINESE POSTMAN PROBLEM ALGORITHM")
        print("="*60)
        print("Note: Graph is known to be non-Eulerian, proceeding to augmentation...")
        
        # Optional: Print matrices for learning/debugging
        if print_matrices:
            self._print_matrix(self.distance_matrix, "Distance Matrix (travel times)")
            adjacency = self.get_adjacency_from_distance()
            self._print_matrix(adjacency, "Adjacency Matrix (derived from distances)")
        
        # Step 1: Find odd-degree vertices (we know there will be some)
        odd_vertices = self.find_odd_degree_vertices()
        
        # Step 2: Find minimum-weight matching of odd vertices
        print(f"\nMatching {len(odd_vertices)} odd vertices for augmentation...")
        matching = self.find_minimum_weight_matching(odd_vertices)
        
        # Step 3: Augment graph to make it Eulerian
        augmented_edges = self.augment_graph_to_eulerian(matching)
        
        # Step 4: Find Eulerian circuit in the augmented graph
        eulerian_circuit = self.find_eulerian_circuit(augmented_edges)
        
        # Step 5: Decompose into trajectories
        trajectories = self.decompose_into_trajectories(eulerian_circuit, decomposition_strategy)
        
        # Step 6: Convert back to Station objects and mark as visited
        final_trajectories = self._finalize_trajectories(trajectories)
        
        # Step 7: Print solution summary
        self._print_solution_summary(final_trajectories)
        
        return final_trajectories
    
    def _finalize_trajectories(self, trajectories: Dict[int, List[str]]) -> Dict[int, List]:
        """
        Convert trajectory station names back to Station objects and mark connections as visited.
        """
        final_trajectories = {}
        
        for train_id, route in trajectories.items():
            # Convert names to Station objects
            station_objects = [self.station_dict[station_name] for station_name in route]
            final_trajectories[train_id] = station_objects
            
            # Mark connections as visited
            for i in range(len(route) - 1):
                station1 = route[i]
                station2 = route[i + 1]
                
                # Find and mark the connection
                for connection in self.connections:
                    if (connection.station_1 == station1 and connection.station_2 == station2) or \
                       (connection.station_1 == station2 and connection.station_2 == station1):
                        connection.mark_visited()
                        break
        
        return final_trajectories
    
    def _print_matrix(self, matrix: np.ndarray, title: str):
        """Print a matrix in a readable format (first 8x8 subset for space)."""
        print(f"\n{title}:")
        print("       ", end="")
        for j in range(min(8, self.num_of_stations)):
            print(f"{self.station_names[j][:7]:>8}", end="")
        print("  ...")
        
        for i in range(min(8, self.num_of_stations)):
            print(f"{self.station_names[i][:7]:>7}", end="")
            for j in range(min(8, self.num_of_stations)):
                if matrix[i][j] == np.inf:
                    print(f"{'∞':>8}", end="")
                elif isinstance(matrix[i][j], (int, np.integer)):
                    print(f"{int(matrix[i][j]):>8}", end="")
                else:
                    print(f"{matrix[i][j]:>8.1f}", end="")
            print("  ...")
        print("...")
    
    def _print_solution_summary(self, trajectories: Dict[int, List]):
        """Print a detailed summary of the solution."""
        print("\n" + "="*60)
        print("SOLUTION SUMMARY")
        print("="*60)
        
        total_time = 0
        covered_edges = set()
        
        # Analyze each trajectory
        for train_id, stations in trajectories.items():
            route_time = 0
            route_edges = []
            
            for i in range(len(stations) - 1):
                s1 = stations[i].name
                s2 = stations[i + 1].name
                edge = tuple(sorted([s1, s2]))
                route_edges.append(edge)
                
                # Add to covered edges
                covered_edges.add(edge)
                
                # Calculate time
                idx1 = self.station_to_index[s1]
                idx2 = self.station_to_index[s2]
                route_time += self.distance_matrix[idx1][idx2]
            
            total_time += route_time
            print(f"Train {train_id}: {len(stations)} stations, {route_time:.0f} minutes")
            print(f"  Route: {stations[0].name} → ... → {stations[-1].name}")
        
        # Calculate coverage
        coverage = len(covered_edges) / len(self.original_edges)
        
        print(f"\nTotal trains used: {len(trajectories)}")
        print(f"Total time: {total_time:.0f} minutes")
        print(f"Coverage: {coverage * 100:.1f}% ({len(covered_edges)}/{len(self.original_edges)} edges)")
        
        # Calculate expected score
        K = coverage * 10000 - (len(trajectories) * 100 + total_time)
        print(f"Expected score: {K:.0f}")
        print("="*60)