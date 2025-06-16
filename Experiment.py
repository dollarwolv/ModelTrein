from Code.Algorithms.CPP import ChinesePostmanProblem

def run_cpp(self, decomposition_strategy='balanced', print_matrices=False):
        """
        Run the Chinese Postman Problem algorithm.
        
        Args:
            decomposition_strategy: 'simple' or 'balanced'
            print_matrices: Whether to show matrix visualizations
        """
    
    
        cpp_solver = ChinesePostmanProblem(
            self.stations,
            self.connections,
            self.station_dict,
            max_trains=20,
            max_minutes=180
        )
        
        self.trajectories = cpp_solver.run(
            decomposition_strategy=decomposition_strategy,
            print_matrices=print_matrices
        )
