"""
Quantum Approximate Optimization Algorithm (QAOA) for Portfolio Optimization
Based on the paper: Quantum-Accelerated Deal Screening and Risk Assessment
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# For quantum simulation
try:
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit import Aer, execute
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Pauli
    from qiskit.opflow import PauliSumOp, MatrixOp
    QISKIT_AVAILABLE = True
except ImportError:
    print("Qiskit not installed. Using classical simulation only.")
    QISKIT_AVAILABLE = False

class QAOAPortfolioOptimizer:
    """
    QAOA implementation for portfolio optimization with cardinality constraints.
    Mathematical formulation from the paper:
    H_C = -μ^T z + λz^T Σz + α(Σz_i - K)^2
    """
    
    def __init__(self, 
                 returns: np.ndarray,
                 covariance: np.ndarray,
                 risk_aversion: float = 0.5,
                 cardinality: int = 10,
                 penalty: float = 1.0,
                 qaoa_depth: int = 3):
        """
        Initialize QAOA Portfolio Optimizer.
        
        Args:
            returns: Expected returns for each asset
            covariance: Covariance matrix of returns
            risk_aversion: Risk aversion parameter (λ)
            cardinality: Number of assets to select (K)
            penalty: Penalty for cardinality constraint (α)
            qaoa_depth: Number of QAOA layers (p)
        """
        self.returns = returns
        self.covariance = covariance
        self.n_assets = len(returns)
        self.risk_aversion = risk_aversion
        self.cardinality = cardinality
        self.penalty = penalty
        self.qaoa_depth = qaoa_depth
        
        # QAOA parameters
        self.gamma = None
        self.beta = None
        self.optimization_results = []
        
    def encode_portfolio_hamiltonian(self) -> Dict:
        """
        Encode portfolio optimization problem as QUBO/Ising Hamiltonian.
        Returns coefficients for the cost Hamiltonian.
        """
        n = self.n_assets
        
        # Linear terms (from expected returns)
        h = -self.returns
        
        # Quadratic terms (from risk/covariance)
        J = self.risk_aversion * self.covariance
        
        # Cardinality constraint penalty
        # Add penalty term: α(Σz_i - K)^2
        for i in range(n):
            h[i] += self.penalty * (-2 * self.cardinality)
            for j in range(n):
                J[i, j] += self.penalty * 2
                
        # Add constant term
        const = self.penalty * self.cardinality**2
        
        return {'h': h, 'J': J, 'const': const}
    
    def create_qaoa_circuit(self, gamma: List[float], beta: List[float]) -> 'QuantumCircuit':
        """
        Create QAOA circuit with given parameters.
        
        Args:
            gamma: Cost Hamiltonian angles
            beta: Mixer Hamiltonian angles
            
        Returns:
            QuantumCircuit implementing QAOA
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for quantum circuit creation")
            
        n = self.n_assets
        qc = QuantumCircuit(n, n)
        
        # Initial state: equal superposition
        qc.h(range(n))
        
        # Get Hamiltonian coefficients
        ham = self.encode_portfolio_hamiltonian()
        
        # Apply QAOA layers
        for p in range(self.qaoa_depth):
            # Cost unitary: e^(-i*gamma*H_C)
            self._apply_cost_unitary(qc, gamma[p], ham)
            
            # Mixer unitary: e^(-i*beta*H_M)
            self._apply_mixer_unitary(qc, beta[p])
        
        # Measure all qubits
        qc.measure(range(n), range(n))
        
        return qc
    
    def _apply_cost_unitary(self, qc: 'QuantumCircuit', gamma: float, ham: Dict):
        """Apply cost Hamiltonian unitary."""
        n = self.n_assets
        
        # Apply phase for linear terms
        for i in range(n):
            qc.rz(2 * gamma * ham['h'][i], i)
        
        # Apply phase for quadratic terms (using CNOT ladder)
        for i in range(n):
            for j in range(i+1, n):
                if abs(ham['J'][i, j]) > 1e-10:
                    qc.cx(i, j)
                    qc.rz(2 * gamma * ham['J'][i, j], j)
                    qc.cx(i, j)
    
    def _apply_mixer_unitary(self, qc: 'QuantumCircuit', beta: float):
        """Apply mixer Hamiltonian unitary (X rotation on all qubits)."""
        for i in range(self.n_assets):
            qc.rx(2 * beta, i)
    
    def classical_expectation(self, gamma: np.ndarray, beta: np.ndarray) -> float:
        """
        Classical simulation of QAOA expectation value.
        Used when Qiskit is not available or for small problems.
        """
        n = self.n_assets
        ham = self.encode_portfolio_hamiltonian()
        
        # Initialize state vector (equal superposition)
        state = np.ones(2**n) / np.sqrt(2**n)
        
        # Apply QAOA evolution
        for p in range(self.qaoa_depth):
            # Cost Hamiltonian evolution
            for idx in range(2**n):
                binary = format(idx, f'0{n}b')
                phase = 0
                
                # Linear terms
                for i in range(n):
                    if binary[i] == '1':
                        phase += ham['h'][i]
                
                # Quadratic terms
                for i in range(n):
                    for j in range(i+1, n):
                        if binary[i] == '1' and binary[j] == '1':
                            phase += ham['J'][i, j]
                
                state[idx] *= np.exp(-1j * gamma[p] * phase)
            
            # Mixer Hamiltonian evolution (simplified)
            # This is an approximation for demonstration
            state = self._apply_mixer_classical(state, beta[p])
        
        # Compute expectation value
        expectation = 0
        for idx in range(2**n):
            binary = format(idx, f'0{n}b')
            energy = ham['const']
            
            for i in range(n):
                if binary[i] == '1':
                    energy += ham['h'][i]
            
            for i in range(n):
                for j in range(i+1, n):
                    if binary[i] == '1' and binary[j] == '1':
                        energy += ham['J'][i, j]
            
            expectation += abs(state[idx])**2 * energy
        
        return expectation.real
    
    def _apply_mixer_classical(self, state: np.ndarray, beta: float) -> np.ndarray:
        """Apply mixer Hamiltonian in classical simulation."""
        n = self.n_assets
        new_state = np.zeros_like(state, dtype=complex)
        
        for idx in range(2**n):
            binary = list(format(idx, f'0{n}b'))
            
            # Apply X gates effect
            for i in range(n):
                # Flip bit i
                flipped_binary = binary.copy()
                flipped_binary[i] = '0' if binary[i] == '1' else '1'
                flipped_idx = int(''.join(flipped_binary), 2)
                
                # Add contribution
                new_state[idx] += np.cos(beta) * state[idx]
                new_state[flipped_idx] += -1j * np.sin(beta) * state[idx]
        
        return new_state / np.sqrt(n)  # Normalize
    
    def optimize_parameters(self, method: str = 'COBYLA', maxiter: int = 100) -> Dict:
        """
        Optimize QAOA parameters using classical optimizer.
        """
        # Initial parameters
        initial_gamma = np.random.uniform(0, 2*np.pi, self.qaoa_depth)
        initial_beta = np.random.uniform(0, np.pi, self.qaoa_depth)
        initial_params = np.concatenate([initial_gamma, initial_beta])
        
        def objective(params):
            gamma = params[:self.qaoa_depth]
            beta = params[self.qaoa_depth:]
            
            # Use classical simulation for expectation
            exp_val = self.classical_expectation(gamma, beta)
            
            # Store optimization history
            self.optimization_results.append({
                'iteration': len(self.optimization_results),
                'expectation': exp_val,
                'gamma': gamma.copy(),
                'beta': beta.copy()
            })
            
            return exp_val  # Minimize negative expected return
        
        # Run optimization
        result = minimize(objective, initial_params, method=method, 
                         options={'maxiter': maxiter})
        
        # Store optimal parameters
        self.gamma = result.x[:self.qaoa_depth]
        self.beta = result.x[self.qaoa_depth:]
        
        return {
            'optimal_value': result.fun,
            'optimal_gamma': self.gamma,
            'optimal_beta': self.beta,
            'optimization_success': result.success,
            'n_iterations': result.nit
        }
    
    def get_solution(self, n_samples: int = 1000) -> Dict:
        """
        Sample from optimized QAOA circuit to get portfolio solution.
        """
        if self.gamma is None or self.beta is None:
            raise ValueError("Parameters not optimized. Run optimize_parameters first.")
        
        # Simulate sampling (simplified)
        n = self.n_assets
        samples = []
        
        for _ in range(n_samples):
            # Random sampling based on QAOA distribution (simplified)
            # In practice, this would be from quantum measurement
            sample = np.random.randint(0, 2, n)
            
            # Apply cardinality constraint
            if np.sum(sample) > self.cardinality:
                indices = np.where(sample == 1)[0]
                keep = np.random.choice(indices, self.cardinality, replace=False)
                sample = np.zeros(n, dtype=int)
                sample[keep] = 1
            
            samples.append(sample)
        
        samples = np.array(samples)
        
        # Find best solution
        best_value = -np.inf
        best_solution = None
        
        ham = self.encode_portfolio_hamiltonian()
        
        for sample in samples:
            if np.sum(sample) == self.cardinality:  # Check cardinality
                value = np.dot(sample, self.returns) - \
                        self.risk_aversion * sample @ self.covariance @ sample
                
                if value > best_value:
                    best_value = value
                    best_solution = sample
        
        return {
            'best_solution': best_solution,
            'best_value': best_value,
            'selected_assets': np.where(best_solution == 1)[0] if best_solution is not None else [],
            'samples': samples
        }
    
    def benchmark_vs_classical(self) -> Dict:
        """
        Compare QAOA solution with classical methods.
        """
        # QAOA solution
        qaoa_result = self.get_solution()
        
        # Greedy solution
        greedy_solution = self._greedy_solution()
        
        # Random solution
        random_solution = self._random_solution()
        
        # Calculate approximation ratios
        results = {
            'qaoa': {
                'solution': qaoa_result['best_solution'],
                'value': qaoa_result['best_value'],
                'selected_assets': qaoa_result['selected_assets']
            },
            'greedy': {
                'solution': greedy_solution['solution'],
                'value': greedy_solution['value'],
                'selected_assets': greedy_solution['selected_assets']
            },
            'random': {
                'solution': random_solution['solution'],
                'value': random_solution['value'],
                'selected_assets': random_solution['selected_assets']
            }
        }
        
        # Calculate approximation ratios
        best_value = max(results['qaoa']['value'], 
                        results['greedy']['value'],
                        results['random']['value'])
        
        for method in results:
            if best_value != 0:
                results[method]['approximation_ratio'] = results[method]['value'] / best_value
            else:
                results[method]['approximation_ratio'] = 0
        
        return results
    
    def _greedy_solution(self) -> Dict:
        """Greedy algorithm for portfolio selection."""
        # Sort assets by Sharpe ratio approximation
        sharpe_approx = self.returns / np.diag(self.covariance)
        sorted_indices = np.argsort(sharpe_approx)[::-1]
        
        # Select top K assets
        solution = np.zeros(self.n_assets, dtype=int)
        solution[sorted_indices[:self.cardinality]] = 1
        
        value = np.dot(solution, self.returns) - \
                self.risk_aversion * solution @ self.covariance @ solution
        
        return {
            'solution': solution,
            'value': value,
            'selected_assets': sorted_indices[:self.cardinality]
        }
    
    def _random_solution(self) -> Dict:
        """Random portfolio selection."""
        indices = np.random.choice(self.n_assets, self.cardinality, replace=False)
        solution = np.zeros(self.n_assets, dtype=int)
        solution[indices] = 1
        
        value = np.dot(solution, self.returns) - \
                self.risk_aversion * solution @ self.covariance @ solution
        
        return {
            'solution': solution,
            'value': value,
            'selected_assets': indices
        }


def generate_portfolio_data(n_assets: int = 50, 
                           time_periods: int = 252,
                           seed: int = 42) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Generate synthetic portfolio data mimicking DAX stocks.
    
    Returns:
        returns: Expected returns
        covariance: Covariance matrix
        price_data: DataFrame with price history
    """
    np.random.seed(seed)
    
    # Generate synthetic price data
    initial_prices = np.random.uniform(20, 200, n_assets)
    
    # Generate returns with realistic parameters
    daily_returns = np.random.multivariate_normal(
        mean=np.random.uniform(-0.001, 0.002, n_assets),  # Daily returns
        cov=np.eye(n_assets) * 0.02**2 + np.random.randn(n_assets, n_assets) * 0.001,
        size=time_periods
    )
    
    # Calculate prices
    prices = np.zeros((time_periods, n_assets))
    prices[0] = initial_prices
    
    for t in range(1, time_periods):
        prices[t] = prices[t-1] * (1 + daily_returns[t])
    
    # Calculate expected returns and covariance
    returns = np.mean(daily_returns, axis=0) * 252  # Annualized
    covariance = np.cov(daily_returns.T) * 252  # Annualized
    
    # Create DataFrame
    dates = pd.date_range(start='2024-01-01', periods=time_periods, freq='D')
    columns = [f'Asset_{i:02d}' for i in range(n_assets)]
    price_data = pd.DataFrame(prices, index=dates, columns=columns)
    
    return returns, covariance, price_data


if __name__ == "__main__":
    # Generate synthetic portfolio data
    print("Generating synthetic portfolio data...")
    returns, covariance, price_data = generate_portfolio_data(n_assets=20)
    
    print(f"Portfolio shape: {returns.shape}")
    print(f"Expected annual return range: [{returns.min():.4f}, {returns.max():.4f}]")
    print(f"Covariance matrix shape: {covariance.shape}")
    
    # Initialize QAOA optimizer
    print("\nInitializing QAOA Portfolio Optimizer...")
    optimizer = QAOAPortfolioOptimizer(
        returns=returns,
        covariance=covariance,
        risk_aversion=0.5,
        cardinality=5,  # Select 5 assets
        penalty=1.0,
        qaoa_depth=3  # p=3 layers as per paper
    )
    
    # Optimize parameters
    print("Optimizing QAOA parameters...")
    optimization_result = optimizer.optimize_parameters(maxiter=50)
    
    print(f"Optimization successful: {optimization_result['optimization_success']}")
    print(f"Optimal value: {optimization_result['optimal_value']:.4f}")
    print(f"Number of iterations: {optimization_result['n_iterations']}")
    
    # Get solution
    print("\nFinding portfolio solution...")
    solution = optimizer.get_solution(n_samples=1000)
    
    print(f"Selected assets: {solution['selected_assets']}")
    print(f"Portfolio value: {solution['best_value']:.4f}")
    
    # Benchmark against classical methods
    print("\nBenchmarking against classical methods...")
    benchmark = optimizer.benchmark_vs_classical()
    
    for method, result in benchmark.items():
        print(f"\n{method.upper()}:")
        print(f"  Value: {result['value']:.4f}")
        print(f"  Approximation ratio: {result['approximation_ratio']:.2%}")
        print(f"  Selected assets: {result['selected_assets']}")
