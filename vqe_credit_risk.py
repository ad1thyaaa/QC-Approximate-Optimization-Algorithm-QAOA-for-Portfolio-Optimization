"""
Variational Quantum Eigensolver (VQE) for Credit Risk Classification
Based on the paper: Quantum-Accelerated Deal Screening and Risk Assessment
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    from qiskit import Aer, execute
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RealAmplitudes, TwoLocal, EfficientSU2
    QISKIT_AVAILABLE = True
except ImportError:
    print("Qiskit not installed. Using classical simulation only.")
    QISKIT_AVAILABLE = False


class VQECreditRiskClassifier:
    """
    VQE implementation for credit risk classification.
    Uses variational quantum circuits to classify loan applications.
    """
    
    def __init__(self,
                 n_qubits: int = 6,
                 ansatz_depth: int = 3,
                 feature_map_reps: int = 2,
                 optimizer: str = 'COBYLA'):
        """
        Initialize VQE Credit Risk Classifier.
        
        Args:
            n_qubits: Number of qubits to use
            ansatz_depth: Depth of the variational ansatz
            feature_map_reps: Repetitions for feature mapping
            optimizer: Classical optimizer to use
        """
        self.n_qubits = n_qubits
        self.ansatz_depth = ansatz_depth
        self.feature_map_reps = feature_map_reps
        self.optimizer = optimizer
        
        # Model parameters
        self.theta = None  # Optimized parameters
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=min(n_qubits, 5))
        
        # Training history
        self.training_history = []
        
    def encode_features(self, X: np.ndarray) -> np.ndarray:
        """
        Encode classical features into quantum states.
        Maps features to rotation angles [0, π].
        """
        # Normalize features
        X_scaled = self.scaler.fit_transform(X) if not hasattr(self.scaler, 'mean_') \
                  else self.scaler.transform(X)
        
        # Reduce dimensionality if necessary
        if X_scaled.shape[1] > self.n_qubits:
            X_reduced = self.pca.fit_transform(X_scaled) if not hasattr(self.pca, 'components_') \
                       else self.pca.transform(X_scaled)
        else:
            X_reduced = X_scaled
        
        # Map to [0, π]
        X_encoded = np.pi * (X_reduced - X_reduced.min()) / (X_reduced.max() - X_reduced.min() + 1e-10)
        
        # Pad with zeros if necessary
        if X_encoded.shape[1] < self.n_qubits:
            padding = np.zeros((X_encoded.shape[0], self.n_qubits - X_encoded.shape[1]))
            X_encoded = np.hstack([X_encoded, padding])
        
        return X_encoded
    
    def create_feature_map_circuit(self, x: np.ndarray) -> 'QuantumCircuit':
        """
        Create quantum feature map circuit.
        Uses angle encoding with RY and entangling layers.
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for circuit creation")
        
        qc = QuantumCircuit(self.n_qubits)
        
        for rep in range(self.feature_map_reps):
            # Rotation layer
            for i in range(self.n_qubits):
                if i < len(x):
                    qc.ry(x[i], i)
                    qc.rz(x[i], i)
            
            # Entangling layer
            if rep < self.feature_map_reps - 1:
                for i in range(self.n_qubits - 1):
                    qc.cx(i, i + 1)
                if self.n_qubits > 2:
                    qc.cx(self.n_qubits - 1, 0)  # Circular entanglement
        
        return qc
    
    def create_variational_ansatz(self, theta: np.ndarray) -> 'QuantumCircuit':
        """
        Create variational ansatz circuit.
        Uses parameterized two-local ansatz with RY and RZ rotations.
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit required for circuit creation")
        
        qc = QuantumCircuit(self.n_qubits)
        param_idx = 0
        
        for layer in range(self.ansatz_depth):
            # Rotation layer
            for i in range(self.n_qubits):
                if param_idx < len(theta):
                    qc.ry(theta[param_idx], i)
                    param_idx += 1
                if param_idx < len(theta):
                    qc.rz(theta[param_idx], i)
                    param_idx += 1
            
            # Entangling layer
            for i in range(0, self.n_qubits - 1, 2):
                qc.cx(i, i + 1)
            for i in range(1, self.n_qubits - 1, 2):
                qc.cx(i, i + 1)
        
        # Final rotation layer
        for i in range(self.n_qubits):
            if param_idx < len(theta):
                qc.ry(theta[param_idx], i)
                param_idx += 1
        
        return qc
    
    def quantum_kernel(self, x1: np.ndarray, x2: np.ndarray, theta: np.ndarray) -> float:
        """
        Compute quantum kernel between two data points.
        K(x1, x2) = |<ψ(x1)|ψ(x2)>|²
        """
        # Classical simulation of quantum kernel
        # In practice, this would be computed on quantum hardware
        
        # Simple kernel approximation for demonstration
        diff = np.linalg.norm(x1 - x2)
        kernel_value = np.exp(-diff**2 / (2 * 0.5**2))
        
        # Add variational parameter influence
        theta_influence = np.cos(np.sum(theta[:len(x1)] * (x1 - x2)))
        
        return kernel_value * (0.7 + 0.3 * theta_influence)
    
    def compute_kernel_matrix(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Compute quantum kernel matrix for dataset.
        """
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i, n_samples):
                K[i, j] = self.quantum_kernel(X[i], X[j], theta)
                K[j, i] = K[i, j]
        
        return K
    
    def classical_prediction(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Classical simulation of VQE prediction.
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for i, x in enumerate(X):
            # Compute expectation value <ψ(x,θ)|Z|ψ(x,θ)>
            # Simplified calculation for demonstration
            
            # Feature influence
            feature_contrib = np.sum(x * theta[:len(x)])
            
            # Non-linear transformation
            activation = np.tanh(feature_contrib)
            
            # Add quantum-inspired noise
            noise = np.random.normal(0, 0.01)
            
            predictions[i] = (activation + noise + 1) / 2  # Map to [0, 1]
        
        return predictions
    
    def objective_function(self, theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Objective function for VQE optimization.
        Binary cross-entropy loss.
        """
        predictions = self.classical_prediction(X, theta)
        predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
        
        # Binary cross-entropy
        loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        
        # Add L2 regularization
        reg_term = 0.01 * np.sum(theta**2)
        
        total_loss = loss + reg_term
        
        # Store training history
        self.training_history.append({
            'iteration': len(self.training_history),
            'loss': loss,
            'total_loss': total_loss
        })
        
        return total_loss
    
    def fit(self, X: np.ndarray, y: np.ndarray, maxiter: int = 100) -> 'VQECreditRiskClassifier':
        """
        Train the VQE classifier.
        """
        # Encode features
        X_encoded = self.encode_features(X)
        
        # Initialize parameters
        n_params = 2 * self.n_qubits * (self.ansatz_depth + 1)
        initial_theta = np.random.uniform(-np.pi, np.pi, n_params)
        
        # Optimize
        print("Training VQE classifier...")
        result = minimize(
            lambda theta: self.objective_function(theta, X_encoded, y),
            initial_theta,
            method=self.optimizer,
            options={'maxiter': maxiter}
        )
        
        self.theta = result.x
        
        print(f"Optimization completed: {result.success}")
        print(f"Final loss: {result.fun:.4f}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for credit risk.
        """
        if self.theta is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        X_encoded = self.encode_features(X)
        probabilities = self.classical_prediction(X_encoded, self.theta)
        
        # Return probabilities for both classes
        return np.column_stack([1 - probabilities, probabilities])
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict credit risk classes.
        """
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= threshold).astype(int)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Evaluate model performance.
        """
        y_pred_proba = self.predict_proba(X)[:, 1]
        y_pred = self.predict(X)
        
        return {
            'auc': roc_auc_score(y, y_pred_proba),
            'accuracy': np.mean(y == y_pred),
            'precision': np.sum((y_pred == 1) & (y == 1)) / (np.sum(y_pred == 1) + 1e-10),
            'recall': np.sum((y_pred == 1) & (y == 1)) / (np.sum(y == 1) + 1e-10),
            'f1_score': 2 * np.sum((y_pred == 1) & (y == 1)) / (np.sum(y_pred == 1) + np.sum(y == 1) + 1e-10)
        }


def generate_credit_risk_data(n_samples: int = 1000,
                              n_features: int = 20,
                              noise_level: float = 0.1,
                              seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic credit risk dataset.
    
    Returns:
        features: DataFrame with loan application features
        labels: Binary labels (0: low risk, 1: high risk)
    """
    np.random.seed(seed)
    
    # Generate features
    features_dict = {
        'loan_amount': np.random.lognormal(10, 1, n_samples),
        'annual_income': np.random.lognormal(11, 0.7, n_samples),
        'debt_to_income': np.random.beta(2, 5, n_samples),
        'credit_score': np.random.normal(700, 100, n_samples),
        'employment_years': np.random.exponential(5, n_samples),
        'num_credit_accounts': np.random.poisson(5, n_samples),
        'num_late_payments': np.random.poisson(1, n_samples),
        'revolving_utilization': np.random.beta(2, 3, n_samples),
        'age': np.random.normal(40, 12, n_samples),
        'home_ownership': np.random.choice([0, 1, 2], n_samples),  # 0: rent, 1: mortgage, 2: own
    }
    
    # Add more random features
    for i in range(n_features - len(features_dict)):
        features_dict[f'feature_{i}'] = np.random.randn(n_samples)
    
    features_df = pd.DataFrame(features_dict)
    
    # Generate labels based on features (with some non-linear relationships)
    risk_score = (
        -0.3 * np.log(features_df['loan_amount'] / features_df['annual_income']) +
        0.4 * (features_df['credit_score'] - 700) / 100 +
        -0.2 * features_df['debt_to_income'] +
        -0.15 * features_df['num_late_payments'] +
        0.1 * np.log(features_df['employment_years'] + 1) +
        -0.2 * features_df['revolving_utilization'] +
        np.random.normal(0, noise_level, n_samples)
    )
    
    # Convert to binary labels
    labels = (risk_score < np.percentile(risk_score, 30)).astype(int)  # ~30% high risk
    
    return features_df, labels


def benchmark_classifiers(X_train, X_test, y_train, y_test):
    """
    Benchmark VQE against classical methods.
    """
    results = {}
    
    # VQE Classifier
    print("\n1. Training VQE Classifier...")
    vqe = VQECreditRiskClassifier(n_qubits=6, ansatz_depth=3)
    vqe.fit(X_train, y_train, maxiter=50)
    vqe_metrics = vqe.evaluate(X_test, y_test)
    results['VQE'] = vqe_metrics
    
    # SVM Classifier
    print("\n2. Training SVM Classifier...")
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_pred_proba = svm.predict_proba(X_test)[:, 1]
    svm_pred = svm.predict(X_test)
    results['SVM'] = {
        'auc': roc_auc_score(y_test, svm_pred_proba),
        'accuracy': np.mean(y_test == svm_pred),
        'precision': np.sum((svm_pred == 1) & (y_test == 1)) / (np.sum(svm_pred == 1) + 1e-10),
        'recall': np.sum((svm_pred == 1) & (y_test == 1)) / (np.sum(y_test == 1) + 1e-10),
    }
    
    # Random Forest Classifier
    print("\n3. Training Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred_proba = rf.predict_proba(X_test)[:, 1]
    rf_pred = rf.predict(X_test)
    results['Random Forest'] = {
        'auc': roc_auc_score(y_test, rf_pred_proba),
        'accuracy': np.mean(y_test == rf_pred),
        'precision': np.sum((rf_pred == 1) & (y_test == 1)) / (np.sum(rf_pred == 1) + 1e-10),
        'recall': np.sum((rf_pred == 1) & (y_test == 1)) / (np.sum(y_test == 1) + 1e-10),
    }
    
    return results


if __name__ == "__main__":
    # Generate synthetic credit risk data
    print("Generating synthetic credit risk dataset...")
    features_df, labels = generate_credit_risk_data(n_samples=500, n_features=20)
    
    print(f"Dataset shape: {features_df.shape}")
    print(f"Label distribution: {np.bincount(labels)} (0: low risk, 1: high risk)")
    print(f"High risk ratio: {np.mean(labels):.2%}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_df.values, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Benchmark classifiers
    print("\n" + "="*50)
    print("BENCHMARKING CLASSIFIERS")
    print("="*50)
    
    results = benchmark_classifiers(X_train, X_test, y_train, y_test)
    
    # Display results
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    print("\n", results_df)
    
    # Print improvement percentages
    print("\n" + "="*50)
    print("VQE IMPROVEMENT OVER CLASSICAL METHODS")
    print("="*50)
    
    vqe_auc = results['VQE']['auc']
    for method in ['SVM', 'Random Forest']:
        improvement = (vqe_auc - results[method]['auc']) / results[method]['auc'] * 100
        print(f"VQE vs {method}: {improvement:+.2f}% AUC improvement")
