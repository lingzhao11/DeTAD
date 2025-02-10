import numpy as np
from numpy.linalg import norm, svd
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.sparse.linalg import eigs
import concurrent.futures
from functools import partial
import argparse
import math
import Filter_TAD

class SymmetricMatrixFactorizationDualReg(BaseEstimator, RegressorMixin):
    def __init__(self, A,  k, lambda_reg=1.0, tau=1.0, phi=1.0, psi=1.0,
                max_iter=5000, tol=1e-6):
        """
        Symmetric Matrix Factorization with Dual Regularization.

        Parameters:
        - A: Contact matrix (n x n)
        - C: Distance-aware matrix (n x n)
        - k: Number of communities/components
        - lambda_reg: Regularization parameter for interaction-weighted regularization (lambda)
        - tau: Regularization parameter for distance-aware regularization (tau)
        - phi, psi: Parameters for constructing the distance-aware matrix C
        - constraint_type: Constraint on H ('nonnegative', 'unit', 'sparse', 'orthogonal', 'l1norm', 'simplex')
        - s: Sparsity level for 'sparse' constraint
        - alpha: Threshold for 'l1norm' constraint
        - max_iter: Maximum number of iterations
        - tol: Tolerance for convergence
        """
        self.A = A
        self.k = k
        self.n = A.shape[0]
        self.lambda_reg = lambda_reg
        self.tau = tau
        self.phi = phi
        self.psi = psi
        self.max_iter = max_iter
        self.tol = tol

    def get_params(self, deep=True):
        """
        Return the parameters of the model as a dictionary.
        """
        return {
            'A': self.A,
            'k': self.k,
            'lambda_reg': self.lambda_reg,
            'tau': self.tau,
            'phi': self.phi,
            'psi': self.psi,
            'max_iter': self.max_iter,
            'tol': self.tol
        }

    def set_params(self, **params):
        """
        Set the parameters of the model from a dictionary.
        """
        for key, value in params.items():
            setattr(self, key, value)
    def compute_degree_and_laplacian(self, W):
        """
        Compute the degree matrix D and Laplacian matrix L for a given weight matrix W.
        """
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        return D, L

    def compute_M(self, A, L_A, L_C):
        """
        Compute the matrix M = A - lambda * L_A - tau * L_C.
        """
        M = A - self.lambda_reg * L_A - self.tau * L_C
        return M

    def compute_gradient(self, H, M):
        """
        Compute the gradient of the objective function.
        """
        grad = 4 * (H @ H.T @ H - M @ H)
        return grad

    def compute_lipschitz_constant(self, H, M, L_A, L_C):
        """
        Compute the Lipschitz constant L_i.
        """
        # Compute spectral norms (largest singular values)
        # For HH^T - M
        HHt_M = H @ H.T - M
        sigma_max_HHt_M = norm(HHt_M, ord=2)  # Spectral norm

        # For H^T H
        HTH = H.T @ H
        sigma_max_HTH = norm(HTH, ord=2)  # Spectral norm

        # For Laplacians
        # Since L_A and L_C are symmetric positive semi-definite, their largest eigenvalues can be computed
        sigma_max_LA = np.linalg.eigvalsh(L_A).max()
        sigma_max_LC = np.linalg.eigvalsh(L_C).max()

        L_i = 4 * sigma_max_HHt_M + 8 * sigma_max_HTH + 2 * self.lambda_reg * sigma_max_LA + 2 * self.tau * sigma_max_LC
        return L_i

    def project_onto_constraint(self, H_temp):
        """
        Project H_temp onto the constraint set defined by self.constraint_type.

        Parameters:
        - H_temp: The matrix to be projected (n x k)

        Returns:
        - H_new: The projected matrix (n x k)
        """
       
        H_new = np.maximum(H_temp, 0)

        return H_new

    def fit(self):
        """
        Fit the model and return the community membership matrix H.
        """
     
        np.random.seed(0)
        H = np.abs(np.random.randn(self.n, self.k))

        self.C = self.construct_distance_matrix()
        # Compute degree and Laplacian matrices
        D_A, L_A = self.compute_degree_and_laplacian(self.A)
        D_C, L_C = self.compute_degree_and_laplacian(self.C)

        for iteration in range(self.max_iter):
            # Compute M
            M = self.compute_M(self.A, L_A, L_C)

            # Compute gradient
            grad = self.compute_gradient(H, M)

            # Compute Lipschitz constant
            L_i = self.compute_lipschitz_constant(H, M, L_A, L_C)

            if L_i == 0:
                break
            # Update H
            t = 1 / (2 * L_i)
            H_temp = H - t * grad
            # Project onto constraints
            H_new = self.project_onto_constraint(H_temp)
            # Check for convergence
            diff = norm(H_new - H, 'fro') / (norm(H, 'fro') + 1e-10)
            if diff < self.tol:
                break
            H = H_new
        return H

    def construct_distance_matrix(self):
        """
        Construct the distance-aware matrix C using parameters phi and psi.
        C_{ij} = phi * exp(-|i - j| / psi)
        """
        indices = np.arange(self.n)
        diff_matrix = np.abs(indices[:, np.newaxis] - indices[np.newaxis, :])
        C = self.phi * np.exp(-diff_matrix / self.psi)
        return C

    def extract_ranges(self, data, min_length=6):
        """
        Extract ranges of indices where values are greater than 0 and continuous, 
        and the length of the range must be greater than or equal to `min_length`.

        Args:
            data (list): List of numerical values.
            min_length (int): Minimum length for the sequence.

        Returns:
            list: A list of ranges in the format [start, end] where the length of the range is >= min_length.
        """
        ranges = []
        start = None
        
        for i, value in enumerate(data):
            if value > 0:  # Only consider values greater than 0
                if start is None:
                    start = i  # Mark the start of the range
            else:
                if start is not None:
                    # If range length is >= min_length, append it
                    if i - start >= min_length:
                        ranges.append([start, i - 1])  # Append as array [start, end]
                    start = None
        if start is not None and len(data) - start >= min_length:  # Handle last range if it extends to the end
            ranges.append([start, len(data) - 1])  # Append as array [start, end]
        
        return ranges
    
    def analyze_all_columns(self, data, min_length=6):
        """
        Analyze all columns and extract ranges for each column.

        Args:
            data (list of list): 2D list of numerical values.
            threshold (float): Threshold for range extraction.

        Returns:
            dict: A dictionary with column indices as keys and range lists as values.
        """
        results = {}
        num_columns = len(data[0])
        column_mean = np.mean(data)
        for col in range(num_columns):
            column_data = [row[col] for row in data]
            # Filter column data based on the mean
            filtered_data = [value if value > column_mean else 0 for value in column_data]
            # Extract ranges using the specified threshold
            results[f"Column {col+1}"] = self.extract_ranges(filtered_data, min_length)
        return results

def process_k(k, A, lambda_reg, tau, phi, psi, max_iter, tol):
    # Create the model and fit it for each value of k
    model = SymmetricMatrixFactorizationDualReg(A, k=k, lambda_reg=lambda_reg, tau=tau, phi=phi, psi=psi,
                                               max_iter=max_iter, tol=tol)
    H = model.fit()
    results = model.analyze_all_columns(H)
    ranges_list = []
    for column, ranges in results.items():
        ranges_list.append(ranges)  
    return ranges_list 

def run(A, lamb, tau, phi, psi, max_iter, tol):
    # Set up the thread pool
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create a partial function to pass the common arguments
        func = partial(process_k, A=A, lambda_reg=lamb, tau=tau, phi=phi, psi=psi, 
                       max_iter=max_iter, tol=tol)
        futures = {executor.submit(func, k): k for k in range(2, math.ceil(len(A)**0.5)+1)}
        candidates = []
        # Collect the results as they finish
        for future in concurrent.futures.as_completed(futures):
            each_candidate = future.result()
            candidates.append(each_candidate) 
    for list in candidates:
        # Flatten the list of ranges
        flat_list = [item for sublist1 in candidates for sublist2 in sublist1 for item in sublist2]
    eval_matrix = A/ A.max()
    mid_boundaries = Filter_TAD.filtering_noise(eval_matrix, flat_list)
    tad = Filter_TAD.score_filter(eval_matrix, mid_boundaries)
    return tad
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", dest="input", type=str, default="simu_data_over_h_gap.txt", help="destination of the data")
    parser.add_argument("-lambda", dest="lamb", type=float, default=0.001057951873872655)
    parser.add_argument("-tau", dest="tau", type=float, default=1.1533782092998626)
    parser.add_argument("-phi", dest="phi",type=float, default=2.06713065615515)
    parser.add_argument("-psi", dest="psi", type=float, default=11.309101670370517)
    parser.add_argument("-max_iter", dest="max_iter",type=int, default=1261)
    parser.add_argument("-tol", dest="tol", type=float, default=0.000546113648027901)
    parser.add_argument("-out", dest="out", type=str, default="result.txt", help="output of the results")
    args = parser.parse_args()
    TADs = run(np.loadtxt(args.input), args.lamb, args.tau, args.phi, args.psi, args.max_iter, args.tol)
    np.savetxt(args.out, TADs, fmt='%i', delimiter='\t')
if __name__ == "__main__":
    main()

