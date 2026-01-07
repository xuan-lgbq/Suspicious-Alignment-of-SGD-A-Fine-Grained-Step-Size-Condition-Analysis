import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd  
import os           

def generate_positive_definite_matrix(d, k, m):
    """
    Generate a d-dimensional symmetric positive definite matrix A where the first k eigenvalues
    are at least m times larger than the remaining ones (lambda_k > m * lambda_{k+1}).
    """
    if k >= d or k < 1:
        raise ValueError("k must be between 1 and d-1")

    # Generate eigenvalues
    large_eigs = np.linspace(100, 80, k)  # Large eigenvalues, decreasing
    small_eigs = np.linspace(7, 1, d - k)  # Small eigenvalues, ensure max small < min large / m
    
    # Adjust to satisfy lambda_k > m * lambda_{k+1}
    lambda_k = large_eigs[-1]
    lambda_k1_max = lambda_k / (m + 0.1)  # Slightly more than m to ensure > m
    small_eigs = small_eigs * (lambda_k1_max / small_eigs[0])

    eigenvalues = np.concatenate([large_eigs, small_eigs])

    # Generate random orthogonal matrix Q
    random_matrix = np.random.randn(d, d)
    Q, _ = np.linalg.qr(random_matrix)

    # Diagonal matrix Lambda
    Lambda = np.diag(eigenvalues)

    # A = Q Lambda Q^T
    A = Q @ Lambda @ Q.T
    return A


def compute_projection_matrix(A, k):
    """
    Compute the projection matrix P onto the subspace spanned by the top k eigenvectors of A.
    """
    eigenvalues, eigenvectors = eigh(A)  # eigh for symmetric matrix, sorted ascending
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    V = eigenvectors[:, :k]  # Top k eigenvectors
    P = V @ V.T
    return P, eigenvalues


def run_sgd(A, P, x0, eta, sigma, T):
    """
    Run SGD with noisy gradients for the quadratic objective 0.5 x^T A x.
    Record loss and alignment over T steps.
    """
    d = A.shape[0]
    x = x0.copy()
    losses = np.zeros(T + 1)
    alignments = np.zeros(T + 1)

    # Initial metrics
    g_true = A @ x
    losses[0] = 0.5 * x.T @ g_true
    if np.linalg.norm(g_true) > 1e-10:
        alignments[0] = np.linalg.norm(P @ g_true) ** 2 / np.linalg.norm(g_true) ** 2
    else:
        alignments[0] = 1.0  # Arbitrary if gradient is zero

    for t in range(1, T + 1):
        g_true = A @ x
        noise = np.random.normal(0, sigma, d)
        g_noisy = g_true + noise
        x = x - eta * g_noisy

        # Metrics
        g_true = A @ x  # Recompute after update
        losses[t] = 0.5 * x.T @ g_true
        if np.linalg.norm(g_true) > 1e-10:
            alignments[t] = np.linalg.norm(P @ g_true) ** 2 / np.linalg.norm(g_true) ** 2
        else:
            alignments[t] = 1.0

    return losses, alignments


def plot_results(steps, losses, alignments, save_path="results.png"):
    """
    Plot loss vs step and alignment vs step and save the plot to a file.
    """
    # Ensure matplotlib works in non-interactive mode
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Loss
    ax1.plot(steps, losses)
    ax1.set_yscale('log')
    ax1.set_xlabel('Step', fontsize=30)
    ax1.set_ylabel('Loss', fontsize=30)
    ax1.set_title('Loss vs Step', fontsize=30)
    ax1.tick_params(axis='x', labelsize=30)  
    ax1.tick_params(axis='y', labelsize=30)  


    # Plot Alignment
    ax2.plot(steps, alignments)
    ax2.set_xlabel('Step', fontsize=30)
    ax2.set_ylabel('Alignment', fontsize=30)
    ax2.set_title('Alignment vs Step', fontsize=30)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', labelsize=30)  
    ax2.tick_params(axis='y', labelsize=30)  

    plt.tight_layout()
    
    # Save the figure to a file instead of showing it
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

    # Close the figure to free up memory
    plt.close(fig)

def analyze_phase_decay(steps, alignments, cut_step, alpha_exp=1.0, alpha_poly=1.0, save_path="phase1_sklearn.png"):
    """
    Perform L2 regularized fitting using sklearn.linear_model.Ridge.
    
    Args:
        alpha_exp (float): Regularization strength for exponential fit.
        alpha_poly (float): Regularization strength for polynomial fit.
    """
    degree = 3
    t_data = steps[:cut_step]
    y_data = alignments[:cut_step]
    
    valid_mask = y_data > 1e-10
    t_fit = t_data[valid_mask].reshape(-1, 1) 
    y_fit = y_data[valid_mask]
    
    # --- A. Exponential Ridge Fit ---
    # Model: ln(y) = w*t + b
    log_y = np.log(y_fit)
    
    # Use StandardScaler to ensure robust regularization for exponential fitting
    pipe_exp = make_pipeline(StandardScaler(), Ridge(alpha=alpha_exp))
    pipe_exp.fit(t_fit, log_y)
    
    # Predict and transform back to original scale
    log_y_pred = pipe_exp.predict(t_fit)
    y_pred_exp = np.exp(log_y_pred)
    r2_exp = r2_score(y_fit, y_pred_exp)

    ridge_step = pipe_exp.named_steps['ridge']
    scaler_step = pipe_exp.named_steps['standardscaler']
    
    # Calculate the decay rate (slope) accounting for the scaler
    raw_coef = ridge_step.coef_[0]
    scale_factor = scaler_step.scale_[0] 
    exp_slope = raw_coef / scale_factor  
    
    # --- B. Polynomial Ridge Fit ---
    # Pipeline: PolynomialFeatures -> StandardScaler -> Ridge
    # Scaling is crucial for Ridge regularization to affect higher-order terms correctly.
    pipe_poly = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(), 
        Ridge(alpha=alpha_poly)
    )
    pipe_poly.fit(t_fit, y_fit)
    
    y_pred_poly = pipe_poly.predict(t_fit)
    r2_poly = r2_score(y_fit, y_pred_poly)
    
    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Flatten for matplotlib plotting
    t_plot = t_fit.flatten()
    
    # Subplot 0: Original Coordinates Comparison
    axes[0].plot(t_plot, y_fit, 'k.', alpha=0.2, label='Data', markersize=20)
    axes[0].plot(t_plot, y_pred_exp, 'r-', linewidth=2, label=f'Ridge Exp ($R^2={r2_exp:.3f}$)')
    axes[0].plot(t_plot, y_pred_poly, 'b--', linewidth=2, label=f'Ridge Poly ($R^2={r2_poly:.3f}$)')
    axes[0].set_title(f"Fit Comparison", fontsize=30)
    axes[0].set_xlabel("Step", fontsize=30)
    axes[0].set_ylabel("Alignment", fontsize=30)
    axes[0].legend()
    
    # Subplot 1: Semi-Log (Checking Exponential Decay)
    axes[1].plot(t_plot, y_fit, 'k.', alpha=0.2, markersize=20)
    axes[1].plot(t_plot, y_pred_exp, 'r-', linewidth=2)
    axes[1].set_yscale('log')
    axes[1].set_title(f"Semi-Log Plot (Exp Fit)", fontsize=30)
    axes[1].set_xlabel("Step", fontsize=30)
    axes[1].set_ylabel("Log(Alignment)", fontsize=30)

    # Subplot 2: Linear (Checking Polynomial fit)
    axes[2].plot(t_plot, y_fit, 'k.', alpha=0.2, markersize=20)
    axes[2].plot(t_plot, y_pred_poly, 'b--', linewidth=2)
    axes[2].set_title(f"Poly Fit (Deg {degree})", fontsize=30)
    axes[2].set_xlabel("Step", fontsize=30)
    axes[2].set_ylabel("Alignment", fontsize=30)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Analysis saved to {save_path}")
    print(f"Exponential R2: {r2_exp:.4f} | Polynomial R2: {r2_poly:.4f}")
    
    # Extract and print polynomial coefficients from the Ridge model
    poly_coefs = pipe_poly.named_steps['ridge'].coef_
    poly_intercept = pipe_poly.named_steps['ridge'].intercept_
    print(f"Poly Model: Intercept={poly_intercept:.4f}, Coefs={poly_coefs}")
    
    plt.close(fig)
    return r2_exp, r2_poly, exp_slope

def analyze_steady_state(alignments, steady_step):
    """
    Analyze the final steps of the simulation to determine the steady-state statistics.
    """
    steady_state_alignments = alignments[-steady_step:]
    expected_alignment_est = np.mean(steady_state_alignments)
    variance_alignment_est = np.var(steady_state_alignments)
    std_alignment_est = np.std(steady_state_alignments)

    print("\n--- Steady-State Analysis Results ---")
    print(f"Estimated E[Alignment]: {expected_alignment_est:.6f}")
    print(f"Estimated Var[Alignment]: {variance_alignment_est:.9f}")
    print(f"Estimated Std[Alignment]: {std_alignment_est:.6f}")
    
    return expected_alignment_est, variance_alignment_est

# Main simulation function
def simulate_quadratic_sgd(m, d=500, k=50, sigma=0.5, eta=0.003, T=30000, seed=42, threshold_ratio=0.1, steady_step=10000, output_dir="sgd_results"):
    np.random.seed(seed)

    m_root_dir = os.path.join(output_dir, f"m_{m}")
    m_dir = os.path.join(m_root_dir, f"seed_{seed}")
    os.makedirs(m_dir, exist_ok=True)

    A = generate_positive_definite_matrix(d, k, m)
    P, eigenvalues = compute_projection_matrix(A, k)

    # Check spectral gap condition
    lambda_k = eigenvalues[k - 1]
    lambda_k1 = eigenvalues[k]
    if lambda_k <= m * lambda_k1:
        raise ValueError(f"Condition not satisfied: {lambda_k} <= {m} * {lambda_k1}")

    print("Eigenvalues:", eigenvalues)

    x0 = np.random.randn(d)
    losses, alignments = run_sgd(A, P, x0, eta, sigma, T)

    steps = np.arange(T + 1)
    
    # Determine the 'cut_step' to separate the transient phase from steady state
    alignments_initial = alignments[1]
    y_min = np.min(alignments)       
    y_first_range = alignments_initial - y_min
    target_threshold = y_min + threshold_ratio * y_first_range
    cut_step = np.where(alignments < target_threshold)[0][0]
    
    filename_prefix = f"m{m}_seed{seed}" 
    
    # Perform Phase 1 (decay phase) analysis
    phase1_save_path = os.path.join(m_dir, f"{filename_prefix}_phase1_analysis.png")
    r2_exp, r2_poly, exp_slope = analyze_phase_decay(
        steps, alignments, cut_step, 
        alpha_exp=1.0, alpha_poly=1.0, 
        save_path=phase1_save_path,
    )
    
    # Plot global results
    global_save_path = os.path.join(m_dir, f"{filename_prefix}_results_global.png")
    plot_results(steps, losses, alignments, save_path=global_save_path)
    
    # Calculate steady-state metrics
    expected_limit, variance_limit = analyze_steady_state(alignments, steady_step)

    return {
        'm': m,
        'E_Alignment': expected_limit,
        'Var_Alignment': variance_limit,
        'Exp_R2': r2_exp,
        'Poly_R2': r2_poly,
        'Exp_Decay_Rate': exp_slope,
        'Cut_Step': cut_step,
        'Lambda_k_ratio': eigenvalues[k-1] / eigenvalues[k]
    }


def run_m_sweep(m_values, **kwargs):
    seed = kwargs['seed']
    all_results = []
    output_dir = kwargs.get('output_dir', "sgd_results")
    os.makedirs(output_dir, exist_ok=True)

    for m in m_values:
        try:
            result = simulate_quadratic_sgd(m=m, **kwargs)
            all_results.append(result)
        except Exception as e:
            print(f"Error running m={m}: {e}")
            all_results.append({'m': m, 'Error': str(e)})

    # Save all results for this seed to a CSV file
    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, f"seed_{seed}_m_sweep_summary.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n All results saved to {csv_path}")
    
    return df_results

if __name__ == "__main__":
    M_VALUES = [5, 10, 20, 50, 100, 200, 300, 400, 500]
    SEEDS_TO_RUN = [42, 87, 568, 12138, 1101, 70425, 4008001]
    FIXED_PARAMS = {
        'd': 500,
        'k': 50,
        'sigma': 0.5,
        'eta': 0.003,
        'T': 30000,          
        'steady_step': 10000, 
    }

    all_runs_summary = []

    for current_seed in SEEDS_TO_RUN:
        print(f"\n=======================================================")
        print(f"       Starting M-Sweep for SEED: {current_seed}       ")
        print(f"=======================================================\n")
        
        seed_params = FIXED_PARAMS.copy()
        seed_params['seed'] = current_seed
        
        # Call the m-value sweep
        results_df = run_m_sweep(M_VALUES, **seed_params)
        
        # Summarize results for the current seed
        results_df['Global_Seed'] = current_seed
        all_runs_summary.append(results_df)

    # Aggregate all seeds into a single final report
    if all_runs_summary:
        final_df = pd.concat(all_runs_summary, ignore_index=True)
        final_csv_path = os.path.join(FIXED_PARAMS.get('output_dir', "sgd_results"), "ALL_SEEDS_summary.csv")
        final_df.to_csv(final_csv_path, index=False)
        print(f"\n--- All {len(SEEDS_TO_RUN)} seeds aggregated to {final_csv_path} ---")
