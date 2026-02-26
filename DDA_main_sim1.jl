# =============================================================================
# Script: DDA for ATT Estimation — Simulation 1
# File: DDA_main_sim1.jl
# Experiments executed on AMD EPYC 7302P (16-core / 32 GB RAM), Linux
#
# Overview:
#   This script implements the DDA estimator for the Average Treatment Effect on
#   the Treated (ATT) under Simulation 1. Each configuration is tuned ONCE using
#   Bayesian optimization (BO), and then evaluated over 400 Monte Carlo (MC)
#   replications in parallel.
#
# Main steps:
#   1) Data generation (Simulation 1 DGP):
#        - Covariates depend on a latent cluster indicator.
#        - Treatment assignment depends on the cluster indicator through eps.
#        - Outcome is linear in covariates plus a constant treatment effect τ=1.
#
#   2) Covariate scaling:
#        - Scale each (non-binary) covariate by its standard deviation.
#
#   3) Control balancing weights χ:
#        - Compute weights on the control group so weighted control covariate
#        - Solver backend is selected by solver_choice ∈ {COSMO, Mosek, COPT}.
#        - COSMO is open-source. 
#        - Mosek and COPT are commercial solvers, with free academic licenses available.
#   4) Outcome model on controls (WL1L0–SCPRSM):
#        - Fit a regularized control outcome regression using SCPRSM with
#          translation proximal operators.
#        - α mixes L1 and L0 penalties; λ sets overall regularization; r is the
#          SCPRSM relaxation/penalty parameter.
#        - Step sizes (γ, δ) are adjusted via backtracking line search.
#
#   5) Hyperparameter tuning (BO, once per configuration):
#        - Tune (α, λ, r) on a single fixed tuning dataset.
#        - Objective is |τ̂(α,λ,r) − τ| on the tuning dataset
#
#   6) Evaluation (MC replications in parallel):
#        - Using fixed tuned (α_opt, λ_opt, r_opt), compute squared error over
#          400 replications: seed_i = seed + gid, gid = 1..400.
#        - Run in parallel using pmap with a CachingPool(workers()).
#
# Reproducibility:
#   - Tuning seed: seed_tune = 12345.
#   - Evaluation seeds: seed_i = seed + gid, gid = 1..400.
#   - Wrapper typically sets seed=12345 ⇒ evaluation seed_i = 12345 + gid.
#
# Parallelism:
#   - If Julia is launched with -p 12, workers are pre-spawned (ids 2..13).
#   - If not, this script will add 12 workers automatically.
#
# Recommended run:
#   JULIA_WORKER_TIMEOUT=300 julia --project=. --bind-to 127.0.0.1 -p 12 DDA_wrapper_sim1.jl
#   (The environment variable increases allowed worker startup time, which can be
#    needed when precompiling/loading packages on workers.)
# Load the required packages (if they are not installed, please install them before loading)
# =============================================================================

using Distributed
using Statistics, LinearAlgebra, Distributions, Random
using BayesianOptimization, GaussianProcesses, ProximalOperators
using JuMP, DataFrames, CSV
using Mosek, MosekTools, COPT, COSMO

# -----------------------------------------------------------------------------
# Worker initialization (ONLY if Julia was started without -p)
# -----------------------------------------------------------------------------
# If Julia was started with -p 12, this block is skipped.
if nprocs() == 1
    proj = Base.active_project()  # use active Project.toml for all workers
    addprocs(12; exeflags="--project=$(proj)", restrict=true)
end

# Load dependencies on every worker (required for pmap)
@everywhere begin
    using Statistics, LinearAlgebra, Distributions, Random
    using BayesianOptimization, GaussianProcesses, ProximalOperators
    using JuMP, DataFrames, CSV
    using Mosek, MosekTools, COPT, COSMO
end

# -----------------------------------------------------------------------------
# Helper: scale covariates, but do NOT scale binary columns
# -----------------------------------------------------------------------------
@everywhere function custom_scale(W::Matrix{Float64}, scale_W::Bool)
    if !scale_W
        return W
    end
    # Function to scale non-binary features
    scl = [std(filter(!isnan, W[:, i]), corrected=false) for i in 1:size(W, 2)]
    # Detect binary columns (optionally with NaNs)
    is_binary = [all(w -> (w == 0 || w == 1) || isnan(w), col) for col in eachcol(W)]
    scl[is_binary] .= 1.0
    scl[scl .== 0.0] .= 1.0
    return W ./ scl'
end

# -----------------------------------------------------------------------------
# Balancing weights χ:
# -----------------------------------------------------------------------------
# Inputs:
#   M              : generic placeholder control covariate matrix (n0 × p) after scaling
#   balance_target : treated-group covariate mean vector (p × 1) after scaling
#   solver_choice  : one of {"Mosek","COPT","COSMO"} (JuMP optimizer backend)
#   ζ              : trade-off weight in the objective (default ζ=0.5)
#
# Optimization problem:
#   Choose weights χ ∈ ℝ^{n0} and slack δ ≥ 0 by solving:
#
#     minimize_{χ, δ ≥ 0}    ζ · δ^2 + (1 − ζ) · ‖χ‖_2^2
#     subject to             Σ_i χ_i = 1,
#                            1e-4 ≤ χ_i ≤ 1 − 1e-4    (numerical stability)
#                            ‖ M'χ − balance_target ‖_∞ ≤ δ.
#
# Interpretation:
#   • δ is the maximum absolute covariate imbalance after weighting.
#   • ‖χ‖_2^2 regularizes weights to avoid extreme solutions.
#   • ζ ∈ (0,1) controls the balance–stability trade-off.
#
# NOTE :
#   The ∞-norm constraint imposed on the vector (M'χ − balance_target):
#
#       −δ ≤ M'χ − balance_target ≤ δ
#
#   which is implemented as two vector inequalities:
#
#       M'χ − balance_target ≤ δ
#      −M'χ + balance_target ≤ δ
# -----------------------------------------------------------------------------
@everywhere function optimize_chi(M::Matrix{Float64},
                                 balance_target::Vector{Float64},
                                 solver_choice::AbstractString;
                                 ζ::Float64 = 0.5)

    n_weights, _ = size(M)

    # Select the convex solver 
    model =
        solver_choice == "Mosek" ? JuMP.Model(Mosek.Optimizer) :
        solver_choice == "COPT"  ? JuMP.Model(COPT.Optimizer)  :
        solver_choice == "COSMO" ? JuMP.Model(COSMO.Optimizer) :
        error("Unsupported solver_choice=$solver_choice (expected Mosek/COPT/COSMO)")

    JuMP.set_silent(model)

    # χ with constraints for numerical stability
    JuMP.@variable(model, 1e-4 <= chi[1:n_weights] <= 1 - 1e-4)

    # δ is a nonnegative slack capturing the infinity-norm imbalance
    JuMP.@variable(model, delta >= 0)

    # Objective: ζ * δ^2 + (1-ζ) * ||χ||_2^2
    JuMP.@objective(model, Min, ζ * delta^2 + (1 - ζ) * sum(chi .^ 2))

    # Constraint
    JuMP.@constraint(model, sum(chi) == 1)

    # Infinity-norm balance constraints: -δ ≤ M'χ - target ≤ δ
    JuMP.@constraint(model,  M' * chi .- balance_target .<= delta)
    JuMP.@constraint(model, -M' * chi .+ balance_target .<= delta)

    JuMP.optimize!(model)

    chi_sol = JuMP.value.(chi)
    # Check that the optimized balancing weights χ are nonnegative
    if any(chi_sol .< 0)
        error("chi invalid: contains negative values")
    end
    return chi_sol
end

# -----------------------------------------------------------------------------
# WL1L0 estimator: SCPRSM with translation proximal operators + backtracking
# -----------------------------------------------------------------------------
# Fits coefficients b on controls (WX, YX) for given hyperparameters (α, λ, r).
#
# Key ingredients:
#   - Coefficient decomposition via translation operators.
#   - Backtracking line search updates step sizes γ and δ for stability.
@everywhere function scprsm_fit(WX::Matrix{Float64}, YX::Vector{Float64},
                               α::Float64, λ::Float64, r::Float64;
                               tol::Float64 = 5e-4, maxit::Int = 5000)
    # Calculate covariances for initializations to SCPRSM
    covar = cov(WX, YX)

    # Initialization
    u = covar[:, 1] * 1e-4
    v = covar[:, 1] * 1e-4
    uvcurr = zero(WX[1, :])

    # Primal variables for SCPRSM blocks
    c = zero(WX[1, :])
    d = zero(WX[1, :])

    # Dual variables
    m  = zero(c)
    m2 = zero(c)
    l  = zero(d)
    l2 = zero(d)

    # Regularization weights
    λ₁ = λ * α           # L1 strength
    λ₂ = λ * (1.0 - α)   # L0 strength

    # Proximal operator building blocks
    hγ = LeastSquares(WX, YX)   # Loss function for L1
    fγ = Translate(hγ, v)       # Translation function L1
    gγ = NormL1(λ₁)             # L1 regularization

    hδ = LeastSquares(WX, YX)   # Loss function for L0
    fδ = Translate(hδ, u)       # Translation function L0
    gδ = NormL0(λ₂)             # L0 regularization

    # Backtracking parameters and initial step sizes
    ℸ, γ, δ = 0.5, 0.9, 0.9
    # Loss function for line search
    loss(x) = 0.5 * norm(WX * x - YX)^2

    for it = 1:maxit
        # γ-block: backtracking based line search
        gradγ = WX' * (WX * c - YX)
        while loss(u) > (loss(c) + gradγ' * (-c) + (1.0 / (2.0 * γ)) * norm(-c)^2)
            γ *= ℸ
        end

        # Save current combined iterate to evaluate dual residual later
        uvcurr = u + v
        # SCPRSM L1 updates
        prox!(c, fγ, u - m, γ)
        # First dual update L1
        m .+= r * (c - u)
        prox!(u, gγ, c + m, γ)
        # Second dual update L1
        m2 .+= r * (c - u)

        # δ-block: backtracking based line search
        gradδ = WX' * (WX * d - YX)
        while loss(v) > (loss(d) + gradδ' * (-d) + (1.0 / (2.0 * δ)) * norm(-d)^2)
            δ *= ℸ
        end

        # SCPRSM L0 updates
        prox!(d, fδ, v - l, δ)
        # First dual update L0
        l .+= r * (d - v)
        prox!(v, gδ, d + l, δ)

        # Stopping criterion
        dualres = (u + v) - uvcurr
        reldualres = dualres / norm(((u + v) + uvcurr) / 2)
        if it % 5 == 2 && norm(reldualres) <= tol
            break
        end
        # Second dual update L0
        l2 .+= r * (d - v)
    end

    # Final coefficient estimate
    return u + v
end

# -----------------------------------------------------------------------------
# Simulation 1 DGP helper: construct outcome coefficients and cluster shifts
# -----------------------------------------------------------------------------
@everywhere function make_b_and_delta(n::Int, p::Int, C::Float64, b_setup::Int, sparse_or_dense::Int)
    # b_setup controls sparsity/decay pattern in the outcome model
    b_raw = if b_setup == 1
        1.0 ./ sqrt.(1:p)                          # dense
    elseif b_setup == 2
        1.0 ./ (9 .+ (1:p))                        # harmonic
    elseif b_setup == 3
        vcat(fill(10.0, 10), fill(1.0, 90), fill(0.0, p - 100))  # moderately sparse
    elseif b_setup == 4
        vcat(fill(10.0, 10), fill(0.0, p - 10))    # very sparse
    else
        error("Invalid b_setup=$b_setup (expected 1..4)")
    end

    # Scale coefficients so ||b||_2 is proportional to C across configs
    b_main = C * b_raw / sqrt(sum(b_raw .^ 2))

    # delta_clust controls confounding strength via cluster-specific covariate shifts
    delta_clust = if sparse_or_dense == 1
        (40 / sqrt(n)) .* repeat([1.0; zeros(9)], p ÷ 10)  # sparse shifts
    elseif sparse_or_dense == 2
        (4 / sqrt(n)) * ones(p)                            # dense shifts
    else
        error("Invalid sparse_or_dense=$sparse_or_dense (use 1=sparse, 2=dense)")
    end

    return b_main, delta_clust
end
# -----------------------------------------------------------------------------
# Simulation 1 DGP: generate a single dataset
# -----------------------------------------------------------------------------
@everywhere function gen_one_dataset(seed_i::Int, n::Int, p::Int,
                                     b_main::Vector{Float64}, delta_clust::Vector{Float64},
                                     eps::Float64)
    Random.seed!(seed_i)

    # Treatment effect is fixed at 1.0 in this DGP
    tau = 1.0

    # Cluster indicator
    CLUST = rand(Bernoulli(0.5), n)
    probs = eps .+ (1 .- 2 * eps) .* CLUST
    X = Int.(rand.(Bernoulli.(probs)))

    # Covariates shifted by cluster-dependent delta_clust
    W = randn(n, p) .+ CLUST .* delta_clust'

    # Outcome: linear in W + noise + treatment effect
    Y = W * b_main .+ randn(n) .+ tau .* X

    return (W=W, Y=Y, X=X, catt=tau)
end

# -----------------------------------------------------------------------------
# 1) Hyperparameter tuning via BO ONCE per configuration
# -----------------------------------------------------------------------------
# BO objective (simulation-only, since catt is known):
#   minimize | tau_hat(α,λ,r) - catt |
function tune_once_bo(seed_tune::Int,
                      n::Int, p::Int, C::Float64, eps::Float64, b_setup::Int,
                      arg1::Float64, arg2::Float64, arg3::Float64,
                      arg4::Float64, arg5::Float64, arg6::Float64,
                      sparse_or_dense::Int,
                      solver_choice::AbstractString;
                      bo_maxiter::Int = 250)

    # Fixed tuning dataset for this configuration
    b_main, delta_clust = make_b_and_delta(n, p, C, b_setup, sparse_or_dense)
    d = gen_one_dataset(seed_tune, n, p, b_main, delta_clust, eps)
    W, Y, X, catt = d.W, d.Y, d.X, d.catt

    idx1 = findall(==(1), X)
    idx0 = findall(==(0), X)

    low  = [arg1, arg2, arg3]
    high = [arg4, arg5, arg6]

    # Fallback (rare): if no treated or no controls, return midpoint
    if isempty(idx1) || isempty(idx0)
        return (low .+ high) ./ 2
    end

    # Build balancing target, control design matrix and control otcome
    W_scl = custom_scale(W, true)
    balance_target = reshape(mean(W_scl[idx1, :], dims=1), :)
    WX = W_scl[idx0, :]
    YX = Y[idx0]

    # weights χ
    chi = optimize_chi(WX, balance_target, solver_choice)

    # BO objective: absolute ATT estimation error on the tuning dataset
    function bo_obj(par::Vector{Float64})
        b = scprsm_fit(WX, YX, par[1], par[2], par[3])
        mu_hat = dot(balance_target, b) + sum(chi .* (YX - WX * b))
        tau_hat = mean(Y[idx1]) - mu_hat
        return abs(tau_hat - catt)
    end

    best_par = (low .+ high) ./ 2

    # Robust BO: retry up to 3 times to handle occasional GP numerical issues
    for attempt in 1:3
        try
            # Seeds BO's internal randomness
            Random.seed!(seed_tune + 10_000 * attempt)

            modeloptimizer = MAPGPOptimizer(
                every=30,
                noisebounds=[-1.0, 10.0],
                kernbounds=[[-3.0, -3.0, -3.0, 0.0], [6.0, 8.0, 8.0, 8.0]],
                maxeval=40
            )

            model = ElasticGPE(
                3,
                mean=MeanConst(0.0),
                kernel=SEArd([2.0, 3.0, 3.0], 1.0),
                logNoise=4.0,
                capacity=1000
            )

            opt = BOpt(
                par -> bo_obj(par),
                model,
                UpperConfidenceBound(),   # acquisition: GP-UCB
                modeloptimizer,
                low, high,
                repetitions=4,
                maxiterations=bo_maxiter,
                sense=Min,
                verbosity=Silent
            )

            re = boptimize!(opt)
            best_par = re[2]
            break
        catch e
            # Common numeric failures in GP fitting
            if (e isa LinearAlgebra.SingularException) || (e isa DomainError)
                continue
            else
                rethrow()
            end
        end
    end

    return best_par
end

# -----------------------------------------------------------------------------
# 2) One evaluation replication (NO BO): uses fixed tuned (α_opt, λ_opt, r_opt)
# -----------------------------------------------------------------------------
@everywhere function run_single_rep_fixed(seed_i::Int,
                                         n::Int, p::Int, C::Float64, eps::Float64, b_setup::Int,
                                         sparse_or_dense::Int,
                                         solver_choice::AbstractString,
                                         α_opt::Float64, λ_opt::Float64, r_opt::Float64)

    b_main, delta_clust = make_b_and_delta(n, p, C, b_setup, sparse_or_dense)
    d = gen_one_dataset(seed_i, n, p, b_main, delta_clust, eps)
    W, Y, X, catt = d.W, d.Y, d.X, d.catt

    idx1 = findall(==(1), X)
    idx0 = findall(==(0), X)
    if isempty(idx1) || isempty(idx0)
        return 0.0
    end

    W_scl = custom_scale(W, true)
    balance_target = reshape(mean(W_scl[idx1, :], dims=1), :)
    WX = W_scl[idx0, :]
    YX = Y[idx0]

    chi = optimize_chi(WX, balance_target, solver_choice)

    # Fit control outcome model with fixed hyperparameters
    b = scprsm_fit(WX, YX, α_opt, λ_opt, r_opt)

    # Hybrid (regression + residual reweighting) estimate of μ0 for treated
    mu_hat = dot(balance_target, b) + sum(chi .* (YX - WX * b))
    tau_hat = mean(Y[idx1]) - mu_hat

    # Return squared error for RMSE aggregation
    return (tau_hat - catt)^2
end

# -----------------------------------------------------------------------------
# Main:
#   1) Tune once
#   2) Evaluate 400 reps in parallel (10 batches × 40 reps)
# -----------------------------------------------------------------------------
function main(n::Int, p::Int, C::Float64, eps::Float64, b_setup::Int,
              arg1::Float64, arg2::Float64, arg3::Float64,
              arg4::Float64, arg5::Float64, arg6::Float64,
              n_rep::Int, sparse_or_dense::Int, solver_choice::AbstractString;
              seed::Int = 12345,
              bo_maxiter::Int = 250)

    nbatches = 10
    batch_size = 40
    @assert n_rep == nbatches * batch_size "Require n_rep=400 (10×40)"

    println("\n" * "="^80)
    println("SIM1 | STARTING PARALLEL DDA (BO ONCE, THEN FIXED)")
    println("n=$n p=$p C=$C eps=$eps (\\varrho=eps) b_setup=$b_setup δ=$(sparse_or_dense==1 ? "sparse" : "dense")")
    println("Solver: $solver_choice | Workers: $(nworkers()) | Reps: $n_rep (10×40)")
    println("Seed(base) = $seed | eval seed_i = seed + gid, gid=1..$n_rep")
    println("BO: tuned once on FIXED seed_tune = 12345 | bo_maxiter = $bo_maxiter")
    println("="^80)

    #   1) Tune once (master)
    seed_tune = 12345
    best_par = tune_once_bo(seed_tune, n, p, C, eps, b_setup,
                            arg1, arg2, arg3, arg4, arg5, arg6,
                            sparse_or_dense, solver_choice;
                            bo_maxiter=bo_maxiter)

    α_opt, λ_opt, r_opt = best_par
    println("TUNED ONCE: α=$α_opt, λ=$λ_opt, r=$r_opt")

    #  2) Evaluate in parallel 
    wp = CachingPool(workers())  # reuse worker processes efficiently
    all_sqerr = Float64[]
    sizehint!(all_sqerr, n_rep)

    total_time = @elapsed begin
        for b in 1:nbatches
            lo = (b - 1) * batch_size + 1
            hi = b * batch_size
            gids = lo:hi

            println("Batch $b / $nbatches (reps $lo:$hi)")

            # Parallel map: each gid is evaluated independently on a worker
            batch_sqerr = pmap(wp, gids) do gid
                seed_i = seed + gid
                run_single_rep_fixed(seed_i, n, p, C, eps, b_setup,
                                     sparse_or_dense, solver_choice,
                                     α_opt, λ_opt, r_opt)
            end

            append!(all_sqerr, batch_sqerr)

            # Checkpoint per batch for later aggregation
            CSV.write(
                "sim1_checkpoint_eval_dda_$(lowercase(String(solver_choice)))_n$(n)_p$(p)_b$(b_setup)_eps$(eps)_delta$(sparse_or_dense)_batch$(b).csv",
                DataFrame(gid = collect(gids), sqerr = batch_sqerr)
            )
        end
    end

    rmse_final = sqrt(mean(all_sqerr))
    avg_time = total_time / n_rep

    println("==========================================")
    println("SIM1 | DONE (BO once)")
    println("RMSE: $rmse_final")
    println("Total time (s): $total_time")
    println("Avg time per rep (s): $avg_time")
    println("==========================================")

    return DataFrame(
        RMSE = rmse_final,
        T_per_iter = avg_time,
        TotT = total_time,
        Solver = String(solver_choice),
        n = n, p = p, C = C, eps = eps,
        b_setup = b_setup, delta_setting = sparse_or_dense,
        n_rep = n_rep,
        seed = seed,
        bo_seed = seed_tune,
        alp = α_opt, lambda = λ_opt, r = r_opt,
        bo_maxiter = bo_maxiter
    )
end
