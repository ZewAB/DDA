# ============================================================================= 
# Script: DDA for ATT Estimation — Simulation 2 (misspecified model)
# File: DDA_main_sim2.jl
# Experiments executed on AMD EPYC 7302P (16-core / 32 GB RAM), Linux
#
# SIM2 (misspecified model) — PARALLEL (BO ONCE PER CONFIG)
#
# Purpose:
#   Estimate the Average Treatment Effect on the Treated (ATT) when the outcome
#   model is misspecified. The estimator is “hybrid” in the sense that it uses:
#     (i) covariate-balancing weights χ on controls, and
#     (ii) a WL1L0-regularized control outcome model fit via SCPRSM.
#
# Experiment design:
#   1) Hyperparameters (α, λ, r) are tuned ONCE per (n, p, solver) configuration
#      using Bayesian optimization (BO) on a single tuning dataset with seed_tune = 12345.
#
#   2) With (α_opt, λ_opt, r_opt) fixed from step (1), performance is evaluated
#      over 400 Monte Carlo replications in parallel:
#         seed_i = seed + gid,  gid = 1,…,400.
#      The wrapper sets seed = 12345 ⇒ seed_i = 12345 + gid.
#
# Batching / parallel scheduling:
#   - n_rep must be 400 = 10 × 40 (10 batches of 40 replications).
#   - Each batch is evaluated via pmap over workers(), and checkpointed to CSV.
#
# Recommended run:
#   JULIA_WORKER_TIMEOUT=300 julia --project=. --bind-to 127.0.0.1 -p 12 DDA_wrapper_sim2.jl
#
# Method overview (core components):
#
#   A) Control balancing weights χ 
#      Solvers supported in this script:
#        - COSMO (open-source)
#        - Mosek (commercial; free academic license available)
#        - COPT  (commercial; free academic license available)
#
#   B) Outcome model on controls (WL1L0–SCPRSM)
#      Fit a regularized control regression using SCPRSM with translation proximal
#      operators:
#        - α controls L1 vs L0 mixing,
#        - λ controls overall regularization strength,
#        - r controls the SCPRSM relaxation.
#
#   C) ATT estimator 
#      μ̂0 = balance_target' b̂0 + Σ_{i : X_i = 0} χ_i (Y_i − W_i' b̂0)
#      τ̂  = mean(Y | X=1) − μ̂0
#
# Outputs:
#   - RMSE across 400 replications
#   - average runtime per replication, total runtime
#   - tuned hyperparameters per configuration / solver
#
#Load the required packages (if they are not installed, please install them before loading)
# =============================================================================

using Distributed
using Statistics, LinearAlgebra, Distributions, Random
using BayesianOptimization, GaussianProcesses, ProximalOperators
using JuMP, DataFrames, CSV
using Mosek, MosekTools, COPT, COSMO

# -----------------------------------------------------------------------------
# Worker initialization (ONLY if Julia was started without -p)
#
# If Julia was launched with -p <nworkers>, then nprocs()>1 and this
# block is skipped. Otherwise, 12 worker processes are added.
# -----------------------------------------------------------------------------
if nprocs() == 1
    proj = Base.active_project()
    addprocs(12; exeflags="--project=$(proj)", restrict=true)
end

# Load all dependencies on every worker so that pmap can execute functions there.
@everywhere begin
    using Statistics, LinearAlgebra, Distributions, Random
    using BayesianOptimization, GaussianProcesses, ProximalOperators
    using JuMP, DataFrames, CSV
    using Mosek, MosekTools, COPT, COSMO
end

# -----------------------------------------------------------------------------
# SIM2 misspecified data-generating process (DGP)
#
# Inputs:
#   rng : reproducibility
#   n   : sample size
#   p   : number of covariates
#
# Outputs (named tuple):
#   W   : covariate matrix (n × p)
#   Y   : outcome vector (n)
#   X   : treatment indicator (n), encoded as Int in {0,1}
#   catt : true ATT under this DGP (mean of tauW among treated-group units)
# -----------------------------------------------------------------------------
@everywhere function miss_data(rng::AbstractRNG, n::Int, p::Int)
    W = randn(rng, n, p)
    # Normalized to ensure E[tauW | X = 1] ≈ 1.
    # The constant tau = 1 is the theoretical target value.
    tauW = log.(1 .+ exp.(-2 .- 2 .* W[:, 1])) ./ 0.915
    ptreat = 1 .- exp.(-tauW)

    X = Int.(rand.(rng, Bernoulli.(ptreat)))

    idx1 = (X .== 1)
    catt = any(idx1) ? mean(tauW[idx1]) : 0.0

    Y = randn(rng, n) .+ tauW .* (2 .* X .- 1) ./ 2 .+ vec(sum(W[:, 1:10], dims=2))

    return (W=W, Y=Y, X=X, catt=catt)
end

# -----------------------------------------------------------------------------
# Helper: scale covariates, but do NOT scale binary columns
#   - Binary columns are left unscaled 
# -----------------------------------------------------------------------------
@everywhere function custom_scale(W::Matrix{Float64}, scale_W::Bool)
    if !scale_W
        return W
    end
    scl = [std(filter(!isnan, W[:, i]), corrected=false) for i in 1:size(W, 2)]
    is_binary = [all(w -> (w == 0 || w == 1) || isnan(w), col) for col in eachcol(W)]
    scl[is_binary] .= 1.0
    scl[scl .== 0.0] .= 1.0
    return W ./ scl'
end

# -----------------------------------------------------------------------------
# Balancing weights χ on controls 
#
# Inputs:
#   M              : generic placeholder control covariate matrix (n0 × p) after scaling
#   balance_target : treated-group covariate mean vector (p) after scaling
#   solver_choice  : "Mosek", "COPT", or "COSMO" (QP backend)
#      ζ trades off balance (δ) and weight stability (‖χ‖₂²).
#
# Optimization problem:
#   minimize_{χ, δ ≥ 0}   ζ·δ² + (1−ζ)‖χ‖₂²
#   subject to            1ᵀχ = 1,
#                         10^{-4} ≤ χ_i ≤ 1−10^{-4},
#                         ‖Mᵀχ − balance_target‖_∞ ≤ δ
#
# Implementation detail:
#   The ∞-norm constraint is encoded as two vector inequalities:
#     M'χ − balance_target ≤ δ
#    −M'χ + balance_target ≤ δ
# -----------------------------------------------------------------------------
@everywhere function optimize_chi(M::Matrix{Float64},
                                 balance_target::Vector{Float64},
                                 solver_choice::AbstractString;
                                 ζ::Float64 = 0.5)

    n_weights, _ = size(M)

    model =
        solver_choice == "Mosek" ? JuMP.Model(Mosek.Optimizer) :
        solver_choice == "COPT"  ? JuMP.Model(COPT.Optimizer)  :
        solver_choice == "COSMO" ? JuMP.Model(COSMO.Optimizer) :
        error("Unsupported solver_choice=$solver_choice (expected Mosek/COPT/COSMO)")

    JuMP.set_silent(model)

    # χ with constraints for numerical stability
    JuMP.@variable(model, 1e-4 <= chi[1:n_weights] <= 1 - 1e-4)

    # δ is a nonnegative slack capturing max imbalance in the ∞-norm
    JuMP.@variable(model, delta >= 0)

    # Objective
    JuMP.@objective(model, Min, ζ * delta^2 + (1 - ζ) * sum(chi .^ 2))

    # Constraint: weights sum to 1
    JuMP.@constraint(model, sum(chi) == 1)

    # Infinity-norm balance constraints: ‖M'χ − balance_target‖∞ ≤ δ
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
# WL1L0–SCPRSM ATT estimator: returns catt_hat (scalar)
#
# Inputs:
#   α, λ, r         : WL1L0 / SCPRSM hyperparameters
#   WX, YX          : scaled control covariates and outcomes
#   Y, X            : full-sample outcomes and treatment indicators
#   balance_target  : treated-group covariate mean (scaled)
#   chi             : control balancing weights χ
#
# Output:
# catt_hat : ATT estimate = mean(Y | X=1) − estimated μ0 
# Numerical notes:
#   - Initialization uses the covariance between controls and outcomes.
#   - Step sizes (γ, δ) are adapted via backtracking line search for stability.
# -----------------------------------------------------------------------------
@everywhere function scprsm_tauhat(α::Float64, λ::Float64, r::Float64,
                                  WX::Matrix{Float64}, YX::Vector{Float64},
                                  Y::Vector{Float64}, X::Vector{Int},
                                  balance_target::Vector{Float64}, chi::Vector{Float64};
                                  tol::Float64 = 5e-4, maxit::Int = 5000)
    # Calculate covariances for initializations to SCPRSM
    covar = cov(WX, YX)

    # Initialization for translated variables
    u = covar[:, 1] * 0.0001
    v = covar[:, 1] * 0.0001
    uvcurr = zero(WX[1, :])

    # Primal variables for the two SCPRSM blocks
    c = zero(WX[1, :])
    d = zero(WX[1, :])

    # Dual variables for SCPRSM updates
    m  = zero(c)
    m2 = zero(c)
    l  = zero(d)
    l2 = zero(d)

    # Regularization weights: L1 and L0 components
    λ₁ = λ * α              # L1 strength
    λ₂ = λ * (1.0 - α)      # L0 strength

    # Proximal operator building blocks for the L1 block
    hγ = LeastSquares(WX, YX)        # squared-loss term for L1
    fγ = Translate(hγ, v)            # translation by v
    gγ = NormL1(λ₁)                  # L1 penalty

    # Proximal operator building blocks for the L0 block
    hδ = LeastSquares(WX, YX)        # squared-loss term for L0
    fδ = Translate(hδ, u)            # translation by u
    gδ = NormL0(λ₂)                  # L0 penalty

    # Backtracking line search parameters and initial step sizes
    ℸ = 0.5    # shrinkage factor
    γ = 0.9    # learning rate for L1 block
    δ = 0.9    # learning rate for L0 block

    # Loss used in the backtracking conditions
    loss(x) = 0.5 * norm(WX * x - YX)^2

    for it = 1:maxit
        # -------------------------
        # γ block (L1 part): choose γ by backtracking, then apply prox updates
        # -------------------------
        gradγ = WX' * (WX * c - YX)
        while loss(u) > (loss(c) + gradγ' * (-c) + (1 / (2γ)) * norm(-c)^2)
            γ *= ℸ
        end

        # Save current combined iterate to evaluate dual residual later
        uvcurr = u + v

        # SCPRSM L1 updates 
        prox!(c, fγ, u - m, γ)
        # First dual update for the L1 block
        m .+= r * (c - u)
        prox!(u, gγ, c + m, γ)
        # Second dual update for the L1 block
        m2 .+= r * (c - u)

        # -------------------------
        # δ block (L0 part): choose δ by backtracking, then apply prox updates
        # -------------------------
        gradδ = WX' * (WX * d - YX)
        while loss(v) > (loss(d) + gradδ' * (-d) + (1 / (2δ)) * norm(-d)^2)
            δ *= ℸ
        end
# SCPRSM L0 updates
        prox!(d, fδ, v - l, δ)
        # First dual update for the L0 block
        l .+= r * (d - v)
        prox!(v, gδ, d + l, δ)

        # -------------------------
        # Stopping criterion 
        # -------------------------
        dualres = (u + v) - uvcurr
        reldualres = dualres / norm(((u + v) + uvcurr) / 2)
        if it % 5 == 2 && norm(reldualres) <= tol
            break
        end

        # Second dual update for the L0 block
        l2 .+= r * (d - v)
    end

    # Final coefficient estimate for the control outcome model
    bfit = u + v
# Estimate of μ0 
    mu_l1l0 = dot(balance_target, bfit)
    residuals = YX - WX * bfit
    mu_residual = sum(chi .* residuals)
    mu_hat = mu_l1l0 + mu_residual

    # ATT estimate: treated mean minus estimated μ0
    eta1 = mean(Y[X .== 1])
    catt_hat = eta1 - mu_hat
    return catt_hat
end

# -----------------------------------------------------------------------------
# 1) Hyperparameter tuning via BO (run ONCE per configuration)
#
# Given (n, p, solver_choice) and bounds for (α, λ, r), this routine:
#   - generates one fixed tuning dataset (seed_tune = 12345),
#   - computes χ on that dataset,
#   - runs Bayesian optimization to minimize |catt_hat - catt_true|.
# -----------------------------------------------------------------------------
function tune_once_bo_fixedseed(n::Int, p::Int,
                                arg1::Float64, arg2::Float64, arg3::Float64,
                                arg4::Float64, arg5::Float64, arg6::Float64,
                                solver_choice::AbstractString;
                                bo_maxiter::Int = 250)

    seed_tune = 12345
    rng_tune = MersenneTwister(seed_tune)

    d0 = miss_data(rng_tune, n, p)
    W0, Y0, X0, catt0 = d0.W, d0.Y, d0.X, d0.catt

    idx1 = findall(==(1), X0)
    idx0 = findall(==(0), X0)

    low  = [arg1, arg2, arg3]
    high = [arg4, arg5, arg6]

    # Fallback (rare): if no treated or no controls, return midpoint
    if isempty(idx1) || isempty(idx0)
        return (low .+ high) ./ 2
    end

    # Construct tuning inputs: scaled covariates, balance_target, controls, and χ
    W0s = custom_scale(W0, true)
    balance_target0 = vec(mean(W0s[idx1, :], dims=1))
    WX0 = W0s[idx0, :]
    YX0 = Y0[idx0]
    chi0 = optimize_chi(WX0, balance_target0, solver_choice)

    # BO objective: absolute ATT estimation error on the tuning dataset
    function bo_obj(par::Vector{Float64})
        catt_hat0 = scprsm_tauhat(par[1], par[2], par[3], WX0, YX0, Y0, X0, balance_target0, chi0)
        return abs(catt_hat0 - catt0)
    end

    best_par = (low .+ high) ./ 2

    # Robust BO: retry up to 3 times to handle occasional GP numerical issues
    for attempt in 1:3
        try
            # Seed BO's internal randomness (does NOT change miss_data seed_tune)
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
                UpperConfidenceBound(),
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
#
# Returns squared error (catt_hat - catt_true)^2 for RMSE aggregation.
# -----------------------------------------------------------------------------
@everywhere function run_single_rep_fixed(seed_i::Int,
                                         n::Int, p::Int,
                                         solver_choice::AbstractString,
                                         α_opt::Float64, λ_opt::Float64, r_opt::Float64)

    rng_i = MersenneTwister(seed_i)
    d = miss_data(rng_i, n, p)
    W, Y, X, catt = d.W, d.Y, d.X, d.catt

    idx1 = findall(==(1), X)
    idx0 = findall(==(0), X)
    if isempty(idx1) || isempty(idx0)
        return 0.0
    end

    # Construct evaluation inputs: scaled covariates, balance_target and χ
    Ws = custom_scale(W, true)
    balance_target = vec(mean(Ws[idx1, :], dims=1))
    WX = Ws[idx0, :]
    YX = Y[idx0]
    chi = optimize_chi(WX, balance_target, solver_choice)

    # ATT estimate with fixed tuned hyperparameters
    catt_hat = scprsm_tauhat(α_opt, λ_opt, r_opt, WX, YX, Y, X, balance_target, chi)

    # Return squared error for RMSE aggregation
    return (catt_hat - catt)^2
end

# -----------------------------------------------------------------------------
# Main driver:
#   1) Tune once via BO using seed_tune = 12345
#   2) Evaluate 400 replications in parallel, checkpointing each batch
# -----------------------------------------------------------------------------
function main(n::Int, p::Int,
              arg1::Float64, arg2::Float64, arg3::Float64,
              arg4::Float64, arg5::Float64, arg6::Float64,
              n_rep::Int, solver_choice::AbstractString;
              seed::Int = 12345,
              bo_maxiter::Int = 250)

    nbatches = 10
    batch_size = 40
    @assert n_rep == nbatches * batch_size "Require n_rep=400 (10×40)"

    seed_tune = 12345

    println("\n" * "="^90)
    println("SIM2 | STARTING PARALLEL DDA (BO ONCE, THEN FIXED) | MISSPECIFIED")
    println("n=$n p=$p | Solver=$solver_choice | Workers=$(nworkers()) | Reps=$n_rep (10×40)")
    println("Bounds: α∈[$arg1,$arg4], λ∈[$arg2,$arg5], r∈[$arg3,$arg6] | bo_maxiter=$bo_maxiter")
    println("="^90)

    # 1) Tune once 
    best_par = tune_once_bo_fixedseed(n, p, arg1, arg2, arg3, arg4, arg5, arg6, solver_choice;
                                      bo_maxiter=bo_maxiter)

    α_opt, λ_opt, r_opt = best_par
    println("SIM2 | TUNED ONCE: α=$α_opt, λ=$λ_opt, r=$r_opt")

    # 2) Evaluate in parallel (batched)
    wp = CachingPool(workers())   # reuse workers efficiently across pmap calls

    all_sqerr = Float64[]
    sizehint!(all_sqerr, n_rep)

    total_time = @elapsed begin
        for b in 1:nbatches
            lo = (b - 1) * batch_size + 1
            hi = b * batch_size
            gids = lo:hi

            println("SIM2 | Batch $b / $nbatches (reps $lo:$hi)")

            # Each gid defines one independent replication with seed_i = seed + gid
            batch_sqerr = pmap(wp, gids) do gid
                seed_i = seed + gid
                run_single_rep_fixed(seed_i, n, p, solver_choice, α_opt, λ_opt, r_opt)
            end

            append!(all_sqerr, batch_sqerr)

            # Checkpoint per batch for fault tolerance and later aggregation
            CSV.write(
                "sim2_checkpoint_eval_dda_$(lowercase(String(solver_choice)))_n$(n)_p$(p)_batch$(b).csv",
                DataFrame(gid = collect(gids), seed = seed .+ collect(gids), sqerr = batch_sqerr)
            )
        end
    end

    rmse_final = sqrt(mean(all_sqerr))
    avg_time = total_time / n_rep

    println("==========================================")
    println("SIM2 | DONE (BO once)")
    println("RMSE: $rmse_final")
    println("Total time (s): $total_time")
    println("Avg time per rep (s): $avg_time")
    println("==========================================")

    return DataFrame(
        n = n, p = p,
        Solver = String(solver_choice),
        seed = seed,
        seed_tune = seed_tune,
        n_rep = n_rep,
        RMSE = rmse_final,
        T_per_iter = avg_time,
        TotT = total_time,
        Arg1 = arg1, Arg2 = arg2, Arg3 = arg3,
        Arg4 = arg4, Arg5 = arg5, Arg6 = arg6,
        bo_maxiter = bo_maxiter,
        alp = α_opt, lambda = λ_opt, r = r_opt
    )
end
