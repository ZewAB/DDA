# =============================================================================
# Many-cluster inference simulation - parallel
# Experiments executed on AMD EPYC 7302P (16-core / 32 GB RAM), Linux
# Two-stage Bayesian tuning per replication:
#   - BO #1 tunes mean-model hyperparams on controls (for tau_hat)
#   - BO #2 tunes variance-model hyperparams on treated (for var1)
# SE uses BOTH parts:  se = sqrt(var0 + var1)
#
# File name: DDA_main_many_cluster.jl
# Recommended runs (choose one solver):
#   JULIA_WORKER_TIMEOUT=300 julia --project=. --bind-to 127.0.0.1 -p 12 DDA_wrapper.jl COSMO
#   JULIA_WORKER_TIMEOUT=300 julia --project=. --bind-to 127.0.0.1 -p 12 DDA_wrapper.jl Mosek
#   JULIA_WORKER_TIMEOUT=300 julia --project=. --bind-to 127.0.0.1 -p 12 DDA_wrapper.jl COPT
#
# Default (Mosek):
#   JULIA_WORKER_TIMEOUT=300 julia --project=. --bind-to 127.0.0.1 -p 12 DDA_wrapper_many_cluster.jl
# =============================================================================

# Load the required packages
using Distributed
using Statistics, LinearAlgebra, Distributions, Random
using BayesianOptimization, GaussianProcesses, ProximalOperators
using JuMP, DelimitedFiles, DataFrames, CSV
using Mosek, MosekTools, COPT, COSMO

# -----------------------------------------------------------------------------
# Worker initialization (executed only if Julia was started without -p)
# -----------------------------------------------------------------------------
if nprocs() == 1
    proj = Base.active_project()  # path to Project.toml
    addprocs(12; exeflags="--project=$(proj)", restrict=true)
end

@everywhere begin
    using Statistics, LinearAlgebra, Distributions, Random
    using BayesianOptimization, GaussianProcesses, ProximalOperators
    using JuMP
    using DelimitedFiles, DataFrames, CSV
    using Mosek, MosekTools, COPT, COSMO
end

# -----------------------------------------------------------------------------
# 1) WL1L0 fitter (on all workers)
# -----------------------------------------------------------------------------
@everywhere function wl1l0_fit_coeff(
    WX::Matrix{Float64},
    YX::Vector{Float64},
    α::Float64,
    λ::Float64,
    r::Float64;
    tol::Float64 = 5e-4,
    maxit::Int = 5000
)
    # Calculate covariances for initializations to SCPRSM
    covar = cov(WX, YX)

    # Initialization
    u = covar[:, 1] * 0.0001
    v = covar[:, 1] * 0.0001
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
    λ₁ = λ * α
    λ₂ = λ * (1.0 - α)

    # Proximal operator building blocks for the L1 block
    hγ = LeastSquares(WX, YX)  # Loss function for L1
    fγ = Translate(hγ, v)      # Translation function for L1
    gγ = NormL1(λ₁)            # L1 penalty

    # Proximal operator building blocks for the L0 block
    hδ = LeastSquares(WX, YX)  # Loss function for L0
    fδ = Translate(hδ, u)      # Translation function L0
    gδ = NormL0(λ₂)            # L0 regularization

    # Backtracking parameters and initial step sizes
    ℸ, γ, δ = 0.5, 0.9, 0.9

    # Loss function for line search
    loss(x) = 0.5 * norm(WX * x - YX)^2

    for it = 1:maxit
        # γ-block (L1 part): backtracking based line search
        gradγ = WX' * (WX * c - YX)
        while loss(u) > (loss(c) + gradγ' * (-c) + (1.0 / (2.0 * γ)) * norm(-c)^2)
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

        # δ block (L0 part): choose δ by backtracking, then apply prox updates
        gradδ = WX' * (WX * d - YX)
        while loss(v) > (loss(d) + gradδ' * (-d) + (1.0 / (2.0 * δ)) * norm(-d)^2)
            δ *= ℸ
        end

        # SCPRSM L0 updates
        prox!(d, fδ, v - l, δ)
        # First dual update for the L0 block
        l .+= r * (d - v)

        prox!(v, gδ, d + l, δ)

        # Stopping criterion
        if it % 5 == 2 && (norm((u + v) - uvcurr) / norm(((u + v) + uvcurr) / 2)) <= tol
            break
        end

        # Second dual update for the L0 block
        l2 .+= r * (d - v)
    end

    # Final coefficient estimate for the control outcome model
    return u + v
end

# -----------------------------------------------------------------------------
# 2a) Robust BO helper (on all workers)
#    Retries numerical failures; deterministic fallback if still fails.
# -----------------------------------------------------------------------------
@everywhere function robust_boptimize(
    bo_func::Function,
    low::Vector{Float64},
    high::Vector{Float64},
    base_seed::Int;
    maxtries::Int = 3,
    maxiter::Int = 30
)
    best_par = nothing
    success = false

    for attempt in 1:maxtries
        try
            # deterministic reseed per attempt
            Random.seed!(base_seed + 1_000_000 * attempt)

            opt = BOpt(
                bo_func,
                ElasticGPE(3, capacity=100),
                UpperConfidenceBound(),
                MAPGPOptimizer(),
                low,
                high,
                maxiterations=maxiter,
                sense=Min,
                verbosity=Silent
            )

            best_par = boptimize!(opt)[2]
            success = true
            break
        catch e
            # Common numeric failures in GP/Cholesky/logdet or LAPACK
            if e isa LinearAlgebra.SingularException ||
               e isa DomainError ||
               e isa PosDefException ||
               e isa ArgumentError
                continue
            else
                rethrow()
            end
        end
    end

    if !success
        best_par = (low .+ high) ./ 2
    end

    return best_par
end

# -----------------------------------------------------------------------------
# 2) One replication (on all workers)
#    Two-stage BO per replication and SE uses var0+var1 (control+treated).
# -----------------------------------------------------------------------------
@everywhere function run_single_rep(
    seed_i::Int,
    n::Int,
    p::Int,
    beta_main::Vector{Float64},
    tau_clust::Vector{Float64},
    clust_ptreat::Vector{Float64},
    solver_choice::AbstractString,
    low_m::Vector{Float64},
    high_m::Vector{Float64},
    low_v::Vector{Float64},
    high_v::Vector{Float64},
)
    Random.seed!(seed_i)

    # Data generation
    nclust = 20
    cluster_center = randn(nclust, p)
    cluster_ids = rand(1:nclust, n)

    W = cluster_center[cluster_ids, :] .+ randn(n, p)
    probs = clust_ptreat[cluster_ids]
    X = Int.(rand(n) .< probs)

    Y = (W * beta_main) .+ randn(n) .+ (tau_clust[cluster_ids] .* X)

    idx1 = findall(==(1), X)
    idx0 = findall(==(0), X)

    # need at least 2 treated obs for mean/var; also need some controls
    if length(idx1) < 2 || length(idx0) < 2
        return (err_sq = 0.0, covered = 0)
    end

    catt_curr = mean(tau_clust[cluster_ids[idx1]])

    # scale W
    scl = [std(W[:, j], corrected=false) for j in 1:p]
    scl[scl .== 0] .= 1.0
    W_scl = W ./ scl'

    WX = W_scl[idx0, :]
    YX = Y[idx0]
    W1 = W_scl[idx1, :]
    Y1 = Y[idx1]

    balance_target = vec(mean(W1, dims=1))  # p-vector

    # optimize chi on control group
    model =
        solver_choice == "Mosek" ? JuMP.Model(Mosek.Optimizer) :
        solver_choice == "COPT"  ? JuMP.Model(COPT.Optimizer)  :
        JuMP.Model(COSMO.Optimizer)

    JuMP.set_silent(model)

    n_ctrl = length(idx0)

    # Balancing weights χ with box constraints for numerical stability
    JuMP.@variable(model, 1e-4 <= chi[1:n_ctrl] <= 1 - 1e-4)

    # Slack variable capturing maximum imbalance
    JuMP.@variable(model, delta >= 0)

    # Objective: minimize imbalance + weight regularization
    JuMP.@objective(model, Min, 0.5 * delta^2 + 0.5 * sum(chi .^ 2))

    # Weight normalization
    JuMP.@constraint(model, sum(chi) == 1)

    # Balance constraints
    JuMP.@constraint(model, delta .+ WX' * chi .>=  balance_target)
    JuMP.@constraint(model, delta .- WX' * chi .>= -balance_target)

    # Solve optimization problem
    JuMP.optimize!(model)

    # Extract optimized balancing weights
    chi_curr = JuMP.value.(chi)

    # Check that the optimized balancing weights χ are nonnegative
    if any(chi_curr .< 0)
        error("chi invalid: contains negative values")
    end

    # -------------------------------------------------------------------------
    # BO #1 (Mean model): tune (α,λ,r) on controls to estimate tau_hat
    # -------------------------------------------------------------------------
    bo_main = par -> begin
        b0 = wl1l0_fit_coeff(WX, YX, par[1], par[2], par[3])
        mu_l1l0 = dot(balance_target, b0)
        residuals = YX .- WX * b0
        mu_hat = mu_l1l0 + sum(chi_curr .* residuals)
        abs((mean(Y1) - mu_hat) - catt_curr)
    end

    α_m, λ_m, r_m = robust_boptimize(bo_main, low_m, high_m, seed_i + 11)[1:3]

    # Final tau estimate using mean-model hyperparams
    b0_f = wl1l0_fit_coeff(WX, YX, α_m, λ_m, r_m)
    mu0_hat = dot(balance_target, b0_f) + sum(chi_curr .* (YX .- WX * b0_f))
    tau_hat = mean(Y1) - mu0_hat

    # var0: weighted control residual variance with df correction
    df0 = sum(b0_f .!= 0.0)
    var0 = sum((chi_curr .^ 2) .* ((YX .- WX * b0_f) .^ 2)) *
        (length(YX) / max(1.0, length(YX) - df0))

    # -------------------------------------------------------------------------
    # BO #2 (Variance model): tune (α,λ,r) on treated to estimate var1
    # -------------------------------------------------------------------------
    bo_var = par -> begin
        b1 = wl1l0_fit_coeff(W1, Y1, par[1], par[2], par[3])
        resid1 = Y1 .- W1 * b1
        df1 = sum(b1 .!= 0.0)
        mean(resid1 .^ 2) / max(1.0, length(Y1) - df1)
    end

    α_v, λ_v, r_v = robust_boptimize(bo_var, low_v, high_v, seed_i + 22)[1:3]

    # var1
    b1_f = wl1l0_fit_coeff(W1, Y1, α_v, λ_v, r_v)
    df1 = sum(b1_f .!= 0.0)
    var1 = (sum((Y1 .- W1 * b1_f) .^ 2) / max(1.0, length(Y1) - df1)) / length(Y1)

    se_hat = sqrt(var0 + var1)
    covered = (abs(tau_hat - catt_curr) / se_hat <= 1.96) ? 1 : 0

    return (err_sq = (tau_hat - catt_curr)^2, covered = covered)
end

# -----------------------------------------------------------------------------
# 3) Main (master): EXACT 10 × 100 = 1000
# -----------------------------------------------------------------------------
function main(
    n::Int, p::Int, C::Float64, eps::Float64, b_setup::Int,
    low_m1::Float64, low_m2::Float64, low_m3::Float64,
    high_m1::Float64, high_m2::Float64, high_m3::Float64,
    low_v1::Float64, low_v2::Float64, low_v3::Float64,
    high_v1::Float64, high_v2::Float64, high_v3::Float64,
    n_rep::Int, solver_choice::AbstractString;
    seed::Int = 12345
)
    nbatches = 10
    batch_size = 100
    @assert n_rep == nbatches * batch_size "Require n_rep = 1000 (10×100) for batching"

    # Outcome model setup
    # b_setup = 1: b_j ∝ 10 · 1{j ≤ 10}
    # b_setup = 2:    b_j ∝ 1 / j^2
    # b_setup = 3:    b_j ∝ 1 / j
    b_raw =
        b_setup == 1 ? vcat(fill(10.0, 10), zeros(p - 10)) :
        b_setup == 2 ? 1.0 ./ (1:p) .^ 2 :
        1.0 ./ (1:p)

    beta_base = C * b_raw / sqrt(sum(b_raw .^ 2))
    nclust = 20
    beta_main = sqrt(nclust) * beta_base

    clust_ptreat = repeat([eps, 1.0 - eps], inner=1, outer=div(nclust, 2))

    # FIXED tau_clust for this (n, p, eps, b_setup)
    # DGP: tau_clust = tau * rand(Exponential(1.0), nclust)
    # Here tau = 1 ⇒ scaling omitted
    Random.seed!(seed)
    tau_clust = rand(Exponential(1.0), nclust)

    println("\n" * "="^45)
    println("STARTING PARALLEL SIMULATION (n=$n, p=$p, Reps=$n_rep)")
    println("Solver: $solver_choice | Workers: $(nworkers())")
    println("="^45)

    low_m  = [low_m1, low_m2, low_m3]
    high_m = [high_m1, high_m2, high_m3]
    low_v  = [low_v1, low_v2, low_v3]
    high_v = [high_v1, high_v2, high_v3]

    wp = CachingPool(workers())

    all_results = Vector{NamedTuple{(:err_sq, :covered), Tuple{Float64, Int}}}(undef, 0)
    sizehint!(all_results, n_rep)

    for b in 1:nbatches
        lo = (b - 1) * batch_size + 1
        hi = b * batch_size
        global_ids = lo:hi

        println("Batch $b / $nbatches (reps $lo:$hi)")

        batch_results = pmap(wp, global_ids) do gid
            seed_i = seed + gid
            run_single_rep(seed_i, n, p, beta_main, tau_clust, clust_ptreat,
                           solver_choice, low_m, high_m, low_v, high_v)
        end

        append!(all_results, batch_results)

        CSV.write(
            "checkpoint_$(lowercase(String(solver_choice)))_n$(n)_p$(p)_b$(b_setup)_eps$(eps)_batch$(b).csv",
            DataFrame(err_sq = [r.err_sq for r in batch_results],
                      covered = [r.covered for r in batch_results])
        )
    end

    return DataFrame(
        RMSE = sqrt(mean(r.err_sq for r in all_results)),
        Coverage = mean(r.covered for r in all_results)
    )
end
