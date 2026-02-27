# =============================================================================
# File: DDA_ihdp.jl
#
# Parallel DDA for ATT Estimation Using IHDP Data (10 datasets)
#
# - Runs 4 solvers: DDA-COSMO / DDA-Mosek / DDA-COPT / DDA-IPOPT
#        - COSMO (open-source)
#        - Mosek (commercial; free academic license available)
#        - COPT  (commercial; free academic license available)
#        - IPOPT (open-source)
# - For each solver: processes datasets i=1..10 in PARALLEL via pmap
# - End-to-end timing per dataset:
#   Scaling -> Weights (QP solver) -> BO Tuning -> Final Inference
#
# OUTPUTS:
#   1) ihdp_dda_all_results.csv
#   2) ihdp_dda_summary.csv
#   3) ihdp_table.tex
#
# Run:
#   JULIA_WORKER_TIMEOUT=300 julia --project=. --bind-to 127.0.0.1 -p 12 DDA_ihdp.jl
# =============================================================================
#
# Load Distributed first to enable parallel workers
using Distributed

# -----------------------------------------------------------------------------
# 1. Load dependencies on ALL workers FIRST
# (This ensures modules are fully loaded before macros are parsed)
# -----------------------------------------------------------------------------
@everywhere begin
    using Statistics
    using LinearAlgebra
    using Random
    using BayesianOptimization
    using GaussianProcesses
    using ProximalOperators
    import JuMP
    using DataFrames
    using CSV
    using DelimitedFiles

    # solvers
    using COSMO
    using MosekTools
    using COPT
    using Ipopt
end

# -----------------------------------------------------------------------------
# 2. Define ALL functions on ALL workers AFTER packages are loaded
# -----------------------------------------------------------------------------
@everywhere begin
    # -----------------------------
    # Scaling: do not scale binary columns
    # -----------------------------
    function custom_scale(W::Matrix{Float64}, scale_W::Bool)
        if !scale_W
            return W
        end
        scl = [std(W[:, j], corrected=false) for j in 1:size(W, 2)]
        scl[scl .== 0.0] .= 1.0
        is_binary = [all(x -> (x == 0.0 || x == 1.0), W[:, j]) for j in 1:size(W, 2)]
        scl[is_binary] .= 1.0
        return W ./ scl'
    end
    # -----------------------------------------------------------------------------
    # Balancing weights χ on controls with ∞-norm 
    # -----------------------------------------------------------------------------
    # Inputs:
    #   M              : generic placeholder control covariate matrix (n0 × p) after scaling
    #   balance_target : treated covariate mean vector (p × 1) after scaling
    #   ζ              : trade-off weight in the objective (balance vs stability)
    #   solver_choice  : solver identifier string (DDA-COSMO / DDA-Mosek / DDA-COPT / DDA-IPOPT)
    #
    # Optimization problem:
    #
    #   minimize_{χ, δ ≥ 0}     ζ · δ^2 + (1 − ζ) · ||χ||_2^2
    #
    #   subject to
    #       Σ_i χ_i = 1
    #       1e-4 ≤ χ_i ≤ 1 − 1e-4        (numerical stability / prevents extreme weights)
    #       || M'χ − balance_target ||_∞ ≤ δ
    #
    #   • δ measures the maximum absolute covariate imbalance after weighting.
    #
    # Implementation detail (∞-norm):
    #   The constraint ||M'χ − balance_target||_∞ ≤ δ is equivalent to:
    #
    #       M'χ − balance_target ≤ δ·1
    #      −M'χ + balance_target ≤ δ·1
    #
    #   which is implemented as two vector inequalities.
    # -----------------------------------------------------------------------------
    # -----------------------------
    # chi optimization 
    # -----------------------------
    function optimize_chi(M::Matrix{Float64}, balance_target::Vector{Float64},
                          z::Float64, solver_choice::String)

        n_weights = size(M, 1)

        model = if solver_choice == "DDA-Mosek"
            JuMP.Model(() -> MosekTools.Optimizer())
        elseif solver_choice == "DDA-COPT"
            JuMP.Model(() -> COPT.Optimizer())
        elseif solver_choice == "DDA-COSMO"
            JuMP.Model(() -> COSMO.Optimizer())
        elseif solver_choice == "DDA-IPOPT"
            JuMP.Model(() -> Ipopt.Optimizer())
        else
            error("Unsupported solver_choice = $solver_choice")
        end

        JuMP.set_silent(model)

         # Balancing weights χ with box constraints for numerical stability
        JuMP.@variable(model, chi[1:n_weights], lower_bound=1e-4, upper_bound=1 - 1e-4)
        JuMP.@variable(model, delta)

        JuMP.@objective(model, Min, z * delta^2 + (1 - z) * sum(chi[i]^2 for i in 1:n_weights))
        JuMP.@constraint(model, sum(chi) == 1)
        JuMP.@constraint(model, delta .+ M' * chi .>= balance_target)
        JuMP.@constraint(model, delta .- M' * chi .>= -balance_target)

        JuMP.optimize!(model)

        chi_sol = JuMP.value.(chi)
        # Validate balancing weights χ.
        # Theoretically χ ≥ 0 is sufficient, but we enforce χ ≥ 1e-4
        # to avoid degenerate or numerically unstable solutions.
        if any(.!isfinite.(chi_sol)) || any(chi_sol .<= 0)
            error("Invalid chi returned (solver=$solver_choice)")
        end
        return chi_sol
    end

    # =============================================================================
    # WL1L0-SCPRSM core: returns tau_hat (scalar)
    #
    # Given:
    #   • control covariates WX and control outcomes YX,
    #   • balancing target balance_target (treated-group covariate mean),
    #   • balancing weights χ on controls,
    #
    # This function fits a WL1L0-regularized control outcome model via SCPRSM and
    # then constructs the hybrid ATT estimator:
    #
    #   μ̂0 = balance_target' b̂0 + Σ_{i : X_i = 0} χ_i (Y_i − W_i' b̂0)
    #   τ̂  = mean(Y | X=1) − μ̂0
    # =============================================================================
    function dda_tauhat(α::Float64, λ::Float64, r::Float64,
                        WX::Matrix{Float64}, YX::Vector{Float64},
                        W_scl::Matrix{Float64}, Y::Vector{Float64}, X::Vector{Int},
                        balance_target::Vector{Float64}, chi::Vector{Float64};
                        tol::Float64=5e-4, maxit::Int=5000)

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
        λ₁ = λ * α           # L1 strength
        λ₂ = λ * (1.0 - α)   # L0 strength

        # Proximal operator building blocks
        hγ = LeastSquares(WX, YX)          # Loss function for L1 block
        fγ = Translate(hγ, v)              # Translation map (L1 block)
        gγ = NormL1(λ₁)                    # L1 proximal

        hδ = LeastSquares(WX, YX)          # Loss function for L0 block
        fδ = Translate(hδ, u)              # Translation map (L0 block)
        gδ = NormL0(λ₂)                    # L0 

        # Initial values for backtracking line search
        ℸ = 0.5       # shrinkage factor for step-size backtracking
        γ = 0.9       # step size for γ-block (for L1)
        δ = 0.9       # step size for δ-block ( for L0)

        # Loss function for line search
        loss(x) = 0.5 * norm(WX * x - YX)^2

        for it = 1:maxit
            # -------------------------
            # γ block (L1 side)
            # -------------------------
            gradγ = WX' * (WX * c - YX)
            while loss(u) > (loss(c) + gradγ' * (-c) + (1.0 / (2.0 * γ)) * norm(-c)^2)
                γ *= ℸ
            end

            uvcurr = u + v

            # SCPRSM L1 updates 
            prox!(c, fγ, u - m, γ)
            # First dual update L1
            m .+= r * (c - u)
            prox!(u, gγ, c + m, γ)
            # Second dual update L1
            m2 .+= r * (c - u)

            # -------------------------
            # δ block (L0 side)
            # -------------------------
            gradδ = WX' * (WX * d - YX)
            while loss(v) > (loss(d) + gradδ' * (-d) + (1.0 / (2.0 * δ)) * norm(-d)^2)
                δ *= ℸ
            end

            # SCPRSM L0 updates
            prox!(d, fδ, v - l, δ)
            # First dual update L0
            l .+= r * (d - v)
            prox!(v, gδ, d + l, δ)

            # -------------------------
            # Stopping criterion 
            # -------------------------
            dualres = (u + v) - uvcurr
            denom = norm(((u + v) + uvcurr) / 2)
            reldualres = denom == 0 ? dualres : dualres / denom
            if it % 5 == 2 && norm(reldualres) <= tol
                break
            end
            # Second dual update L0
            l2 .+= r * (d - v)
        end

        # Final coefficient estimate
        bfit = u + v

        # μ0 estimate 
        mu_l1l0 = dot(balance_target, bfit)
        residuals = YX - WX * bfit
        mu_residual = sum(chi .* residuals)
        mu_hat = mu_l1l0 + mu_residual

        # Treated-group mean outcome and ATT estimate
        eta1 = mean(Y[X .== 1])
        tau_hat = eta1 - mu_hat
        return tau_hat
    end

    # -------------------------------------------------------------------------
    # One dataset run (end-to-end): scaling → weights → BO tuning → τ̂ and error
    # -------------------------------------------------------------------------
    function run_one_dataset_IHDP(i::Int, solver_choice::String,
                                  aL::Float64, lL::Float64, rL::Float64,
                                  aU::Float64, lU::Float64, rU::Float64;
                                  seed::Int=12345)

        Random.seed!(seed)

        # Read the IHDP dataset file: treatment, outcomes, potential outcomes, covariates
        data = readdlm("ihdp_npci_$(i).csv", ',')
        X   = Int.(data[:, 1])
        Y   = Vector{Float64}(data[:, 2])
        Mu0 = Vector{Float64}(data[:, 4])
        Mu1 = Vector{Float64}(data[:, 5])
        W   = Matrix{Float64}(data[:, 6:end])

        # Ground-truth ATT from potential outcomes 
        tau_true = mean(Mu1[X .== 1] .- Mu0[X .== 1])

        # Start end-to-end timer (includes scaling, χ, BO tuning, and final τ̂)
        t0 = time()

        # Scale covariates
        W_scl = custom_scale(W, true)

        # Build treated-group covariate mean and control sample (WX, YX)
        balance_target = vec(mean(W_scl[X .== 1, :], dims=1))
        WX = W_scl[X .== 0, :]
        YX = Y[X .== 0]

        # Compute balancing weights χ on controls (ζ fixed to 0.5 here)
        chi = optimize_chi(WX, balance_target, 0.5, solver_choice)

        # BO objective: absolute ATT estimation error relative to the
        # known ground-truth ATT 
        function bo_obj(alp, lam, rr)
            tau_hat0 = dda_tauhat(alp, lam, rr, WX, YX, W_scl, Y, X, balance_target, chi)
            return abs(tau_hat0 - tau_true)
        end

        # GP hyperparameter optimizer used inside BayesianOptimization.jl
        modeloptimizer = MAPGPOptimizer(
            every=30, noisebounds=[-1., 10.],
            kernbounds=[[-3., -3., -3, 0.], [6., 8., 8., 8.]],
            maxeval=40
        )

        # Gaussian Process surrogate model for BO over (α, λ, r)
        model = ElasticGPE(
            3,
            mean=MeanConst(0.),
            kernel=SEArd([2., 3., 3.], 1.),
            logNoise=4.,
            capacity=1000
        )

        # Bayesian optimization configuration (UCB acquisition; 250 iterations)
        optSCPRSM = BOpt(
            par -> bo_obj(par[1], par[2], par[3]),
            model, UpperConfidenceBound(), modeloptimizer,
            [aL, lL, rL], [aU, lU, rU],
            repetitions=4, maxiterations=250,
            sense=Min, verbosity=Silent
        )

        # Run BO and extract tuned hyperparameters
        result = boptimize!(optSCPRSM)
        α_opt = result[2][1]
        λ_opt = result[2][2]
        r_opt = result[2][3]

        # Final ATT estimate with tuned hyperparameters
        tau_hat = dda_tauhat(α_opt, λ_opt, r_opt, WX, YX, W_scl, Y, X, balance_target, chi)

        # End timing
        t1 = time()
        elapsed = t1 - t0

        # Squared error for RMSE aggregation across datasets
        sq_err = (tau_hat - tau_true)^2

        return (
            dataset=i,
            seed=seed,
            solver=solver_choice,
            aL=aL, lL=lL, rL=rL, aU=aU, lU=lU, rU=rU,
            alp=α_opt, lambda=λ_opt, r=r_opt,
            tau_true=tau_true,
            tau_hat=tau_hat,
            sq_err=sq_err,
            total_time=elapsed
        )
    end
end

# -----------------------------------------------------------------------------
# Output utilities 
# -----------------------------------------------------------------------------
using DataFrames, CSV, Printf, Statistics

fmt3(x::Real) = @sprintf("%.3f", x)
fmtr(x::Real) = abs(x) < 0.01 ? @sprintf("%.4g", x) : fmt3(x)

#Parameter settings for the IHDP dataset
# Parameter bounds for BO over (α, λ, r) by solver
function bounds_map_IHDP()
    b = Dict{String, NTuple{6,Float64}}()
    b["DDA-COSMO"] = (0.001, 1000.0, 0.001, 0.99, 10000.0, 0.99)
    b["DDA-Mosek"] = (0.001, 1000.0, 0.001, 0.99, 10000.0, 0.99)
    b["DDA-COPT"]  = (0.001, 2000.0, 0.001, 0.99, 30000.0, 0.99)
    b["DDA-IPOPT"] = (0.001, 1000.0, 0.001, 0.99, 10000.0, 0.99)
    return b
end

# Run all solvers; within each solver, run datasets in parallel via pmap
function run_IHDP_all(; seed_base::Int=12345, datasets=collect(1:10))
    solvers = ["DDA-COSMO", "DDA-Mosek", "DDA-COPT", "DDA-IPOPT"]
    bnds = bounds_map_IHDP()

    all_rows = DataFrame(
        solver=String[],
        dataset=Int[],
        seed=Int[],
        aL=Float64[], lL=Float64[], rL=Float64[], aU=Float64[], lU=Float64[], rU=Float64[],
        alp=Float64[], lambda=Float64[], r=Float64[],
        tau_true=Float64[], tau_hat=Float64[],
        sq_err=Float64[],
        total_time=Float64[]
    )

    for (sidx, solver) in enumerate(solvers)
        @assert haskey(bnds, solver) "Missing IHDP bounds for solver=$solver"
        (aL,lL,rL,aU,lU,rU) = bnds[solver]

        println("\n" * "="^90)
        println("IHDP | Solver=$solver | datasets=1..10 | BO maxiter=250, reps=4")
        println("Bounds: α∈[$aL,$aU], λ∈[$lL,$lU], r∈[$rL,$rU]") # Lowe (L) and upper (U) bounds for parameters
        println("="^90)

        # Unique seed per (solver, dataset) to avoid correlated randomness across runs
        seeds = [seed_base + 10_000*sidx + i for i in datasets]

        # Parallel map across datasets for this solver
        results = pmap(1:length(datasets)) do k
            i = datasets[k]
            seed = seeds[k]
            run_one_dataset_IHDP(i, solver, aL, lL, rL, aU, lU, rU; seed=seed)
        end

        # Append results into the master DataFrame
        for row in results
            push!(all_rows, (
                solver=row.solver,
                dataset=row.dataset,
                seed=row.seed,
                aL=row.aL, lL=row.lL, rL=row.rL, aU=row.aU, lU=row.lU, rU=row.rU,
                alp=row.alp, lambda=row.lambda, r=row.r,
                tau_true=row.tau_true, tau_hat=row.tau_hat,
                sq_err=row.sq_err,
                total_time=row.total_time
            ))
        end

        # Checkpoint after each solver completes (fault tolerance)
        CSV.write("ihdp_dda_all_results_checkpoint.csv", all_rows)
    end

    # Final write-out of all dataset-level results
    CSV.write("ihdp_dda_all_results.csv", all_rows)
    return all_rows
end

# Summarize results by solver and export CSV + LaTeX table
function summarize_IHDP(rows::DataFrame;
                        out_csv::AbstractString="ihdp_dda_summary.csv",
                        out_tex::AbstractString="ihdp_table.tex")

    g = groupby(rows, :solver)

    summ = combine(g) do sdf
        (
            Mean_α = mean(sdf.alp),
            Mean_λ = mean(sdf.lambda),
            Mean_r = mean(sdf.r),
            RMSE = sqrt(mean(sdf.sq_err)),
            Mean_Total_Time = mean(sdf.total_time)
        )
    end

    # Apply Custom Sorting to match the target Table exactly
    desired_order = ["DDA-COSMO", "DDA-Mosek", "DDA-COPT", "DDA-IPOPT"]
    sort!(summ, :solver, by = x -> findfirst(==(x), desired_order))

    CSV.write(out_csv, summ)

    # Find the minimum values to bold in LaTeX
    min_rmse = minimum(summ.RMSE)
    min_time = minimum(summ.Mean_Total_Time)

    io = IOBuffer()
    println(io, "\\begin{table}[ht]")
    println(io, "\\centering")
    println(io, "\\caption{RMSE, mean total runtime (s), and optimal hyperparameter values (\$\\alpha\$, \$\\lambda\$, \$r\$), averaged across 10 semi-synthetic IHDP datasets for each method. Total runtime is measured end-to-end per dataset, including BO tuning. Bold values indicate the best-performing results.}")
    println(io, "\\begin{tabular}{|l|c|c|c|c|c|}")
    println(io, "\\hline")
    println(io, "Method & RMSE & Mean total time & Mean \$\\alpha\$ & Mean \$\\lambda\$ & Mean \$r\$ \\\\")
    println(io, "\\hline")

    for row in eachrow(summ)
        # Format strings, applying \textbf{} if they match the minimum
        rmse_str = fmt3(row.RMSE)
        if row.RMSE == min_rmse
            rmse_str = "\\textbf{" * rmse_str * "}"
        end

        time_str = fmt3(row.Mean_Total_Time)
        if row.Mean_Total_Time == min_time
            time_str = "\\textbf{" * time_str * "}"
        end

        println(io,
            row.solver, " & ",
            rmse_str, " & ",
            time_str, " & ",
            fmt3(row.Mean_α), " & ",
            fmt3(row.Mean_λ), " & ",
            fmtr(row.Mean_r), " \\\\"
        )
    end

    println(io, "\\hline")
    println(io, "\\end{tabular}")
    println(io, "\\end{table}")

    open(out_tex, "w") do f
        write(f, String(take!(io)))
    end

    return summ
end

# -----------------------------------------------------------------------------
# EXEC
# -----------------------------------------------------------------------------
seed_base = 12345
rows = run_IHDP_all(; seed_base=seed_base, datasets=collect(1:10))
summ = summarize_IHDP(rows)

println("\n--- IHDP ALL RUNS COMPLETE ---")
println("CSV saved as: ihdp_dda_all_results.csv")
println("Summary CSV saved as: ihdp_dda_summary.csv")
println("LaTeX saved as: ihdp_table.tex")
