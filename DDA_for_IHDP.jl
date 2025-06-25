# =============================================================================
# Script: DDA for ATT Estimation Using IHDP Data
# Main components:
# - Weight computation using different solvers
# - WL1L0-SCPRSM via Proximal Operators
#-  Data-adaptive learning rate (γ,δ)
# - Data-adaptive Bayesian Optimization for tuning (α, λ, r)
# =============================================================================
# Load the required packages (if they are not installed, please install them before loading)

# Required packages
using Statistics
using LinearAlgebra
using Distributions
using Random
using BayesianOptimization
using GaussianProcesses
using ProximalOperators
using JuMP
using Mosek
using MosekTools
using DelimitedFiles
using DataFrames
using CSV
using COPT
using COSMO
using Ipopt

# Custom covariate scaling
function custom_scale(W, scale_W)
    if scale_W
        scl = [std(filter(!isnan, W[:, i]), corrected=false) for i in 1:size(W, 2)]
        is_binary = [all(w -> (w == 0 || w == 1) || isnan(w), ww) for ww in eachcol(W)]
        scl[is_binary] .= 1  # Do not scale binary features
        W_scl = W ./ scl'
    else
        W_scl = W
    end
    return W_scl
end

# chi optimization
function optimize_chi(M::Matrix{Float64}, balance_target::Vector{Float64}, z::Float64=0.5, solver::String=solver_choice)
    n_weights, _ = size(M)

    model = if solver == "Mosek"
        JuMP.Model(() -> Mosek.Optimizer())
    elseif solver == "COPT"
        JuMP.Model(() -> COPT.Optimizer())
    elseif solver == "COSMO"
        JuMP.Model(() -> COSMO.Optimizer())
    elseif solver == "Ipopt"
        JuMP.Model(() -> Ipopt.Optimizer())
    else
        error("Unsupported solver: $solver")
    end

    JuMP.set_silent(model)

    @variable(model, chi[1:n_weights], lower_bound=1e-4, upper_bound=1 - 1e-4)
    @variable(model, delta)

    @objective(model, Min, z * delta^2 + (1 - z) * sum(chi[i]^2 for i in 1:n_weights))
    @constraint(model, sum(chi) == 1)
    @constraint(model, delta .+ M' * chi .>= balance_target)
    @constraint(model, delta .- M' * chi .>= -balance_target)

    JuMP.optimize!(model)

    chi_sol = JuMP.value.(chi)
    if any(chi_sol .< 0)
        error("chi contains invalid (non-positive) values")
    end

    return chi_sol
end

# The WL1L0-SCPRSM function that will use the optimized hyperparameters
function SCPRSM_bo1(α, λ, r, WX, YX, Y, X, balance_target, chi, tau)
# Calculate covariances for inits to SCPRSM
  covar = cov(WX, YX)
  # Convergence tolerance of SCPRSM
  tol = 5e-4
# Maximum number of SCPRSM iterations
  maxit = 5000

  # Initial values
  u = covar[:, 1] * 0.0001
  v = covar[:, 1] * 0.0001
  uvcurr = zero(WX[1, :])
  c = zero(WX[1, :])
  d = zero(WX[1, :])
  m = zero(c)
  m2 = zero(c)
  l = zero(d)
  l2 = zero(d)

  # Regularization weights
  λ₁ = λ * α
  λ₂ = λ * (1.0 - α)
    
  gradγ = zero(c)
  hγ = LeastSquares(WX, YX)          # Loss function for L1
  fγ = Translate(hγ, v)              # Translated loss
  gγ = NormL1(λ₁)                    # Regularization function for L1

  gradδ = zero(d)
  hδ = LeastSquares(WX, YX)          # Loss function for L0
  fδ = Translate(hδ, u)              # Translated loss
  gδ = NormL0(λ₂)                    # Regularization function for L0

  # Initial values for backtracking line search
  ℸ = 0.5                            # shrinkage factor
  γ = 0.9                            # learning rate for L1 
  δ = 0.9                            # learning rate for L0 

  # Loss function for line search
  loss(x) = 0.5 * norm(WX * x - YX)^2

  for it = 1:maxit
    # Line search for γ 
    gradγ = WX' * (WX * c - YX)
    while loss(u) > (loss(c) + gradγ' * (-c) + (1.0 / (2.0 * γ)) * norm(-c)^2)
      γ *= ℸ
    end
    uvcurr = u + v
    # SCPRSM update L1 
    prox!(c, fγ, u - m, γ)
    # First dual update L1 
    m .+= r * (c - u)
    prox!(u, gγ, c + m, γ)
    # Second dual update L1 
    m2 .+= r * (c - u)

    # Line search for δ 
    gradδ = WX' * (WX * d - YX)
    while loss(v) > (loss(d) + gradδ' * (-d) + (1.0 / (2.0 * δ)) * norm(-d)^2)
      δ *= ℸ
    end

    # SCPRSM update L0 
    prox!(d, fδ, v - l, δ)
    # First dual update L0 
    l .+= r * (d - v)
    prox!(v, gδ, d + l, δ)

    # Stopping criterion
    dualres = (u + v) - uvcurr
    reldualres = dualres / norm(((u + v) + uvcurr) / 2)
    if it % 5 == 2 && (norm(reldualres) <= tol)
      break
    end

     # Second dual update L0 
    l2 .+= r * (d - v)
  end

  # Estimate tau and return RMSE
  l1l0_fit = u + v
  mu_l1l0 = reshape(balance_target, 1, length(balance_target)) * l1l0_fit
  residuals = YX - WX * l1l0_fit
  mu_residual = sum(chi .* residuals)
  mu_hat = mu_l1l0 .+ mu_residual

  var_hat = sum(chi .^ 2 .* residuals .^ 2) *
            length(chi) / max(1, length(chi) - sum(l1l0_fit .!= 0))

  eta1 = mean(Y[X .== 1])
  tau_hat = eta1 .- mu_hat
  rmse = sqrt(mean((tau_hat .- tau).^2)) # RMSE

  return rmse
end

# Main function
function main(arg1::Float64, arg2::Float64, arg3::Float64, arg4::Float64, arg5::Float64, arg6::Float64, solver_choice::String)
    α_values = Float64[]
    λ_values = Float64[]
    r_values = Float64[]
    total_rmse = 0.0
    total_time = 0.0

    for i in 1:10
        println("\nProcessing dataset $i")
        data = readdlm("ihdp_npci_$i.csv", ',')
        X = data[:, 1]
        Y = data[:, 2]
        Mu0 = data[:, 4]
        Mu1 = data[:, 5]
        W = data[:, 6:end]
        tau = mean(Mu1[X .== 1] .- Mu0[X .== 1])

        scale_W = true
        W_scl = custom_scale(W, scale_W)
        balance_target = reshape(mean(W_scl[X .== 1, :], dims=1), :)
        WX = W_scl[X .== 0, :]
        YX = Y[X .== 0]
        chi = optimize_chi(WX, balance_target, 0.5, solver_choice)

# The WL1L0-SCPRSM function for BO
SCPRSM_bo(par::Vector) = SCPRSM_bo(par[1], par[2], par[3])

function SCPRSM_bo(α, λ, r)
  covar = cov(WX, YX)
  # Convergence tolerance and max iterations
  tol = 5e-4
  maxit = 5000

  # Initialize primal and dual variables
  u = covar[:, 1] * 0.0001
  v = covar[:, 1] * 0.0001
  uvcurr = zero(WX[1, :])
  c = zero(WX[1, :])
  d = zero(WX[1, :])
  m = zero(c)
  m2 = zero(c)
  l = zero(d)
  l2 = zero(d)

  # Regularization parameters
  λ₁ = λ * α
  λ₂ = λ * (1.0 - α)

  gradγ = zero(c)
  hγ = LeastSquares(WX, YX)
  fγ = Translate(hγ, v)
  gγ = NormL1(λ₁)

  gradδ = zero(d)
  hδ = LeastSquares(WX, YX)
  fδ = Translate(hδ, u)
  gδ = NormL0(λ₂)

 # Initial values for backtracking line search
  ℸ = 0.5 
  γ = 0.9
  δ = 0.9

  # Loss function for line search
  loss(x) = 0.5 * norm(WX * x - YX)^2

  for it = 1:maxit
    # Line search for γ 
    gradγ = WX' * (WX * c - YX)
    while loss(u) > (loss(c) + gradγ' * (-c) + (1.0 / (2.0 * γ)) * norm(-c)^2)
      γ *= ℸ
    end

    uvcurr = u + v

    # L1 updates
    prox!(c, fγ, u - m, γ)
    # First dual update L1      
    m .+= r * (c - u)
    prox!(u, gγ, c + m, γ)
    # Second dual update L1
    m2 .+= r * (c - u)

    # Line search for δ 
    gradδ = WX' * (WX * d - YX)
    while loss(v) > (loss(d) + gradδ' * (-d) + (1.0 / (2.0 * δ)) * norm(-d)^2)
      δ *= ℸ
    end

    # L0 updates
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

  # Post-processing for causal effect estimation
  l1l0_fit = u + v
  mu_l1l0 = reshape(balance_target, 1, length(balance_target)) * l1l0_fit
  residuals = YX - WX * l1l0_fit
  mu_residual = sum(chi .* residuals)
  mu_hat = mu_l1l0 .+ mu_residual
  eta1 = mean(Y[X .== 1])
  tau_hat = eta1 .- mu_hat

  rmse = sqrt(mean((tau_hat .- tau).^2))
  return rmse
end

        modeloptimizer = MAPGPOptimizer(every=30, noisebounds=[-1., 10.],
            kernbounds=[[-3., -3., -3, 0.], [6., 8., 8., 8.]], maxeval=40)

        model = ElasticGPE(3, mean=MeanConst(0.), kernel=SEArd([2., 3., 3.], 1.),
            logNoise=4., capacity=1000)

        optSCPRSM = BOpt(par -> SCPRSM_bo(par[1], par[2], par[3]),
            model, UpperConfidenceBound(), modeloptimizer,
            [arg1, arg2, arg3], [arg4, arg5, arg6], repetitions=4, maxiterations=250,
            sense=Min, verbosity=Silent)

        result = boptimize!(optSCPRSM)

        push!(α_values, result[2][1])
        push!(λ_values, result[2][2])
        push!(r_values, result[2][3])

        start_time = time()
        
        final_rmse = SCPRSM_bo1(result[2][1], result[2][2], result[2][3], WX, YX, Y, X, balance_target, chi, tau)
        elapsed_time = time() - start_time

        total_rmse += final_rmse
        total_time += elapsed_time

        println("Dataset $i completed with RMSE: $final_rmse, Time: $elapsed_time seconds")
    end

    results_IHDP = DataFrame(
        Mean_α = mean(α_values),
        Mean_λ  = mean(λ_values),
        Mean_r = mean(r_values),
        Mean_RMSE = total_rmse / 10,
        Mean_Time = total_time / 10,
        Arg1_Lower = arg1,
        Arg2_Lower = arg2,
        Arg3_Lower = arg3,
        Arg1_Upper = arg4,
        Arg2_Upper = arg5,
        Arg3_Upper = arg6,
        Solver = solver_choice
    )

    println("\nFinal Results:")
    println(results_IHDP)

    output_filename = "results_IHDP_$(arg1)_$(arg2)_$(arg3)_$(arg4)_$(arg5)_$(arg6)_$(solver_choice).csv"
    CSV.write(output_filename, results_IHDP)
    println("\nResults saved to $output_filename")
end
