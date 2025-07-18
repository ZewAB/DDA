# =============================================================================
# Script: DDA for ATT Estimation for the second simulation setting
# Main components:
# - Weight computation using different solvers
# - WL1L0-SCPRSM via Proximal Operators
# - Data-adaptive learning rate (γ,δ)
# - Data-adaptive Bayesian Optimization (BO) for tuning (α, λ, r)
# =============================================================================
# Load the required packages (if they are not installed, please install them before loading)

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
using DataFrames, CSV
using COPT, COSMO
function main(n, p, arg1, arg2, arg3, arg4, arg5, arg6, n_rep, solver_choice)
    println("Number of observations = ", n, ", Number of predictors = ", p)
    println("Lower and upper bounds for α, λ, and r in BO,resp.: ", arg1, ", ", arg2, ", ", arg3, ", ", arg4, ", ", arg5, ", ", arg6)
    println("Simulation replications: ", n_rep)
    println("solver choice: ", solver_choice)
    # Define the miss especified model
    function miss_data()
        # Generate a random covariate matrix W (n x p)
        W = randn(n, p)
       # Compute tauW (normalized such that E[tauW | X = 1] = 1)
        tauW = log.(1 .+ exp.(-2 .- 2 .* W[:, 1])) ./ 0.915
         # Probability of treatment assignment based on tauW
        ptreat = 1 .- exp.(-tauW)
        # Sample treatment assignment X from Bernoulli distribution
        X = rand.(Bernoulli.(ptreat))
         # Mean of tauW among treated units
        tauW_mean = mean(tauW[X .== 1])
        # Outcome Y depends on tauW, treatment X, and sum of first 10 features
        Y = randn(n) .+ tauW .* (2 .* X .- 1) ./ 2 .+ sum(W[:, 1:10], dims=2)
        return (W=W, Y=Y, X=X, tau=tauW_mean)  
    end

# Function to scale non-binary features
function custom_scale(W, scale_W)
    if scale_W
        scl = [std(filter(!isnan, W[:, i]), corrected=false) for i in 1:size(W, 2)]
        is_binary = [all(w -> (w == 0 || w == 1) || isnan(w), ww) for ww in eachcol(W)]
        scl[is_binary] .= 1  # do not scale binary columns
        W_scl = W ./ scl'
    else
        W_scl = W
    end
    return W_scl
end

# Define optimize_chi with the chosen solver
function optimize_chi(M::Matrix{Float64}, balance_target::Vector{Float64}, z::Float64=0.5, solver::String=solver_choice)
    n_weights, _ = size(M)
    
    # Create the optimization model based on the chosen solver
    model = if solver == "Mosek"
        JuMP.Model(() -> Mosek.Optimizer())
    elseif solver == "COPT"
        JuMP.Model(() -> COPT.Optimizer())
    elseif solver == "COSMO"
        JuMP.Model(() -> COSMO.Optimizer())
    else
        error("Unsupported solver: $solver")
    end

    JuMP.set_silent(model)
    
    # Define variables
    chi = @variable(model, [1:n_weights], lower_bound=1e-4, upper_bound=1-1e-4)
    delta = @variable(model)
    
    # Define objective (quadratic)
    @objective(model, Min, z * delta^2 + (1 - z) * sum(chi[i]^2 for i in 1:n_weights))
    
    # Define constraints
    @constraint(model, sum(chi) == 1)
    @constraint(model, delta .+ M' * chi .>= balance_target)
    @constraint(model, delta .- M' * chi .>= -balance_target)
    
    # Solve optimization
    JuMP.optimize!(model)
    
    chi_sol = JuMP.value.(chi)
    if any(chi_sol .< 0)
        error("chi is not valid: contains negative values")
    end
    
    return chi_sol
end

data = miss_data()
W, Y, X, tau = data[1], data[2], data[3], data[4]

# Scale W if needed
scale_W = true
W_scl = custom_scale(W, scale_W)
    
# Compute balance target
balance_target = reshape(Statistics.mean(W_scl[X .== 1, :], dims=1), :)
WX = W_scl[X .== 0, :]
YX = Y[X .== 0];
chi = optimize_chi(WX,balance_target)
    
# The WL1L0-SCPRSM function for BO
SCPRSM_bo(par::Vector) = SCPRSM_bo(par[1], par[2], par[3])
function SCPRSM_bo(par1, par2, par3)
# Calculate covariances for inits to SCPRSM
  covar = cov(WX, YX)
 # Convergence tolerance of SCPRSM
  tol = 5e-4
# Maximum number of SCPRSM iterations
  maxit = 5000
        
  # Hyperparameters
  α = par1
  λ = par2
  r = par3

  # Initialization
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
  hγ = LeastSquares(WX, YX)    # Loss function for L1
  fγ = Translate(hγ, v)        # Translated loss
  gγ = NormL1(λ₁)              # Regularization function for L1

  gradδ = zero(d)
  hδ = LeastSquares(WX, YX)   # Loss function for L0
  fδ = Translate(hδ, u)       # Translated loss
  gδ = NormL0(λ₂)             # Regularization function for L0

   # Initial values for backtracking line search
  ℸ = 0.5            # shrinkage factor
  γ = 0.9            # learning rate for L1 
  δ = 0.9            # learning rate for L0 

  # Loss function for line search     
  loss(x) = 0.5 * norm(WX * x - YX)^2

  for it = 1:maxit
    # Line search for γ 
    gradγ = WX' * (WX * c - YX)
    while loss(u) > (loss(c) + gradγ' * (-c) + (1 / (2γ)) * norm(-c)^2)
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
    while loss(v) > (loss(d) + gradδ' * (-d) + (1 / (2δ)) * norm(-d)^2)
      δ *= ℸ
    end

    # SCPRSM update L0 
    prox!(d, fδ, v - l, δ)
    # First dual update L0 
    l .+= r * (d - v)
    prox!(v, gδ, d + l, δ)

    # Check convergence
    dualres = (u + v) - uvcurr
    reldualres = dualres / norm(((u + v) + uvcurr) / 2)
    if it % 5 == 2 && norm(reldualres) <= tol
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
  eta1 = mean(Y[X .== 1])
  tau_hat = eta1 .- mu_hat
  rmse = sqrt(mean((tau_hat .- tau).^2))  # RMSE

  return rmse
end

    # Optimize the hyperparameters of the GP using maximum a posteriori (MAP) estimates every 30 steps
    modeloptimizer = MAPGPOptimizer(every=30, noisebounds=[-1., 10.],
        kernbounds=[[-3., -3., -3, 0.], [6., 8., 8., 8.]],
        maxeval=40)

    model = ElasticGPE(3, mean=MeanConst(0.), kernel=SEArd([2., 3., 3.], 1.),
        logNoise=4., capacity=1000)

    optSCPRSM = BOpt(par -> SCPRSM_bo(par[1], par[2], par[3]), model, 
        UpperConfidenceBound(), modeloptimizer,
        [arg1, arg2, arg3], [arg4, arg5, arg6], repetitions=4, maxiterations=250,
        sense=Min,
    verbosity=Silent)

    reSCPRSMbo = boptimize!(optSCPRSM)
    
 # The WL1L0-SCPRSM function for optimized hyperparameters
function SCPRSM_bo1(α, λ, r, WX2, YX2, Y2, balance_target2, chi2, tau2)
  covar = cov(WX2, YX2)

  # Convergence parameters
  tol = 5e-4
  maxit = 5000

  # Initialization
  u = covar[:,1] * 0.0001
  v = covar[:,1] * 0.0001
  uvcurr = zero(WX2[1,:])
  c = zero(WX2[1,:])
  d = zero(WX2[1,:])
  m = zero(c)
  m2 = zero(c)
  l = zero(d)
  l2 = zero(d)

  # Regularization parameters
  λ₁ = λ * α
  λ₂ = λ * (1.0 - α)

  gradγ = zero(c)
  hγ = LeastSquares(WX2, YX2)
  fγ = Translate(hγ, v)
  gγ = NormL1(λ₁)

  gradδ = zero(d)
  hδ = LeastSquares(WX2, YX2)
  fδ = Translate(hδ, u)
  gδ = NormL0(λ₂)

  # Backtracking parameters
  ℸ = 0.5
  γ = 0.9
  δ = 0.9

  # Loss function for line search
  loss(x) = 0.5 * norm(WX2 * x - YX2)^2

  for it = 1:maxit
    # Line search for γ 
    gradγ = WX2' * (WX2 * c - YX2)
    while loss(u) > (loss(c) + gradγ' * (-c) + (1 / (2γ)) * norm(-c)^2)
      γ *= ℸ
    end

    uvcurr = u + v

    # SCPRSM update L1
    prox!(c, fγ, u - m, γ)
    m .+= r * (c - u)
    prox!(u, gγ, c + m, γ)
    m2 .+= r * (c - u)

    # Line search for δ (L0 block)
    gradδ = WX2' * (WX2 * d - YX2)
    while loss(v) > (loss(d) + gradδ' * (-d) + (1 / (2δ)) * norm(-d)^2)
      δ *= ℸ
    end

     # SCPRSM update L0 
    prox!(d, fδ, v - l, δ)
    # Second dual update L0
    l .+= r * (d - v)
    prox!(v, gδ, d + l, δ)

    # Stopping criterion
    dualres = (u + v) - uvcurr
    reldualres = dualres / norm(((u + v) + uvcurr)/2)
    if it % 5 == 2 && (norm(reldualres) <= tol)
      break
    end

    # Second dual update L0 
    l2 .+= r * (d - v)
  end

  # Estimate tau and return RMSE
  l1l0_fit = u + v
  mu_l1l0 = reshape(balance_target2, 1, length(balance_target2)) * l1l0_fit
  residuals = YX2 - WX2 * l1l0_fit
  mu_residual = sum(chi2 .* residuals)
  mu_hat = mu_l1l0 .+ mu_residual
  eta1 = mean(Y2[X .== 1])
  tau_hat = eta1 .- mu_hat
  rmse = sqrt(mean((tau_hat .- tau2).^2))
  return rmse
end
# Calculate average time per iteration
total_time = @elapsed begin
num_repeats = n_rep
rmses = Float64[]  # To store RMSEs from each iteration

for i in 1:num_repeats
    # Step 1: Generate data
     data2 = miss_data()
   W2, Y2, X2, tau2 = data2[1], data2[2], data2[3], data2[4]

    # Scale W if needed
    global scale_W = true
    W_scl2 = custom_scale(W2, scale_W)
    
    # Compute balance target
    balance_target2 = reshape(Statistics.mean(W_scl2[X2 .== 1, :], dims=1), :)
   WX2 = W_scl2[X2 .== 0, :]
   YX2 = Y2[X2 .== 0]

    # Step 4: Optimize chi
    chi2 = optimize_chi(WX2, balance_target2)

    # Step 5: Compute RMSE
    rmse_res = SCPRSM_bo1(reSCPRSMbo[2][1], reSCPRSMbo[2][2], reSCPRSMbo[2][3], WX2, YX2, Y2, balance_target2, chi2, tau2)
    push!(rmses, rmse_res)  # Store 
    # Print RMSE for the current iteration
   println("Iteration $i: RMSE = $rmse_res")        
end
end 
average_time = total_time / num_repeats

    # Print summary
println("Total time: $total_time seconds")
println("Average time per iteration: $average_time seconds")
println("Mean_RMSE: ", mean(rmses))
    
# Write summary results to a CSV file    
results_miss = DataFrame(
    Mean_RMSE = mean(rmses),
    Arg1 = arg1, 
    Arg2 = arg2, 
    Arg3 = arg3,
    Arg4 = arg4, 
    Arg5 = arg5, 
    Arg6 = arg6,
    alp = reSCPRSMbo[2][1],
    lambda = reSCPRSMbo[2][2],
    r = reSCPRSMbo[2][3],
    Solver = solver_choice,
    Total_Time = total_time,
    Average_Time = average_time,
    num_repeats  = n_rep
)
println(results_miss)
CSV.write("results_miss_$(n)_$(p)_$(arg1)_$(arg2)_$(arg3)_$(arg4)_$(arg5)_$(arg6)_$(n_rep)_$(solver_choice).csv", results_miss)
end
