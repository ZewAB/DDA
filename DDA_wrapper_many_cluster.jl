# =============================================================================
# DDA WRAPPER FOR MANY-CLUSTER INFERENCE — PARALLEL EXECUTION & LATEX REPORTING
# Experiments executed on AMD EPYC 7302P (16-core / 32 GB RAM), Linux
#
# File name: DDA_wrapper_many_cluster.jl
#
# Recommended runs (choose one solver):
#   JULIA_WORKER_TIMEOUT=300 julia --project=. --bind-to 127.0.0.1 -p 12 DDA_wrapper_many_cluster.jl COSMO
#   JULIA_WORKER_TIMEOUT=300 julia --project=. --bind-to 127.0.0.1 -p 12 DDA_wrapper_many_cluster.jl Mosek
#   JULIA_WORKER_TIMEOUT=300 julia --project=. --bind-to 127.0.0.1 -p 12 DDA_wrapper_many_cluster.jl COPT
#
# Default (Mosek):
#   JULIA_WORKER_TIMEOUT=300 julia --project=. --bind-to 127.0.0.1 -p 12 DDA_wrapper_many_cluster.jl
# =============================================================================

# Load main many cluster implementation
include("DDA_main_many_cluster.jl")

using DataFrames, CSV, Printf

# ------------------------------------------------------------
# Helpers (Formatting and Labeling)
# ------------------------------------------------------------
fmt2(x) = @sprintf("%.2f", x)
cell(rmse, cov) = "$(fmt2(rmse)) ($(fmt2(cov)))"

# ------------------------------------------------------------
# Run All Configurations
# ------------------------------------------------------------
function run_all_solver(solver_choice::AbstractString; seed=12345)
    rows = DataFrame(n=Int[], p=Int[], eps=Float64[], b_setup=Int[],
                     solver=String[], RMSE=Float64[], Coverage=Float64[])
    
    #Parameter settings for many-cluster inference
    # Configuration Array: 12 Scenarios (n, p, C, varrho (eps), b_setup, BO_Bounds..., n_rep, solver)
    runs = [
        (400, 800, 3.0, 0.1, 1, 0.001, 0.1, 0.001, 0.99, 1000.0, 0.99, 0.001, 0.001, 0.0001, 0.99, 10.0, 0.99, 1000, solver_choice),
        (400, 800, 3.0, 0.25, 1, 0.001, 0.1, 0.001, 0.99, 1000.0, 0.99, 0.001, 0.001, 0.0001, 0.99, 10.0, 0.99, 1000, solver_choice),
        (400, 800, 3.0, 0.25, 2, 0.001, 0.001, 0.001, 0.99, 500.0, 0.99, 0.001, 0.001, 0.0001, 0.99, 100.0, 0.99, 1000, solver_choice),
        (400, 800, 3.0, 0.1, 2, 0.001, 0.001, 0.001, 0.99, 500.0, 0.99, 0.001, 0.001, 0.0001, 0.99, 100.0, 0.99, 1000, solver_choice),
        (400, 800, 3.0, 0.25, 3, 0.001, 0.001, 0.1, 0.99, 500.0, 0.99, 0.001, 0.001, 0.0001, 0.99, 100.0, 0.99, 1000, solver_choice),
        (400, 800, 3.0, 0.1, 3, 0.1, 100.0, 0.1, 0.99, 1000.0, 0.99, 0.1, 0.01, 0.1, 0.99, 1000.0, 0.99, 1000, solver_choice),

        (400, 1600, 3.0, 0.1, 1, 0.001, 100.0, 0.0001, 0.99, 500.0, 0.99, 0.001, 0.001, 0.0001, 0.99, 100.0, 0.99, 1000, solver_choice),
        (400, 1600, 3.0, 0.25, 1, 0.001, 0.1, 0.001, 0.99, 1000.0, 0.99, 0.001, 0.001, 0.0001, 0.99, 10.0, 0.99, 1000, solver_choice),
        (400, 1600, 3.0, 0.1, 2, 0.001, 0.001, 0.001, 0.99, 500.0, 0.99, 0.001, 0.001, 0.0001, 0.99, 100.0, 0.99, 1000, solver_choice),
        (400, 1600, 3.0, 0.25, 2, 0.0001, 0.001, 0.0001, 0.99, 100.0, 0.99, 0.001, 0.001, 0.0001, 0.99, 100.0, 0.99, 1000, solver_choice),
        (400, 1600, 3.0, 0.1, 3, 0.01, 0.001, 0.001, 0.99, 100.0, 0.99, 0.001, 0.001, 0.0001, 0.99, 100.0, 0.99, 1000, solver_choice),
        (400, 1600, 3.0, 0.25, 3, 0.1, 100.0, 0.01, 0.99, 1000.0, 0.99, 0.001, 0.001, 0.0001, 0.99, 500.0, 0.99, 1000, solver_choice)
    ]

    for args in runs
        # Use invokelatest to safely call main() after include
        out = Base.invokelatest(main, args...; seed=seed)

        push!(rows, (n=args[1], p=args[2], eps=args[4], b_setup=args[5],
                     solver=String(solver_choice),
                     RMSE=out.RMSE[1], Coverage=out.Coverage[1]))
    end

    CSV.write("$(lowercase(solver_choice))_results.csv", rows)
    return rows
end

# ------------------------------------------------------------
# Build LaTeX Table
# ------------------------------------------------------------
function build_latex_table(rows::DataFrame; method_name="DDA", outfile="table.tex")
    solver_txt = unique(rows.solver)[1]
    np_pairs = sort(unique(rows[:, [:n, :p]]), [:n, :p])

    function get_cell(n, p, b, eps)
        sub = rows[(rows.n .== n) .& (rows.p .== p) .& (rows.b_setup .== b) .& (rows.eps .== eps), :]
        return isempty(sub) ? "--" : cell(sub.RMSE[1], sub.Coverage[1])
    end

    io = IOBuffer()

    
    println(io, raw"\begin{table}[ht]")
    println(io, raw"\centering")
    println(io, raw"{\tiny")

    println(io, "\\caption{RMSE (coverage) averaged over 1000 simulation replications for $method_name-$solver_txt, with a target coverage of 0.95.}")

    println(io, raw"\begin{tabular}{|ccc|cc|cc|cc|}")
    println(io, raw"\hline")
    println(io, raw"$n$ & $p$ & Method & \multicolumn{2}{c|}{$b_j \propto \mathbf{1}(j \le 10)$} & \multicolumn{2}{c|}{$b_j \propto 1/j^2$} & \multicolumn{2}{c|}{$b_j \propto 1/j$} \\\\")
    
    println(io, raw" & & & $\varrho=0.25$ & $\varrho=0.1$ & $\varrho=0.25$ & $\varrho=0.1$ & $\varrho=0.25$ & $\varrho=0.1$ \\\\ \hline")

    for i in 1:nrow(np_pairs)
        n, p = np_pairs.n[i], np_pairs.p[i]
        c1_25, c1_1 = get_cell(n, p, 1, 0.25), get_cell(n, p, 1, 0.1)
        c2_25, c2_1 = get_cell(n, p, 2, 0.25), get_cell(n, p, 2, 0.1)
        c3_25, c3_1 = get_cell(n, p, 3, 0.25), get_cell(n, p, 3, 0.1)

        println(io, "$n & $p & $method_name-$solver_txt & $c1_25 & $c1_1 & $c2_25 & $c2_1 & $c3_25 & $c3_1 \\\\ \\hline")
    end

    # FIX: close tabular, close tiny group, then close table
    println(io, raw"\end{tabular}")
    println(io, raw"}")
    println(io, raw"\end{table}")

    tex_str = String(take!(io))
    open(outfile, "w") do f
        write(f, tex_str)
    end
    return tex_str
end

# ------------------------------------------------------------
# Execution
# ------------------------------------------------------------
solver_input = length(ARGS) >= 1 ? ARGS[1] : "Mosek"

results_df = run_all_solver(solver_input)

final_tex = build_latex_table(results_df; outfile="$(lowercase(solver_input))_table.tex")

println("\n--- SIMULATION COMPLETE ---")
println("CSV saved as: $(lowercase(solver_input))_results.csv")
println("LaTeX saved as: $(lowercase(solver_input))_table.tex")
