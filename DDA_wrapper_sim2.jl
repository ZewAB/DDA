# =============================================================================
# File: DDA_wrapper_sim2.jl
# Experiments executed on AMD EPYC 7302P (16-core / 32 GB RAM), Linux
#
# SIM2 wrapper (misspecified):
# Runs DDA-COSMO / DDA-Mosek / DDA-COPT over:
#   n ∈ {400, 1000}, p ∈ {100, 200, 400, 800, 1600}
#
# Uses DDA_main_sim2.jl:
#   - BO once per config (bo_maxiter=250)
#   - seed_tune = 12345
#   - evaluation seed_i = seed + gid, wrapper passes seed=12345
#
# Saves:
#   - sim2_dda_all_results.csv
#   - sim2_dda_all_results_checkpoint.csv
#   - sim2_dda_cosmo_results.csv / sim2_dda_mosek_results.csv / sim2_dda_copt_results.csv
#   - sim2_table_n400_part1.tex   (p=100,200,400)
#   - sim2_table_n400_part2.tex   (p=800,1600)
#   - sim2_table_n1000_part1.tex  (p=100,200,400)
#   - sim2_table_n1000_part2.tex  (p=800,1600)
#
# Recommended run:
#   JULIA_WORKER_TIMEOUT=300 julia --project=. --bind-to 127.0.0.1 -p 12 DDA_wrapper_sim2.jl
# =============================================================================

# Load main SIM2 implementation
include("DDA_main_sim2.jl")

using DataFrames, CSV, Printf

fmt3(x::Real) = @sprintf("%.3f", x)
bold(s::AbstractString) = "\\textbf{$s}"
ismin(x, xmin; atol=1e-12) = isfinite(x) && abs(x - xmin) <= atol
cell3(x::Real, best::Real) = ismin(x, best) ? bold(fmt3(x)) : fmt3(x)

# -----------------------------------------------------------------------------
# SIM2 lower (L) and upper (U) bounds map
# bounds[method][(n,p)] = (αL, λL, rL, αU, λU, rU)
# -----------------------------------------------------------------------------
function bounds_map_sim2()
    b = Dict{String, Dict{Tuple{Int,Int}, NTuple{6,Float64}}}()

    # Parameter settings for the second experimental setup

    b["DDA-COSMO"] = Dict(
        # n=400
        (400, 100)  => (0.001, 0.01,   0.0001, 0.99, 100.0,  0.99),
        (400, 200)  => (0.001, 0.01,   0.0001, 0.99, 100.0,  0.99),
        (400, 400)  => (0.8,   50.0,   0.005,  0.99, 210.0,  0.08),   
        (400, 800)  => (0.92,  120.0,  0.005,  0.99, 320.0,  0.02),   
        (400, 1600) => (0.975, 190.0,  0.009,  0.99, 240.0,  0.013), 

        # n=1000
        (1000, 100)  => (0.05,   200.0, 0.05,   0.99, 2000.0, 0.99),
        (1000, 200)  => (0.0001, 500.0, 0.0001, 0.99, 1500.0, 0.99),
        (1000, 400)  => (0.0001, 0.01,  0.0001, 0.99, 100.0,  0.99),
        (1000, 800)  => (0.70,   200.0, 0.002,  0.99, 2500.0, 0.04),  
        (1000, 1600) => (0.975,  190.0, 0.009,  0.99, 240.0,  0.013)
    )

    b["DDA-Mosek"] = Dict(
        # n=400
        (400, 100)  => (0.01,  0.01,   0.001,  0.99, 100.0,  0.99),
        (400, 200)  => (0.001, 0.01,   0.0001, 0.99, 100.0,  0.99),
        (400, 400)  => (0.97,  190.0,  0.009,  0.99, 210.0,  0.015), 
        (400, 800)  => (0.97,  170.0,  0.008,  0.99, 250.0,  0.013),  
        (400, 1600) => (0.975, 190.0,  0.009,  0.99, 240.0,  0.013),  

        # n=1000
        (1000, 100)  => (0.05,   200.0, 0.05,   0.99, 2000.0, 0.99),
        (1000, 200)  => (0.0001, 500.0, 0.0001, 0.99, 1500.0, 0.99),
        (1000, 400)  => (0.0001, 0.01,  0.0001, 0.99, 100.0,  0.99),
        (1000, 800)  => (0.70,   200.0, 0.002,  0.99, 2500.0, 0.04),  
        (1000, 1600) => (0.2,    0.01,  0.01,   0.99, 100.0,  0.99)
    )

    b["DDA-COPT"] = Dict(
        # n=400
        (400, 100)  => (0.01,  0.01,   0.001,  0.99, 100.0,  0.99),
        (400, 200)  => (0.001, 0.01,   0.0001, 0.99, 100.0,  0.99),
        (400, 400)  => (0.8,   50.0,   0.005,  0.99, 210.0,  0.08),  
        (400, 800)  => (0.97,  170.0,  0.008,  0.99, 250.0,  0.013),  
        (400, 1600) => (0.975, 190.0,  0.009,  0.99, 240.0,  0.013),  

        # n=1000
        (1000, 100)  => (0.05,   200.0, 0.05,   0.99, 2000.0, 0.99),
        (1000, 200)  => (0.0001, 500.0, 0.0001, 0.99, 1500.0, 0.99),
        (1000, 400)  => (0.0001, 0.01,  0.0001, 0.99, 100.0,  0.99),
        (1000, 800)  => (0.70,   200.0, 0.002,  0.99, 2500.0, 0.04),  
        (1000, 1600) => (0.2,    0.01,  0.01,   0.99, 100.0,  0.99)
    )

    return b
end

# -----------------------------------------------------------------------------
# Run all methods/configs
# -----------------------------------------------------------------------------
function run_all_methods(; seed::Int=12345, n_rep::Int=400, bo_maxiter::Int=250)
    methods = ["DDA-COSMO", "DDA-Mosek", "DDA-COPT"]
    bnds = bounds_map_sim2()

    ns = [400, 1000]
    ps = [100, 200, 400, 800, 1600]

    rows = DataFrame(
        method=String[], solver=String[],
        n=Int[], p=Int[],
        RMSE=Float64[], TIter=Float64[], TotT=Float64[],
        alp=Float64[], lambda=Float64[], r=Float64[],
        Arg1=Float64[], Arg2=Float64[], Arg3=Float64[],
        Arg4=Float64[], Arg5=Float64[], Arg6=Float64[],
        seed=Int[], seed_tune=Int[], bo_maxiter=Int[]
    )

    for method in methods
        solver_choice = split(method, "-")[2]  # "COSMO" / "Mosek" / "COPT"

        for n in ns, p in ps
            @assert haskey(bnds[method], (n, p)) "Missing bounds for $method (n=$n,p=$p)"
            (aL, lL, rL, aU, lU, rU) = bnds[method][(n, p)]

            println("\n" * "="^100)
            println("SIM2 RUN | $method | n=$n p=$p | reps=$n_rep")
            println("Bounds: α∈[$aL,$aU], λ∈[$lL,$lU], r∈[$rL,$rU]")
            println("Wrapper seed=$seed -> eval seed_i = seed + gid")
            println("Tuning seed is fixed inside main(): seed_tune = 12345")
            println("="^100)

            out = Base.invokelatest(
                main,
                n, p,
                aL, lL, rL, aU, lU, rU,
                n_rep, solver_choice;
                seed=seed,
                bo_maxiter=bo_maxiter
            )

            push!(rows, (
                method=method, solver=String(solver_choice),
                n=n, p=p,
                RMSE=out.RMSE[1], TIter=out.T_per_iter[1], TotT=out.TotT[1],
                alp=out.alp[1], lambda=out.lambda[1], r=out.r[1],
                Arg1=aL, Arg2=lL, Arg3=rL, Arg4=aU, Arg5=lU, Arg6=rU,
                seed=seed, seed_tune=out.seed_tune[1], bo_maxiter=out.bo_maxiter[1]
            ))

            CSV.write("sim2_dda_all_results_checkpoint.csv", rows)
        end

        CSV.write("sim2_$(lowercase(replace(method, '-' => '_')))_results.csv",
                  rows[rows.method .== method, :])
    end

    CSV.write("sim2_dda_all_results.csv", rows)
    return rows
end

# -----------------------------------------------------------------------------
# Build FOUR LaTeX tables:
#   n=400  : part1 (p=100,200,400), part2 (p=800,1600)
#   n=1000 : part1 (p=100,200,400), part2 (p=800,1600)
# Bold = best (minimum) across methods for each (p, metric) block.
# -----------------------------------------------------------------------------
function build_tables_by_n(rows::DataFrame;
                           outfile400::AbstractString="sim2_table_n400.tex",
                           outfile1000::AbstractString="sim2_table_n1000.tex")

    methods = ["DDA-COSMO","DDA-Mosek","DDA-COPT"]

    function pick(method::String, n::Int, p::Int)
        sub = rows[(rows.method .== method) .& (rows.n .== n) .& (rows.p .== p), :]
        isempty(sub) && error("Missing row: method=$method n=$n p=$p")
        return sub[1, :]
    end

    function mins_for(n::Int, p::Int)
        block = rows[(rows.n .== n) .& (rows.p .== p), :]
        isempty(block) && error("Empty block n=$n p=$p")
        return (minimum(block.RMSE), minimum(block.TIter), minimum(block.TotT))
    end

    function build_one(nval::Int, ps_sub::Vector{Int}, outfile::AbstractString; caption_suffix::String="")
        io = IOBuffer()
        println(io, "\\begin{table}[ht]")
        println(io, "\\centering")
        println(io,
            "\\caption{(SIM2) RMSE, average time per iteration (T/Iter (s)), and total time (Tot.T (s)) for DDA under misspecified model with \$n = $(nval)\$" *
            (isempty(caption_suffix) ? "" : " " * caption_suffix) *
            ". Bold values indicate best performance (minimum) across DDA solvers for each \$(p,\\text{metric})\$ block.}"
        )
        println(io, "\\begin{tabular}{|l|" * "ccc|"^length(ps_sub) * "}")
        println(io, "\\hline")

        print(io, "Method ")
        for p in ps_sub
            print(io, "& \\multicolumn{3}{c|}{\$p=$(p)\$} ")
        end
        println(io, "\\\\")
        println(io, "\\cline{2-" * string(1 + 3*length(ps_sub)) * "}")

        print(io, " ")
        for _ in ps_sub
            print(io, "& RMSE & T/Iter & Tot.T ")
        end
        println(io, "\\\\")
        println(io, "\\hline")

        mins = Dict{Int,Tuple{Float64,Float64,Float64}}()
        for p in ps_sub
            mins[p] = mins_for(nval, p)
        end

        for m in methods
            print(io, m, " ")
            for p in ps_sub
                r = pick(m, nval, p)
                (rm_min, ti_min, tt_min) = mins[p]
                print(io, "& ", cell3(r.RMSE, rm_min),
                          " & ", cell3(r.TIter, ti_min),
                          " & ", cell3(r.TotT, tt_min), " ")
            end
            println(io, "\\\\")
        end

        println(io, "\\hline")
        println(io, "\\end{tabular}")
        println(io, "\\end{table}")

        open(outfile, "w") do f
            write(f, String(take!(io)))
        end
        return nothing
    end

    ps_part1 = [100, 200, 400]
    ps_part2 = [800, 1600]

    out400_1  = replace(outfile400,  ".tex" => "_part1.tex")
    out400_2  = replace(outfile400,  ".tex" => "_part2.tex")
    out1000_1 = replace(outfile1000, ".tex" => "_part1.tex")
    out1000_2 = replace(outfile1000, ".tex" => "_part2.tex")

    build_one(400,  ps_part1, out400_1;  caption_suffix="(Part 1: \$p\\in\\{100,200,400\\}\$)")
    build_one(400,  ps_part2, out400_2;  caption_suffix="(Part 2: \$p\\in\\{800,1600\\}\$)")
    build_one(1000, ps_part1, out1000_1; caption_suffix="(Part 1: \$p\\in\\{100,200,400\\}\$)")
    build_one(1000, ps_part2, out1000_2; caption_suffix="(Part 2: \$p\\in\\{800,1600\\}\$)")

    return nothing
end

# ------------------------- EXEC -------------------------
seed = 12345
n_rep = 400
bo_maxiter = 250

results_df = run_all_methods(; seed=seed, n_rep=n_rep, bo_maxiter=bo_maxiter)
CSV.write("sim2_dda_all_results.csv", results_df)

build_tables_by_n(results_df; outfile400="sim2_table_n400.tex", outfile1000="sim2_table_n1000.tex")

println("\n--- SIM2 ALL RUNS COMPLETE ---")
println("CSV saved as: sim2_dda_all_results.csv")
println("LaTeX saved as: sim2_table_n400_part1.tex, sim2_table_n400_part2.tex, sim2_table_n1000_part1.tex, sim2_table_n1000_part2.tex")
