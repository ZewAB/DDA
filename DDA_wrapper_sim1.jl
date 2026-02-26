# =============================================================================
# File: DDA_wrapper_sim1.jl
# Experiments executed on AMD EPYC 7302P (16-core / 32 GB RAM), Linux
#
# SIM1 wrapper: runs DDA-COSMO / DDA-Mosek / DDA-COPT over 8 configs
# using DDA_main_sim1.jl (BO once per config, bo_maxiter=250).
#
# SEEDS (SIM1):
#   - tuning seed 12345 
#   - evaluation seeds are seed_i = 12345 + gid, gid=1..400
#
## Recommended run:
#   JULIA_WORKER_TIMEOUT=300 julia --project=. --bind-to 127.0.0.1 -p 12 DDA_wrapper_sim1.jl
#
# Saves:
#   - sim1_dda_all_results.csv
#   - sim1_table2.tex
#   - sim1_table3.tex
#   - per-method CSVs
# =============================================================================

# Load main SIM1 implementation
include("DDA_main_sim1.jl")

using DataFrames, CSV, Printf

fmt3(x::Real) = @sprintf("%.3f", x)
bold(s::AbstractString) = "\\textbf{$s}"
ismin(x, xmin; atol=1e-12) = isfinite(x) && abs(x - xmin) <= atol
cell3(x::Real, best::Real) = ismin(x, best) ? bold(fmt3(x)) : fmt3(x)

const BSET_LABEL = Dict(1 => "dense", 2 => "harmonic", 3 => "moderately sparse", 4 => "very sparse")
const DSET_LABEL = Dict(2 => "dense", 1 => "sparse")  # 2=dense δ, 1=sparse δ

# Parameter settings for the first experimental setup
function bounds_map()
    b = Dict{String, Dict{Tuple{Int,Int}, NTuple{6,Float64}}}()

    b["DDA-COSMO"] = Dict(
        (1, 2) => (0.001, 0.001,   0.01,   0.99, 1000.0, 0.99), 
        (1, 1) => (0.01,  0.01,   0.0001, 0.99, 200.0,  0.99),
        (2, 2) => (0.001, 10.0,   0.001,  0.99, 1000.0, 0.99),
        (2, 1) => (0.001, 0.001,  0.001,  0.99, 400.0,  0.99),
        (3, 2) => (0.1,   0.1,    0.01,   0.99, 300.0,  0.99),
        (3, 1) => (0.1,   200.0,  0.5,    0.99, 2000.0, 0.99),
        (4, 2) => (0.001, 0.01,   0.0001, 0.99, 100.0,  0.99),
        (4, 1) => (0.1,   0.01,   0.01,   0.99, 100.0,  0.99)
    )

    
    b["DDA-Mosek"] = Dict(
        (1, 2) => (0.001, 0.001,  0.01,   0.99, 1000.0, 0.99),
        (1, 1) => (0.01,  0.01,   0.0001, 0.99, 200.0,  0.99),
        (2, 2) => (0.001, 10.0,  0.001,  0.99, 1000.0, 0.99), 
        (2, 1) => (0.001, 0.001,  0.001,  0.99, 400.0,  0.99),
        (3, 2) => (0.1,   0.1,    0.01,   0.99, 300.0,  0.99),
        (3, 1) => (0.1,   200.0,  0.5,    0.99, 2000.0, 0.99),
        (4, 2) => (0.001, 0.01,   0.0001, 0.99, 100.0,  0.99),
        (4, 1) => (0.1,   0.01,   0.01,   0.99, 100.0,  0.99)
    )

    b["DDA-COPT"] = Dict(
        (1, 2) => (0.1,   0.01,   0.01,   0.99, 100.0,  0.99),
        (1, 1) => (0.01,  0.01,   0.0001, 0.99, 200.0,  0.99),
        (2, 2) => (0.001, 10.0,   0.001,  0.99, 1000.0, 0.99),
        (2, 1) => (0.001, 0.001,  0.001,  0.99, 400.0,  0.99),
        (3, 2) => (0.1,   0.1,    0.01,   0.99, 300.0,  0.99),
        (3, 1) => (0.1,   200.0,  0.5,    0.99, 2000.0, 0.99),
        (4, 2) => (0.001, 0.01,   0.0001, 0.99, 100.0,  0.99),
        (4, 1) => (0.1,   0.01,   0.01,   0.99, 100.0,  0.99)
    )

    return b
end

function run_all_methods(; seed::Int=12345,
                         n::Int=500, p::Int=2000, C::Float64=2.0, eps::Float64=0.2,
                         n_rep::Int=400, bo_maxiter::Int=250)

    methods = ["DDA-COSMO", "DDA-Mosek", "DDA-COPT"]
    bnds = bounds_map()

    rows = DataFrame(
        method=String[], solver=String[],
        n=Int[], p=Int[], C=Float64[], eps=Float64[],
        b_setup=Int[], sparse_or_dense=Int[],
        b_setting=String[], delta_setting=String[],
        RMSE=Float64[], TIter=Float64[], TotT=Float64[],
        alp=Float64[], lambda=Float64[], r=Float64[],
        bo_seed=Int[], bo_maxiter=Int[], seed=Int[]
    )

    configs = [(b_setup, sOD) for b_setup in 1:4 for sOD in (2, 1)]

    for method in methods
        solver_choice = split(method, "-")[2]

        for (b_setup, sOD) in configs
            @assert haskey(bnds[method], (b_setup, sOD)) "Missing bounds for $method (b_setup=$b_setup, δ=$sOD)"
            (aL, lL, rL, aU, lU, rU) = bnds[method][(b_setup, sOD)]

            b_lab = BSET_LABEL[b_setup]
            d_lab = DSET_LABEL[sOD]

            println("\n" * "="^85)
            println("SIM1 RUN | $method | b=$b_lab | δ=$d_lab | n=$n p=$p C=$C eps=$eps reps=$n_rep")
            println("Bounds: α∈[$aL,$aU], λ∈[$lL,$lU], r∈[$rL,$rU]")
            println("seed=$seed | BO once bo_maxiter=$bo_maxiter | tuning seed fixed at 12345")
            println("="^85)

            out = Base.invokelatest(
                main,
                n, p, C, eps, b_setup,
                aL, lL, rL, aU, lU, rU,
                n_rep, sOD, solver_choice;
                seed=seed,
                bo_maxiter=bo_maxiter
            )

            rmse  = out.RMSE[1]
            titer = out.T_per_iter[1]
            totT  = out.TotT[1]
            alp   = out.alp[1]
            lam   = out.lambda[1]
            ropt  = out.r[1]
            bseed = out.bo_seed[1]
            bmit  = out.bo_maxiter[1]

            push!(rows, (
                method=method, solver=String(solver_choice),
                n=n, p=p, C=C, eps=eps,
                b_setup=b_setup, sparse_or_dense=sOD,
                b_setting=b_lab, delta_setting=d_lab,
                RMSE=rmse, TIter=titer, TotT=totT,
                alp=alp, lambda=lam, r=ropt,
                bo_seed=bseed, bo_maxiter=bmit, seed=seed
            ))

            CSV.write("sim1_dda_all_results_checkpoint.csv", rows)
        end

        CSV.write("sim1_$(lowercase(replace(method, '-' => '_')))_results.csv", rows[rows.method .== method, :])
    end

    CSV.write("sim1_dda_all_results.csv", rows)
    return rows
end

function build_table2_table3(rows::DataFrame; outfile2::AbstractString="sim1_table2.tex", outfile3::AbstractString="sim1_table3.tex")
    methods = ["DDA-COSMO","DDA-Mosek","DDA-COPT"]

    function block_mins(bset::String, dset::String)
        block = rows[(rows.b_setting .== bset) .& (rows.delta_setting .== dset), :]
        isempty(block) && error("Empty block: b_setting=$bset delta_setting=$dset")
        return (minimum(block.RMSE), minimum(block.TIter), minimum(block.TotT))
    end

    function pick(method::String, bset::String, dset::String)
        sub = rows[(rows.method .== method) .& (rows.b_setting .== bset) .& (rows.delta_setting .== dset), :]
        isempty(sub) && error("Missing row: method=$method b_setting=$bset delta_setting=$dset")
        return sub[1, :]
    end

    # ---- TABLE 2 ----
    io2 = IOBuffer()
    println(io2, raw"\begin{table}[ht]")
    println(io2, raw"\centering")
    println(io2, raw"{\small")
    println(io2, raw"\caption{(SIM1) RMSE, average time per iteration (T/Iter (s)), and total time (Tot.T (s)) for DDA using dense and harmonic $b$ settings. Bold values indicate best performance.}")
    println(io2, raw"\begin{tabular}{|l|ccc|ccc|ccc|ccc|}")
    println(io2, raw"\hline")
    println(io2, raw"\shortstack{b Setting\\$\delta$ Setting} & \multicolumn{6}{c|}{dense} & \multicolumn{6}{c|}{harmonic} \\")
    println(io2, raw"\cline{2-13}")
    println(io2, raw" & \multicolumn{3}{c|}{dense} & \multicolumn{3}{c|}{sparse} & \multicolumn{3}{c|}{dense} & \multicolumn{3}{c|}{sparse} \\")
    println(io2, raw"\cline{2-13}")
    println(io2, raw" & RMSE & T/Iter & Tot.T & RMSE & T/Iter & Tot.T & RMSE & T/Iter & Tot.T & RMSE & T/Iter & Tot.T \\")
    println(io2, raw"\hline")

    mins2 = Dict{Tuple{String,String},Tuple{Float64,Float64,Float64}}()
    for bset in ("dense","harmonic"), dset in ("dense","sparse")
        mins2[(bset,dset)] = block_mins(bset, dset)
    end

    for m in methods
        r_dd = pick(m, "dense", "dense")
        r_ds = pick(m, "dense", "sparse")
        r_hd = pick(m, "harmonic", "dense")
        r_hs = pick(m, "harmonic", "sparse")

        (rmse_dd, titer_dd, tot_dd) = mins2[("dense","dense")]
        (rmse_ds, titer_ds, tot_ds) = mins2[("dense","sparse")]
        (rmse_hd, titer_hd, tot_hd) = mins2[("harmonic","dense")]
        (rmse_hs, titer_hs, tot_hs) = mins2[("harmonic","sparse")]

        println(io2,
            m, " & ",
            cell3(r_dd.RMSE, rmse_dd), " & ", cell3(r_dd.TIter, titer_dd), " & ", cell3(r_dd.TotT, tot_dd), " & ",
            cell3(r_ds.RMSE, rmse_ds), " & ", cell3(r_ds.TIter, titer_ds), " & ", cell3(r_ds.TotT, tot_ds), " & ",
            cell3(r_hd.RMSE, rmse_hd), " & ", cell3(r_hd.TIter, titer_hd), " & ", cell3(r_hd.TotT, tot_hd), " & ",
            cell3(r_hs.RMSE, rmse_hs), " & ", cell3(r_hs.TIter, titer_hs), " & ", cell3(r_hs.TotT, tot_hs),
            raw" \\"
        )
    end

    println(io2, raw"\hline")
    println(io2, raw"\end{tabular}")
    println(io2, raw"}")
    println(io2, raw"\end{table}")
    open(outfile2, "w") do f
        write(f, String(take!(io2)))
    end

    # ---- TABLE 3 ----
    io3 = IOBuffer()
    println(io3, raw"\begin{table}[ht]")
    println(io3, raw"\centering")
    println(io3, raw"{\tiny")
    println(io3, raw"\caption{(SIM1) RMSE, average time per iteration (T/Iter (s)), and total time (Tot.T (s)) for DDA using moderately sparse and very sparse $b$ settings. Bold values indicate best performance.}")
    println(io3, raw"\begin{tabular}{|l|ccc|ccc|ccc|ccc|}")
    println(io3, raw"\hline")
    println(io3, raw"\shortstack{b Setting\\$\delta$ Setting} & \multicolumn{6}{c|}{moderately sparse} & \multicolumn{6}{c|}{very sparse} \\")
    println(io3, raw"\cline{2-13}")
    println(io3, raw" & \multicolumn{3}{c|}{Dense} & \multicolumn{3}{c|}{sparse} & \multicolumn{3}{c|}{dense} & \multicolumn{3}{c|}{sparse} \\")
    println(io3, raw"\cline{2-13}")
    println(io3, raw" & RMSE & T/Iter & Tot.T & RMSE & T/Iter & Tot.T & RMSE & T/Iter & Tot.T & RMSE & T/Iter & Tot.T \\")
    println(io3, raw"\hline")

    mins3 = Dict{Tuple{String,String},Tuple{Float64,Float64,Float64}}()
    for bset in ("moderately sparse","very sparse"), dset in ("dense","sparse")
        mins3[(bset,dset)] = block_mins(bset, dset)
    end

    for m in methods
        r_md = pick(m, "moderately sparse", "dense")
        r_ms = pick(m, "moderately sparse", "sparse")
        r_vd = pick(m, "very sparse", "dense")
        r_vs = pick(m, "very sparse", "sparse")

        (rmse_md, titer_md, tot_md) = mins3[("moderately sparse","dense")]
        (rmse_ms, titer_ms, tot_ms) = mins3[("moderately sparse","sparse")]
        (rmse_vd, titer_vd, tot_vd) = mins3[("very sparse","dense")]
        (rmse_vs, titer_vs, tot_vs) = mins3[("very sparse","sparse")]

        println(io3,
            m, " & ",
            cell3(r_md.RMSE, rmse_md), " & ", cell3(r_md.TIter, titer_md), " & ", cell3(r_md.TotT, tot_md), " & ",
            cell3(r_ms.RMSE, rmse_ms), " & ", cell3(r_ms.TIter, titer_ms), " & ", cell3(r_ms.TotT, tot_ms), " & ",
            cell3(r_vd.RMSE, rmse_vd), " & ", cell3(r_vd.TIter, titer_vd), " & ", cell3(r_vd.TotT, tot_vd), " & ",
            cell3(r_vs.RMSE, rmse_vs), " & ", cell3(r_vs.TIter, titer_vs), " & ", cell3(r_vs.TotT, tot_vs),
            raw" \\"
        )
    end

    println(io3, raw"\hline")
    println(io3, raw"\end{tabular}")
    println(io3, raw"}")
    println(io3, raw"\end{table}")
    open(outfile3, "w") do f
        write(f, String(take!(io3)))
    end

    return nothing
end

# ------------------------- EXEC -------------------------
seed = 12345
n = 500
p = 2000
C = 2.0 # Scaling factor
eps = 0.2 # Treatment probability
n_rep = 400
bo_maxiter = 250

results_df = run_all_methods(; seed=seed, n=n, p=p, C=C, eps=eps, n_rep=n_rep, bo_maxiter=bo_maxiter)
CSV.write("sim1_dda_all_results.csv", results_df)

build_table2_table3(results_df; outfile2="sim1_table2.tex", outfile3="sim1_table3.tex")

println("\n--- SIM1 ALL RUNS COMPLETE ---")
println("CSV saved as: sim1_dda_all_results.csv")
println("LaTeX saved as: sim1_table2.tex and sim1_table3.tex")
