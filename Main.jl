if !isdefined(Main, :SupplyChains)
    include("SupplyChains.jl")
end
if !isdefined(Main, :CascadeModel)
    include("CascadeModel.jl")
end

using .SupplyChains
using .CascadeModel

using Random
using Statistics
using Distributions
using ProgressMeter
using LinearAlgebra
using ForwardDiff: derivative

using JLD2
using Dates

using CairoMakie
using GraphMakie
CairoMakie.activate!()


##### EXAMPLE CONSTRAINT MATRICES #####

# Used for figures 2, 3, 4
function nested_region_matrix(n1,n2,v1,v2)
    c = zeros(n1+n2,n1+n2)
    c[1:n1,1:n1] .= v1
    c[(n1+1):end,n1:end] .= v2
    return c
end

# Used for figures 5,6
function twoish_layer_matrix(n1,n2, v1,v2)
    c = zeros(n1+n2+1,n1+n2+1)
    c[1:(n1+1),1:(n1+1)] .= v1
    c[(n1+1):(n1+n2+1),(n1+1):(n1+n2+1)] .= v2
    return c
end


##### CALCULATION & SIMULATION #####

# Cascade size calculation in three flavors
function analytic_cascade_size(constraint_mat::AbstractMatrix, in_deg_dist::DiscreteDistribution, ρ0::Real, steps::Integer = 5000, cutoff = 50)
    L = size(constraint_mat)[1]
    q = fill(Float64(ρ0), L)    # the initial value matters! this iteration can have multiple fixed points
    q_new = copy(q)

    function g(x::Float64)::Float64 # the almost-generating function term that we evaluate a bunch of times
        res = 1.0
        for k in 1:cutoff
            res -= pdf(in_deg_dist, k) * x^k
        end
        return res
    end

    for _ in 1:steps
        for i in 1:L
            product = 1.0
            for j in 1:L
                c = constraint_mat[j,i]
                if !iszero(c)
                    product *= (1.0-c) + c*g(q[j])
                end
            end
            q_new[i] = 1 - (1-ρ0)*product
        end
        if maximum(abs.(q-q_new)) < 1e-9
            break
        end
        q .= q_new
    end

    rho_final = mean(q)

    return rho_final
end
function analytic_cascade_size_poisson(constraint_mat::AbstractMatrix, z::Real, ρ0::Real, steps::Integer = 5000)
    L = size(constraint_mat)[1]
    q = fill(Float64(ρ0), L)    # the initial value matters! this iteration can have multiple fixed points
    q_new = copy(q)

    for _ in 1:steps
        for i in 1:L
            product = 1.0
            for j in 1:L
                c = constraint_mat[j,i]
                if !iszero(c)
                    product *= 1.0 + c*( exp(-z) - exp(z*(q[j]-1.0)) )
                end
            end
            q_new[i] = 1 - (1-ρ0)*product
        end
        if maximum(abs.(q-q_new)) < 1e-9
            break
        end
        q .= q_new
    end

    rho_final = mean(q)

    return rho_final
end
function analytic_cascade_size_poisson_itercount(constraint_mat::AbstractMatrix, z::Real, ρ0::Real, steps::Integer = 5000)
    L = size(constraint_mat)[1]
    q = fill(Float64(ρ0), L)    # the initial value matters! this iteration can have multiple fixed points
    q_new = copy(q)
    itc = 0

    for _ in 1:steps
        itc += 1
        for i in 1:L
            product = 1.0
            for j in 1:L
                c = constraint_mat[j,i]
                if !iszero(c)
                    product *= 1.0 + c*( exp(-z) - exp(z*(q[j]-1.0)) )
                end
            end
            q_new[i] = 1 - (1-ρ0)*product
        end
        if maximum(abs.(q-q_new)) < 1e-9
            break
        end
        q .= q_new
    end
    rho_final = mean(q)

    return rho_final, itc
end

# critical_z_poisson gives the phase transitions predicted by the analytic cascade condition
function solve(f::Function, xmin, xmax, y::Number, x0::AbstractArray{T}; tolerance::Real = 1e-6)::Array{T} where T<:Number
    res = T[]
    for x in x0
        for _ in 1:16
            if !isfinite(x) || !(xmin < x < xmax)
                break
            end
            df = derivative(f, x)
            x = x - (f(x) - y)/df
        end
        if isfinite(x) && xmin < x < xmax  && abs(f(x)-y) < tolerance
            push!(res, x)
        end
    end
    return res
end
function deduplicate(a::Array{T}; tolerance::Real = 1e-6)::Array{T} where T<:Real
    if isempty(a)
        return T[]
    end
    res = [a[1]]
    for x in a[2:end]
        if minimum(abs.(res .- x)) > tolerance
            push!(res, x)
        end
    end
    return res
end
function critical_z_poisson(constraint_mat::AbstractMatrix, zmin, zmax)
    λc = maximum(abs.(eigvals(Matrix(constraint_mat))))
    #println("LAMBDA_C = $λc")
    xs = solve((z->pdf(Poisson(z),1)) ,zmin, zmax, 1/λc, range(zmin,zmax, length=20))
    xs = deduplicate(xs)
    return xs
end

# Cascade probability calculation
function analytic_cascade_prob_poisson(constraint_mat::AbstractMatrix, z::Real, steps::Integer = 10_000)
    # some constants
    L = size(constraint_mat)[1]
    rowsums = sum(constraint_mat, dims=2)
    r_inv = 1.0 ./ rowsums
    zv = z*exp(-z)

    # iterate to fixed point
    x = fill(1e-3, L)
    x_new = zeros(L)
    for _ in 1:steps
        F = exp.(zv * rowsums .* (x .- 1.0))
        x_new .= r_inv .* constraint_mat * F
        if maximum(abs.(x-x_new)) < 1e-9
            break
        end
        x .= x_new
    end

    # return P_trig:
    F = exp.(zv * rowsums .* (x .- 1.0))
    return 1.0 - mean(F)
end

# These return the final cascade size, starting from macroscopic seeds
function simulate(sc::SupplyChain, rho0::Real)::Float64
    iv = rand(sc.N_vertices) .< rho0
    sim = initSim(sc, iv)
    for _ in 1:10_000
        step!(sim)
    end
    return mean(getActive(sim))
end
function simulate(sc::SupplyChain, rho0::Real, runs::Integer)::Float64
    results = zeros(runs)
    iv = rand(sc.N_vertices) .< rho0
    sim = initSim(sc, iv)
    for _ in 1:10_000
        step!(sim)
    end
    results[1] = mean(getActive(sim))

    for r in 2:runs
        iv = rand(sc.N_vertices) .< rho0
        reinitSim!(sim, iv)
        for _ in 1:10_000
            step!(sim)
        end
        results[r] = mean(getActive(sim))
    end
    
    return mean(results)
end


##### PLOTS WITHOUT SIMULATIONS #####

# plots analytic cascade size and probability. fast.
function figure_analytic(c, zmin, zmax)
    size_constr = Float64[]
    prob_constr = Float64[]
    Zs = range(zmin,zmax, 1000)
    
    @showprogress for z in Zs
        analytic = analytic_cascade_size_poisson(c,z, 1e-9, 1000_000)
        push!(size_constr, analytic)
        analytic = analytic_cascade_prob_poisson(c,z, 1000_000)
        push!(prob_constr, analytic)
    end

    f = Figure(size=(800,400))
    #top = f[1,1]
    #Label(top, "Final cascade sizes for ρ0 = $rho0",
    #        valign=:bottom, tellwidth=false, font=:bold)
    ax = Axis(f[1, 1], xlabel="z", limits=(zmin,zmax,-.025,1.025))

    color0 = :black
    color1 = :red

    lines!(ax, Zs, size_constr,  label="Size, C" , color = color0)
    lines!(ax, Zs, prob_constr,  label="Prob, C" , color = color1)

    #axislegend(ax, position=:rc)

    Legend(f[1,2], ax, framevisible=true, merge=true, unique=true,
        orientation=:vertical, tellwidth = true, tellheight = false)

    display(f)
    return f
end

# Figure 6
function figure_cusp()
    zmin = 0
    zmax = 5

    c1 = twoish_layer_matrix(20,20, .95, .12)
    c2 = twoish_layer_matrix(20,20, .95, .131)
    c3 = twoish_layer_matrix(20,20, .95, .14)
    c4 = twoish_layer_matrix(20,20, .95, .25)
    c5 = twoish_layer_matrix(20,20, .95, .22)

    size1 = Float64[]
    size2 = Float64[]
    size3 = Float64[]
    size4 = Float64[]
    size5 = Float64[]
    Zs2 = range(zmin,zmax, 500)
    
    @showprogress for z in Zs2
        analytic = analytic_cascade_size_poisson(c1,z, 1e-9, 1000_000)
        push!(size1, analytic)
        analytic = analytic_cascade_size_poisson(c2,z, 1e-9, 1000_000)
        push!(size2, analytic)
        analytic = analytic_cascade_size_poisson(c3,z, 1e-9, 1000_000)
        push!(size3, analytic)
        analytic = analytic_cascade_size_poisson(c4,z, 1e-9, 1000_000)
        push!(size4, analytic)
        analytic = analytic_cascade_size_poisson(c5,z, 1e-9, 1000_000)
        push!(size5, analytic)
    end

    f = Figure(size=(700,400))
    #top = f[1,1]
    #Label(top, "Final cascade sizes for ρ0 = $rho0",
    #        valign=:bottom, tellwidth=false, font=:bold)
    ax = Axis(f[1, 1], limits=(zmin,zmax,-.025,1.025), xticklabelsvisible=false,yticklabelsvisible=false)

    color0 = :black
    color1 = :gray
    color2 = :pink
    color1 = :black
    color2 = :black

    lines!(ax, Zs2, size1,  label="" , color = color1, linestyle=:dash)
    lines!(ax, Zs2, size3,  label="" , color = color2, linestyle=:dash)
    lines!(ax, Zs2, size4,  label="" , color = color1, linestyle=:dot)
    lines!(ax, Zs2, size5,  label="" , color = color2, linestyle=:dot)
    lines!(ax, Zs2, size2,  label="" , color = color0)


    #axislegend(ax, position=:rc)

    #Legend(f[1,2], ax, framevisible=true, merge=true, unique=true,
    #    orientation=:vertical, tellwidth = true, tellheight = false)

    display(f)
    return f
end

# A bit of a gimmick: you can automatically count the phase transitions along a given parameter sweep.
function figure_itercount(c, zmin, zmax)
    size_constr = Float64[]
    itcount = Float64[]
    Zs2 = range(zmin,zmax, 1000)
    
    @showprogress for z in Zs2
        analytic, itc = analytic_cascade_size_poisson_itercount(c,z, 1e-9, 1000_000)
        push!(size_constr, analytic)
        push!(itcount, itc)
    end
    itcount /= maximum(itcount)

    f = Figure(size=(800,400))
    #top = f[1,1]
    #Label(top, "Final cascade sizes for ρ0 = $rho0",
    #        valign=:bottom, tellwidth=false, font=:bold)
    ax = Axis(f[1, 1], xlabel="z", limits=(zmin,zmax,-.025,1.025))#, xticks=zmin:zmax)

    color0 = :black
    color1 = :red

    transition_count = 0
    for i in 2:(length(itcount)-1)
        if itcount[i-1] < itcount[i] > itcount[i+1]
            transition_count += 1
        end
    end
    println("Transitions Counted: $transition_count")


    lines!(ax, Zs2, size_constr,  label="Size" , color = color0)
    lines!(ax, Zs2, itcount,  label="Iterations" , color = color1)

    #axislegend(ax, position=:rc)

    Legend(f[1,2], ax, framevisible=true, merge=true, unique=true,
        orientation=:vertical, tellwidth = true, tellheight = false)

    display(f)
    return f
end
function heatmap_transitions()
    Zs2 = range(0,10, 250)

    Ps = range(0,1,25)
    Qs = range(0,1,25)

    transitions = zeros(length(Ps),length(Qs))
    
    @showprogress for (i,p) in enumerate(Ps), (j,q) in enumerate(Qs)
        c = twoish_layer_matrix(12,12, p,q)
        itcount = Float64[]
        for z in Zs2
            _, itc = analytic_cascade_size_poisson_itercount(c,z, 1e-9, 1000_000)
            push!(itcount, itc)
        end

        transition_count = 0
        for k in 2:(length(itcount)-1)
            if itcount[k-1] < itcount[k] > itcount[k+1]
                transition_count += 1
            end
        end
        transitions[i,j] = transition_count
    end


    f = Figure(size=(550,500))
    ax = Axis(f[1, 1], limits=(0,1,0,1))

    hm = heatmap!(ax, Ps,Qs, transitions)

    Colorbar(f[1,2],hm)

    display(f)
    return f
end


###### SIMULATE TO STRUCT ######
# (save yourself time and write these structs to disk with JLD2.save_object)
# microscopic seeds
mutable struct SimulationDataMicroscopic
    # Arbitrary Text, just in case
    name::String
    description::String

    # Fill this automatically
    history::String

    # Actual Data
    constraint_mat::AbstractMatrix
    n_nodes::Int64
    observations::Vector{Tuple{Float64,Float64}} # Pairs of (z,ρ_final) (with ρ_final normalized to [0,1])
end
function empty_data_micro(c, N)::SimulationDataMicroscopic
    new_history = "$(Dates.format(now(), "yyyy-mm-dd HH:MM")) : Created."
    data = SimulationDataMicroscopic("", "", new_history, copy(c), N, Tuple{Float64,Float64}[])
    return data
end
function extend_data_micro!(data::SimulationDataMicroscopic, runs::Integer, samples_per_run::Integer, Zs::AbstractVector)
    new_observations = Array{Tuple{Float64,Float64}}(undef, length(Zs)*runs*samples_per_run )
    fill!(new_observations, (NaN64,NaN64))

    index = 1
    for z in Zs
        println("")
        println("----------- Now at z=$z -----------")
        println("")
        for _ in 1:runs
            sc = sc_constrained_poisson(data.n_nodes, data.constraint_mat, z)
            c_sizes = cascadeSizes(sc, samples_per_run) ./ data.n_nodes
            for s in c_sizes
                new_observations[index] = (z,s)
                index += 1
            end
        end
    end
    push!(data.observations, new_observations...)

    data.history *= "\n$(Dates.format(now(), "yyyy-mm-dd HH:MM")) : Extended with $(length(new_observations)) observations:\n\
                        \t\truns = $runs\n\t\tsamples_per_run = $samples_per_run\n\t\tlength(Zs) = $(length(Zs))\n\t\tZs = $Zs"

    return data
end
function extend_data_micro!(data::SimulationDataMicroscopic, runs::Integer, samples_per_run::Integer, zmin::Real, zmax::Real)
    Zs = range(zmin,zmax, length=25)
    return extend_data_micro!(data, runs, samples_per_run, Zs)
end

# macroscopic seeds
mutable struct SimulationDataMacroscopic
    # Arbitrary Text, just in case
    name::String
    description::String

    # Fill this automatically
    history::String

    # Actual Data
    constraint_mat::AbstractMatrix
    rho0::Float64
    n_nodes::Int64
    observations::Vector{Tuple{Float64,Float64}} # Pairs of (z,ρ_final) (with ρ_final normalized to [0,1])
end
function empty_data_macro(c, N, rho0)::SimulationDataMacroscopic
    new_history = "$(Dates.format(now(), "yyyy-mm-dd HH:MM")) : Created."
    data = SimulationDataMacroscopic("", "", new_history, copy(c), rho0, N, Tuple{Float64,Float64}[])
    return data
end
function extend_data_macro!(data::SimulationDataMacroscopic, runs::Integer, samples_per_run::Integer, Zs::AbstractVector)
    new_observations = Array{Tuple{Float64,Float64}}(undef, length(Zs)*runs*samples_per_run )
    fill!(new_observations, (NaN64,NaN64))

    index = 1
    for z in Zs
        println("")
        println("----------- Now at z=$z -----------")
        println("")
        for _ in 1:runs
            sc = sc_constrained_poisson(data.n_nodes, data.constraint_mat, z)
            for _ in 1:samples_per_run
                s = simulate(sc, data.rho0)
                new_observations[index] = (z,s)
                index += 1
            end
        end
    end
    push!(data.observations, new_observations...)

    data.history *= "\n$(Dates.format(now(), "yyyy-mm-dd HH:MM")) : Extended with $(length(new_observations)) observations:\n\
                        \t\truns = $runs\n\t\tsamples_per_run = $samples_per_run\n\t\tlength(Zs) = $(length(Zs))\n\t\tZs = $Zs"

    return data
end
function extend_data_macro!(data::SimulationDataMacroscopic, runs::Integer, samples_per_run::Integer, zmin::Real, zmax::Real)
    Zs = range(zmin,zmax, length=25)
    return extend_data_macro!(data, runs, samples_per_run, Zs)
end


###### PLOT FROM STRUCT ######
# (load these structs from disk with JLD2.load_object)
function figure_micro(data::SimulationDataMicroscopic)
    analytic_size = Float64[]
    analytic_prob = Float64[]

    zmin = minimum(first.(data.observations))
    zmax = maximum(first.(data.observations))

    # Analytic results
    Zs2 = range(zmin,zmax, 500)
    @showprogress for z in Zs2
        analytic = analytic_cascade_size_poisson(data.constraint_mat,z, 1e-9, 1000_000)
        push!(analytic_size, analytic)

        analytic = analytic_cascade_prob_poisson(data.constraint_mat,z, 1000_000)
        push!(analytic_prob, analytic)
    end

    # Observed Probabilities for each z
    observed_probs = Tuple{Float64,Float64}[]
    sorted_observations = sort(data.observations)
    current_z = first(sorted_observations[1])
    current_cascades = 0
    current_count = 0
    for o in sorted_observations
        if first(o) != current_z
            push!(observed_probs, (current_z, current_cascades / current_count) )
            current_cascades = 0
            current_count = 0
            current_z = first(o)
        end
        current_count += 1
        if last(o) > 1e-2
            current_cascades += 1
        end
    end

    # Observed Distinct Cascade Sizes for each z
    unique_observations = sort(unique(data.observations))
    observed_sizes = Tuple{Float64,Float64}[]
    last_z, last_s = Inf64, Inf64
    for o in unique_observations
        if last(o) < 1e-2                   # don't draw small cascades
            #continue
        end
        if first(o) != last_z
            push!(observed_sizes, o)
            (last_z, last_s) = o
        end
        if abs(last(o) - last_s) > 1e-3     # filter out near-duplicates to draw fewer points
            push!(observed_sizes, o)
            (last_z, last_s) = o
        end
    end
    println("Original sizes:  $(length(data.observations))")
    println("Surviving sizes: $(length(observed_sizes))")

    f = Figure(size=(700,400))
    #f = Figure(size=(500,300))#350
    ax = Axis(f[1, 1], xticklabelsvisible=false,yticklabelsvisible=false, #xlabel="Mean Degree per layer",# ylabel="Cascade Size",
                limits=(zmin,zmax,-.025,1.025))#, yticks=0:.1:.5)

    color0 = :black
    color1 = :black

    # indicate analytically prediced transitions
    #for x in critical_z_poisson(data.constraint_mat, zmin, zmax)
    #    lines!(ax, [(x,0),(x,1)], color = :red)
    #end

    lines!(ax, Zs2, analytic_size,  label="Cascade Size (Theory)" , color = color1)
    lines!(ax, Zs2, analytic_prob,  label="Probability (Theory)"      , color = color0 , linestyle= :dash)

    scatter!(ax, observed_sizes,  label="Cascade Sizes (Observed)" , marker=:rect  , color = color1, markersize = 12)
    scatter!(ax, observed_probs,  label="Probability (Observed)"      , marker=:circle  , color = :white, strokewidth=1, strokecolor=color0)

    #axislegend(ax, position=:cc)

    #Legend(f[1,2], ax, framevisible=true, merge=true, unique=true,
    #    orientation=:vertical, tellwidth = true, tellheight = false)

    display(f)
    return f
end
function figure_macro(data::SimulationDataMacroscopic)
    zmin = minimum(first.(data.observations))
    zmax = maximum(first.(data.observations))
    Zs2 = range(zmin,zmax, 500)

    analytic_size = Float64[]
    @showprogress for z in Zs2
        analytic = analytic_cascade_size_poisson(data.constraint_mat,z, data.rho0)
        push!(analytic_size, analytic)
    end


    # Observed Distinct Cascade Sizes for each z
    unique_observations = sort(unique(data.observations))
    observed_sizes = Tuple{Float64,Float64}[]
    last_z, last_s = Inf64, Inf64
    for o in unique_observations
        if first(o) != last_z
            push!(observed_sizes, o)
            (last_z, last_s) = o
        end
        if abs(last(o) - last_s) > 1e-3     # filter out near-duplicates to draw fewer points
            push!(observed_sizes, o)
            (last_z, last_s) = o
        end
    end
    println("Original sizes:  $(length(data.observations))")
    println("Surviving sizes: $(length(observed_sizes))")


    f = Figure(size=(800,400))
    top = f[1,1]
    #Label(top, "Final cascade sizes for ρ0 = $rho0",
    #        valign=:bottom, tellwidth=false, font=:bold)
    ax = Axis(f[1, 1], limits=(zmin,zmax,-.025,1.025),# xlabel="z", ylabel="Final Cascade Size")#, xticks=zmin:zmax)
                xticklabelsvisible=false, yticklabelsvisible=false)
    color0 = :black
    color1 = :black

    # indicate analytically prediced transitions
    #for x in critical_z_poisson(c, zmin, zmax)
    #    lines!(ax, [(x,0),(x,1)], color = :red)
    #end

    lines!(ax, Zs2, analytic_size,  label="Theory" , color = color1)

    scatter!(ax, observed_sizes, label="Simulation" , marker=:rect  , color = color1, markersize = 12)

    #axislegend(ax, position=:rc)

    #Legend(f[1,2], ax, framevisible=true, merge=true, unique=true,
    #    orientation=:vertical, tellwidth = true, tellheight = false)

    display(f)
    return f
end

##### EXAMPLE USE #####
# To produce something like Figure 4, you could do:
# c = nested_region_matrix(3,3, 0.94, 1.0)
# data = empty_data_micro(c, 1000_000)
# extend_data_micro!(data, 1, 500, 0.5:.05:1.75)
# figure_micro(data)