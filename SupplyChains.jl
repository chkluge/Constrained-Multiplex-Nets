module SupplyChains
export SupplyChain, sc_constrained_poisson

using Graphs
using Random
using Distributions
using ProgressMeter

##### DATA TYPE #####
struct SupplyChain
    G::Array{SimpleDiGraph{Int64}, 1}
    N_industries::Int64
    N_vertices::Int64
    industry::Array{Int64, 1}           # type of node
end

##### HELPER FUNCTIONS #####
function pickWeighted(vals, weights, totalweight=sum(weights))::Int64
    # I compared this to StatsBase.sample and it has equal speed, less memory use
    x = rand() * totalweight
    for v in vals
        w = weights[v]
        if w > x
            return v
        end
        x -= w
    end
    @assert(false, "No return in pickWeighted: x = $x, totalweight = $totalweight")
end

##### RANDOM GRAPH GENERATORS #####

# for some reason this isn't implemented by Graphs.jl ?
# generate directed graph with length(k_in) vertices, vertex i has indegree k_in[i] and outdegree k_out[i]
function random_configuration_model_directed(k_in::Vector{Int64}, k_out::Vector{Int64}=k_in, verbose::Bool=false)::SimpleDiGraph
    N = length(k_in)

    @assert(length(k_out) == N, "Degree sequences need to be of equal length! length(k_in)=$(length(k_in)), length(k_out)=$(length(k_out))")
    @assert(minimum(k_in) ≥ 0 && minimum(k_out) ≥ 0, "Degree sequence has negative elements!")
    @assert(maximum(k_in) < N && maximum(k_out) < N, "Degree sequence has excessively large elements! maximum(k_in) = $(maximum(k_in)), maximum(k_out) = $(maximum(k_out))")
    #@assert(sum(k_in) == sum(k_out), "Total indegrees must equal total outdegrees! sum(k_in)=$(sum(k_in)), sum(k_out)=$(sum(k_out))")
    if verbose && sum(k_in) ≠ sum(k_out)
        println("WARNING: Total indegrees don't equal total outdegrees! sum(k_in)=$(sum(k_in)), sum(k_out)=$(sum(k_out))")
    end
    # The above conditions are necessary, but not sufficient. I know. I don't care.

    N = length(k_in)
    g = SimpleDiGraph(N)
    stubs_total = min(sum(k_in), sum(k_out))

    # Setup for fast sampling of many vertices with fewer distinct degrees
    verts_with_istubs = Vector{Set{Int64}}()
    for _ in 1:maximum(k_in)
        push!(verts_with_istubs, Set{Int64}())
    end
    for v in 1:N
        k_in[v]≤0 || push!(verts_with_istubs[k_in[v]], v)
    end
    verts_with_ostubs = Vector{Set{Int64}}()
    for _ in 1:maximum(k_out)
        push!(verts_with_ostubs, Set{Int64}())
    end
    for v in 1:N
        k_out[v]≤0 || push!(verts_with_ostubs[k_out[v]], v)
    end

    weights_i = length.(verts_with_istubs) .* (1:length(verts_with_istubs))
    weights_o = length.(verts_with_ostubs) .* (1:length(verts_with_ostubs))

    c = 0
    while stubs_total > 0
        # first sample the stubcounts...
        dout = pickWeighted(1:length(verts_with_ostubs), weights_o, sum(weights_o))
        din  = pickWeighted(1:length(verts_with_istubs), weights_i, sum(weights_i))

        # ... and then some vertices with those stubcounts
        v = rand(verts_with_ostubs[dout])
        w = rand(verts_with_istubs[din])

        if v != w && !has_edge(g,v,w)
            add_edge!(g,v,w)

            # ... bookkeeping ...
            pop!(verts_with_ostubs[dout], v)
            weights_o[dout] -= dout
            if dout > 1
                push!(verts_with_ostubs[dout-1], v)
                weights_o[dout-1] += dout-1
            end

            pop!(verts_with_istubs[din], w)
            weights_i[din] -= din
            if din > 1
                push!(verts_with_istubs[din-1], w)
                weights_i[din-1] += din-1
            end

            stubs_total -= 1
        else
            c += 1
        end
        if c > 1000
            if verbose
                println("Failed to satisfy degree sequence. Missing edges: $(stubs_total)")
            end
            break
        end
    end
    return g
end

##### RANDOM SUPPLY CHAINS #####

# This is a multiplex configuration model, with each node constrained to have out-stubs only on one layer.
# joint multi-degree distribution of a node on layer l is given by deg_dist[l]
# Samples from deg_dist are vectors of length (layers+1): 
#   Components 1:layers are the in-degrees. 
#   The last component is the out-degree on layer industry[v].
#   Out-degrees on the other layers are 0.
# verts_per_layer gives the number of nodes on each layer.
# Unfortunately, the odds of sampling a reasonable degree sequence are low.
#   (But the mismatch between in- and out- degrees becomes small for large networks.)
function sc_joint_full(verts_per_layer::AbstractVector{Int64}, deg_dist::AbstractVector{T})::SupplyChain where T<:DiscreteMultivariateDistribution

    # validate input
    N_inds = length(deg_dist)
    N_verts = sum(verts_per_layer)
    @assert(all(length.(deg_dist) .== N_inds+1), "sc_joint: deg_dist has inconsistent dimensions!")
    @assert(length(deg_dist) == length(verts_per_layer), "sc_joint: length(verts_per_layer)=$(length(verts_per_layer)); expected: $N_inds")
    @assert(all(verts_per_layer .> 0), "sc_joint: Layer with 0 nodes!")

    # label nodes
    layer_offset = zeros(Int64, N_inds)
    layer_offset[1] = 0
    for l in 2:N_inds
        layer_offset[l] = layer_offset[l-1] + verts_per_layer[l-1]
    end
    industry = zeros(Int64, sum(verts_per_layer))
    for l in 1:N_inds
        industry[layer_offset[l]+1 : layer_offset[l]+verts_per_layer[l]] .= l
    end

    # generate degree sequences
    idegseqs = Array{Int64}[]
    odegseqs = Array{Int64}[]
    for l in 1:N_inds
        push!(idegseqs, zeros(Int64, N_verts))
        push!(odegseqs, zeros(Int64, N_verts))
    end

    for v in 1:N_verts
        for _ in 1:1000
            D = rand(deg_dist[industry[v]])
            if any(D[1:end-1] .> verts_per_layer) || D[end] ≥ N_verts
                continue # obviously bad sample, reject
            end
            odegseqs[industry[v]][v] = D[end] 
            for l in 1:N_inds
                idegseqs[l][v] = D[l]
            end
            break # this is a good sample, move on to next vertex
        end
    end

    # now try to make the sequences valid.
    @showprogress for _ in 1:1_000_000
        if !all(sum.(idegseqs) .== sum.(odegseqs))
            # resample one vertex
            v = rand(1:N_verts)
            for _ in 1:1000
                D = rand(deg_dist[industry[v]])
                if any(D[1:end-1] .> verts_per_layer) || D[end] ≥ N_verts
                    continue # obviously bad sample, reject
                end
                odegseqs[industry[v]][v] = D[end] 
                for l in 1:N_inds
                    idegseqs[l][v] = D[l]
                end
                break # this is a good sample, move on to next vertex
            end
        end
    end

    if !all(sum.(idegseqs) .== sum.(odegseqs))
        println("sc_joint_full: Stub mismatch of size $(maximum(abs.(sum.(idegseqs) .- sum.(odegseqs))))")
    end

    G = [random_configuration_model_directed(idegseqs[l],odegseqs[l]) for l in 1:N_inds]

    return SupplyChain(G, N_inds, N_verts, industry)
end

# This is the actual "Constrained Multiplex Network" used in the paper.
function sc_constrained_poisson(N_verts::Integer, constraints::AbstractMatrix{Bool}, mean_indegree::Real)::SupplyChain

    # validate input
    N_inds = size(constraints)[1]
    @assert(size(constraints)[2] == N_inds, "Constraint matrix needs to be square!")

    # label nodes
    industry = rand(1:N_inds, N_verts)

    # generate degree sequences
    idegseqs = Array{Int64}[]
    odegseqs = Array{Int64}[]
    for l in 1:N_inds
        push!(idegseqs, zeros(Int64, N_verts))
        push!(odegseqs, zeros(Int64, N_verts))
    end

    for i in 1:N_inds
        inmask = constraints[i,:][industry]
        outmask = (industry .== i)
        mean_outdegree = count(inmask) / count(outmask) * mean_indegree

        idegseqs[i][inmask] .= rand(Poisson(mean_indegree), count(inmask))
        odegseqs[i][outmask].= rand(Poisson(mean_outdegree), count(outmask))

        idegseqs[i] .= min.(idegseqs[i], count(outmask))
        odegseqs[i] .= min.(odegseqs[i], count(inmask))
        idegseqs[i][inmask .&& outmask]  .= min.(idegseqs[i][inmask .&& outmask], count(outmask)-1)
        odegseqs[i][inmask .&& outmask]  .= min.(odegseqs[i][inmask .&& outmask], count(inmask)-1)

        # resample until valid
        for _ in 1:1000
            if sum(idegseqs[i]) == sum(odegseqs[i])
                break
            end
            v = rand((1:N_verts)[inmask])
            w = rand((1:N_verts)[outmask])
            idegseqs[i][v] = rand(Poisson(mean_indegree))
            odegseqs[i][w] = rand(Poisson(mean_outdegree))
            if outmask[v]
                idegseqs[i][v] = min(idegseqs[i][v], count(outmask)-1)
            else
                idegseqs[i][v] = min(idegseqs[i][v], count(outmask))
            end
            if inmask[w]
                odegseqs[i][w] = min(odegseqs[i][w], count(inmask)-1)
            else
                odegseqs[i][w] = min(odegseqs[i][w], count(inmask))
            end
        end
    end

    if !all(sum.(idegseqs) .== sum.(odegseqs))
        println("sc_joint_full: Stub mismatch of size $(maximum(abs.(sum.(idegseqs) .- sum.(odegseqs))))")
    end

    G = [random_configuration_model_directed(idegseqs[l],odegseqs[l]) for l in 1:N_inds]

    return SupplyChain(G, N_inds, N_verts, industry)
end
function sc_constrained_poisson(N_verts::Integer, constraints::AbstractMatrix{Float64}, mean_indegree::Real)::SupplyChain
    
     # validate input
     N_inds = size(constraints)[1]
     @assert(size(constraints)[2] == N_inds, "Constraint matrix needs to be square!")
     @assert(all(0 .≤ constraints .≤ 1), "Constraints need to be in [0,1]!")
 
     # label nodes
     industry = rand(1:N_inds, N_verts)
 
     # generate degree sequences
     idegseqs = Array{Int64}[]
     odegseqs = Array{Int64}[]
     for l in 1:N_inds
         push!(idegseqs, zeros(Int64, N_verts))
         push!(odegseqs, zeros(Int64, N_verts))
     end
 
     for i in 1:N_inds
         inmask = rand(N_verts) .< constraints[i,:][industry]
         outmask = (industry .== i)
         mean_outdegree = count(inmask) / count(outmask) * mean_indegree
 
         idegseqs[i][inmask] .= rand(Poisson(mean_indegree), count(inmask))
         odegseqs[i][outmask].= rand(Poisson(mean_outdegree), count(outmask))
 
         idegseqs[i] .= min.(idegseqs[i], count(outmask))
         odegseqs[i] .= min.(odegseqs[i], count(inmask))
         idegseqs[i][inmask .&& outmask]  .= min.(idegseqs[i][inmask .&& outmask], count(outmask)-1)
         odegseqs[i][inmask .&& outmask]  .= min.(odegseqs[i][inmask .&& outmask], count(inmask)-1)
 
         # resample until valid
         for _ in 1:1000
             if true || sum(idegseqs[i]) == sum(odegseqs[i]) # ACTUALLY DON'T RESAMPLE IT TAKES FOREVER
                 break
             end
             v = rand((1:N_verts)[inmask])
             w = rand((1:N_verts)[outmask])
             idegseqs[i][v] = rand(Poisson(mean_indegree))
             odegseqs[i][w] = rand(Poisson(mean_outdegree))
             if outmask[v]
                 idegseqs[i][v] = min(idegseqs[i][v], count(outmask)-1)
             else
                 idegseqs[i][v] = min(idegseqs[i][v], count(outmask))
             end
             if inmask[w]
                 odegseqs[i][w] = min(odegseqs[i][w], count(inmask)-1)
             else
                 odegseqs[i][w] = min(odegseqs[i][w], count(inmask))
             end
         end
     end
 
     if !all(sum.(idegseqs) .== sum.(odegseqs))
         println("sc_joint_full: Stub mismatch of size $(maximum(abs.(sum.(idegseqs) .- sum.(odegseqs))))")
     end
 
     G = [random_configuration_model_directed(idegseqs[l],odegseqs[l]) for l in 1:N_inds]
 
     return SupplyChain(G, N_inds, N_verts, industry)
end

end # END MODULE