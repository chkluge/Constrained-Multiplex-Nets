module CascadeModel
export SimulationState, initSim, reinitSim!, step!, getActive, cascadeSizes

using Main.SupplyChains
using Graphs

using StatsBase: sample
using ProgressMeter

struct SimulationState{T<:Integer}
    sc::SupplyChain
    resilience::Matrix{T}    # number of inactive suppliers of each vertex on each layer
    cascade_front::Vector{T} # list of newly active vertices ("cascade front")
    new_front::Vector{T}     # sort of a front buffer - back buffer situation
end

function initSim(sc::SupplyChain, initial_state::BitVector)::SimulationState
    @assert length(sc.G) == sc.N_industries > 0
    @assert vertices(sc.G[1]) == 1:sc.N_vertices
    @assert length(initial_state) == sc.N_vertices

    T = Int64
    for t in [Int8, Int16, Int32]
        if sc.N_vertices ≤ typemax(t)
            T = t
            break
        end
    end

    resil = zeros(T, (sc.N_vertices, sc.N_industries))

    for i in 1:sc.N_industries
        for v in 1:sc.N_vertices
            if initial_state[v]
                resil[v,i] = -1
            else
                resil[v,i] = indegree(sc.G[i],v)
            end
        end
    end

    active = zeros(T,sc.N_vertices)
    na = zeros(T,sc.N_vertices)
    active[1:sum(initial_state)] .= (1:sc.N_vertices)[initial_state]

    #active= (1:sc.N_vertices)[initial_state]
    #na = Int32[]

    return SimulationState(sc, resil, active, na)
end

function reinitSim!(sim::SimulationState, initial_state::BitVector)::SimulationState
    @assert length(initial_state) == sim.sc.N_vertices

    for i in 1:sim.sc.N_industries
        for v in 1:sim.sc.N_vertices
            if initial_state[v]
                sim.resilience[v,i] = -1
            else
                sim.resilience[v,i] = indegree(sim.sc.G[i],v)
            end
        end
    end

    sim.cascade_front .= 0
    sim.new_front .= 0

    # This, but no alloc: sim.cascade_front[1:sum(initial_state)] .= (1:sim.sc.N_vertices)[initial_state]
    c=1
    for v in 1:sim.sc.N_vertices
        if initial_state[v]
            sim.cascade_front[c] = v
            c += 1
        end
    end

    return sim
end

function step!(sim::SimulationState)
    c = 1
    for v in sim.cascade_front
        if v == 0
            break
        end
        for i in 1:sim.sc.N_industries
            for n in outneighbors(sim.sc.G[i], v)
                sim.resilience[n,i] -= 1
                if sim.resilience[n,i] == 0
                    sim.resilience[n,:] .= -1
                    sim.new_front[c] = n
                    c += 1
                end
            end
        end
    end

    sim.new_front[c] = 0
    for i in 1:c
        @inbounds sim.cascade_front[i] = sim.new_front[i]
    end
end

function getActive(sim::SimulationState)::BitVector
    return sim.resilience[:,1] .< 0
end

function cascadeSizes(sc::SupplyChain, samples::Int64; showprogress=false)
    if samples ≥ sc.N_vertices
        samples = sc.N_vertices
    end

    sizes = zeros(Int64, samples)
    sample_verts = sample(1:sc.N_vertices, samples, replace=false)

    prog = Progress(samples; enabled = showprogress)
    sim = initSim(sc, falses(sc.N_vertices))
    for (s,v) in enumerate(sample_verts)
        reinitSim!(sim, (1:sc.N_vertices).==v)

        for _ in 1:10_000
            step!(sim)
        end

        sizes[s] = sum(getActive(sim))
        next!(prog)
    end
    return sizes
end

end # END MODULE