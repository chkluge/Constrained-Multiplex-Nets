using Distributions

function B(n,p,x) # !almost! 1-(binomial CDF)
    s = 0.0
    bin = Binomial(n,p)
    for k in 0:(Int(ceil(x))-1)
        s += pdf(bin, k)
    end
    return 1.0 - s
end

function analytic_flattened_size_poisson(constraint_mat::AbstractMatrix, z::Real, ρ0::Real, steps::Integer = 5000)
    L = size(constraint_mat)[1]
    q     = Float64(ρ0)    # the initial value matters! this iteration can have multiple fixed points
    q_new = Float64(ρ0)

    CUTOFF = 100

    in_participation = vec(sum(constraint_mat, dims = 1))
    out_participation= vec(sum(constraint_mat, dims = 2))
    @assert(sum(in_participation) ≈ sum(out_participation), "in_participation = $in_participation, out_participation = $out_participation")

    for _ in 1:steps
        s = 0.0
        for α in 1:L
            if in_participation[α] == 0.0
                continue
            end
            factor = out_participation[α] / sum(out_participation)
            for k in 1:CUTOFF
                pois = Poisson(z*in_participation[α])
                s += factor * pdf(pois, k) * B(k, q, k / in_participation[α])
            end
        end
        q_new = ρ0 + (1-ρ0) * s

        if abs(q-q_new) < 1e-9
            break
        end
        q = q_new
    end

    s = 0.0
    for α in 1:L
        if in_participation[α] == 0.0
            continue
        end
        for k in 1:CUTOFF
            pois = Poisson(z*in_participation[α])
            s += pdf(pois, k) * B(k, q, k / in_participation[α])
        end
    end
    rho_final = ρ0 + (1-ρ0) * s / L

    return rho_final
end