module AnalyticResults
export analytic_cascade_size, analytic_cascade_size_poisson, analytic_cascade_size_poisson_itercount, critical_z_poisson, analytic_cascade_prob_poisson

using Distributions: pdf, DiscreteDistribution, mean, Poisson
using LinearAlgebra: eigvals
using ForwardDiff: derivative

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

end # END MODULE