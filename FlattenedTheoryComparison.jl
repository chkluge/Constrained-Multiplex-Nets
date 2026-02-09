include("Main.jl")
include("FlattenedTheory.jl")

function figure_flattened(c, zmin, zmax)
    size_constr = Float64[]
    size_proj = Float64[]
    Zs = range(zmin,zmax, 500)
    
    @showprogress for z in Zs
        analytic = analytic_cascade_size_poisson(c,z, 1e-6, 10_000)
        push!(size_constr, analytic)
        analytic = analytic_flattened_size_poisson(c, z, 1e-6, 10_000)
        push!(size_proj, analytic)
    end

    f = Figure(size=(800,400))
    ax = Axis(f[1, 1], xlabel="z", ylabel="ρ", limits=(zmin,zmax,-.025,1.025))

    color0 = :black
    color1 = :red

    lines!(ax, Zs, size_constr,  label="Multiplex" , color = color0)
    lines!(ax, Zs, size_proj,  label="Flattened" , color = color1, linestyle=:dash)

    axislegend(ax, position=:rt)

    display(f)
    return f
end

function matrix1(p)
    m = ones(3,3)
    m[:,3] .= p
    return m
end
function matrix2()
    m = nested_region_matrix(3,3, 1.0,0.91)
    return m
end
function matrix3()
    m = nested_region_matrix(3,3, 0.94,1.0)
    return m
end
function matrix4()
    m = twoish_layer_matrix(20,20, 0.95,0.131)
    return m
end

# this function generates the figures in the appendix
function flattened_plots()
    f = figure_flattened(ones(3,3), 0.25, 1.75)
    save("projected_ones33.pdf", f)
    f = figure_flattened(matrix2(), 0.25, 1.75)
    save("projected_nestedA.pdf", f)
    f = figure_flattened(matrix3(), 0.25, 1.75)
    save("projected_nestedB.pdf", f)
    f = figure_flattened(matrix4(), 0.0, 5)
    save("projected_cusp.pdf", f)
    return nothing
end