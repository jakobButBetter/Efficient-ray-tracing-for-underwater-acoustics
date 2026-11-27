function free_point_source_u(r::AbstractVector, z::AbstractVector, z_s::Float64, omega::Float64, c_0::Float64)
    ni, nj = length(r), length(z)
    u = zeros(ComplexF64, nj, ni)
    k_0 = omega / c_0
    for i in 1:ni
        for j in 1:nj
            R = sqrt(r[i]^2 + (z[j] - z_s)^2)
            u[j, i] = exp(im * k_0 * R) / (4π * R)
        end
    end
    return u
end

function sembounded_point_source_u(r::AbstractVector, z::AbstractVector, z_s::Float64, omega::Float64, c_0::Float64)
    ni, nj = length(r), length(z)
    u = zeros(ComplexF64, nj, ni)
    k_0 = omega / c_0
    for i in 1:ni
        for j in 1:nj
            R_1 = sqrt(r[i]^2 + (z[j] - z_s)^2)
            R_2 = sqrt(r[i]^2 + (z[j] + z_s)^2)
            u[j, i] = exp(im * k_0 * R_1) / (4π * R_1) - exp(im * k_0 * R_2) / (4π * R_2)
        end
    end
    return u
end

function plot_tl(
    r::AbstractVector,
    z::AbstractVector,
    TL::AbstractMatrix;
    clims::Tuple{Real, Real} = (30, 90),
    z_s::Union{Nothing, Real} = nothing,
    cmap = cgrad(:viridis, rev = true),
    savepath::Union{Nothing, AbstractString} = nothing,
)
    p = heatmap(
        r,
        z,
        TL;
        yflip = true,
        clims = clims,
        color = cmap,
        xlabel = "Range [m]",
        ylabel = "Depth [m]",
        colorbar_title = "TL [dB]",
        framestyle = :box,
        tickfontsize = 10,
        guidefontsize = 11,
        size = (650, 350),
        margin = 4Plots.mm,
    )

    if !isnothing(z_s)
        scatter!(p, [r[1]], [z_s]; color = :red, marker = :star5, markersize = 7, label = false)
    end

    if !isnothing(savepath)
        savefig(p, savepath)
    end

    return p
end


r, z = make_grid((200, 100), (10.0, 1.0))

p1 = free_point_source_u(r, z, 50.0, 2π*100.0, 1500.0) |>
    u -> plot_tl(r, z, -20 .* log10.(4π .* abs.(u)); z_s = 50.0)
display(p1)

p2 = sembounded_point_source_u(r, z, 50.0, 2π*100.0, 1500.0) |>
    u -> plot_tl(r, z, -20 .* log10.(4π .* abs.(u)); z_s = 50.0)
display(p2)