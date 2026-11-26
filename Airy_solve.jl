using SpecialFunctions
using Roots

struct WaveguideParams
    c0::Float64       # Sound speed at z = 0 [m/s]
    g::Float64        # Sound speed gradient [1/s] (g != 0)
    z_floor::Float64  # Sea floor depth [m]
    z_s::Float64      # Source depth [m]
    f::Float64        # Frequency [Hz]
end

"""
Find all propagating mode eigenvalues (zeta0_m, kr_m) for the linear n^2 waveguide.
"""
function find_propagating_modes(p::WaveguideParams; n_grid::Int=5000)
    omega = 2pi * p.f
    L = (p.c0^3 / (2 * omega^2 * abs(p.g)))^(1/3)
    s = sign(p.g)
    deltazeta = s * p.z_floor / L

    # Dispersion relation D(zeta0) = 0
    D(zeta0) = airyai(zeta0) * airybiprime(zeta0 + delta_zeta) - airybi(zeta0) * airyaiprime(zeta0 + delta_zeta)

    # Propagating modes require k_r^2 > 0  =>  zeta0 > -L^2 * (omega/c0)^2
    # Oscillatory modes require min(zeta0, zetaf) < 0
    zeta0_min = -L^2 * (omega / p.c0)^2
    zeta0_max = s > 0 ? 0.0 : -deltazeta

    # Bracket sign changes on a dense grid, then refine with Brent's method
    grid = range(zeta0_min, zeta0_max, length=n_grid)
    vals = D.(grid)
    zeta0_roots = Float64[]

    for i in 1:(length(grid) - 1)
        if isfinite(vals[i]) && isfinite(vals[i+1]) && vals[i] * vals[i+1] < 0
            root = find_zero(D, (grid[i], grid[i+1]), Roots.Brent())
            push!(zeta0_roots, root)
        end
    end

    # Convert zeta0_m -> k_{r,m}
    kr_modes = [sqrt(max(0.0, zeta0 / L^2 + (omega / p.c0)^2)) for zeta0 in zeta0_roots]
    return zeta0_roots, kr_modes, L
end

"""
Evaluate the analytical pressure field u(r, z) on a 2D grid (r_vec, z_vec).
Returns a ComplexF64 matrix of size (length(z_vec), length(r_vec)).
"""
function analytical_pressure(p::WaveguideParams, r_vec, z_vec; n_grid::Int=5000)
    zeta0_roots, kr_modes, L = find_propagating_modes(p; n_grid=n_grid)
    s = sign(p.g)
    delta_zeta = s * p.z_floor / L

    u = zeros(ComplexF64, length(z_vec), length(r_vec))

    for (zeta0, kr) in zip(zeta0_roots, kr_modes)
        zetaf = zeta0 + delta_zeta
        ai0, bi0 = airyai(zeta0), airybi(zeta0)

        # Eigenmode function psi_m(z)
        psi(z) = bi0 * airyai(zeta0 + s * z / L) - ai0 * airybi(zeta0 + s * z / L)

        # Analytical normalization constant N_m (Eq. 2.32)
        psi_f = psi(p.z_floor)
        N_m = s * L * (zetaf * psi_f^2 + 1 / π^2)

        # Source and receiver mode amplitudes
        psi_s = psi(p.z_s)
        psi_z = psi.(z_vec)

        # Radial Hankel factor H_0^(1)(k_r * r)
        H_r = hankelh1.(0, kr .* r_vec)

        # Accumulate outer product into u(z, r)
        coeff = (im / 4) * (psi_s / N_m)
        @. u += coeff * psi_z * transpose(H_r)
    end
    
    return u
end