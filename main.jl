using StaticArrays
using ForwardDiff
using Interpolations
using DifferentialEquations
using Plots
using LinearAlgebra
abstract type AbstractSoundSpeed end
abstract type AbstractTopography end

struct LinearSoundSpeed <: AbstractSoundSpeed
    c0::Float64
    g::Float64
end

struct FlatBottom <: AbstractTopography
    max_depth::Float64
end

struct Environment{S <: AbstractSoundSpeed, T <: AbstractTopography}
    ssp::S
    topo::T
end


function sound_speed(model::LinearSoundSpeed, x::SVector{3})
    z = x[3]
    return model.c0 + model.g * z
end


function grad_c(model::AbstractSoundSpeed, x::SVector{3})
    return ForwardDiff.gradient(pos -> sound_speed(model, pos), x)
end

function hess_c(model::AbstractSoundSpeed, x::SVector{3})
    return ForwardDiff.hessian(pos -> sound_speed(model, pos), x)
end


struct GriddedSoundSpeed{I} <: AbstractSoundSpeed
    itp::I
end

function GriddedSoundSpeed(xs::AbstractRange, ys::AbstractRange, zs::AbstractRange, C::Array{Float64, 3})
    raw_itp = interpolate(C, BSpline(Cubic(Line(OnGrid()))))
    scaled_itp = extrapolate(scale(raw_itp, xs, ys, zs), Flat())
    
    return GriddedSoundSpeed(scaled_itp)
end

function sound_speed(model::GriddedSoundSpeed, x::SVector{3})
    return model.itp(x[1], x[2], x[3])
end
function slowness_force(model::AbstractSoundSpeed, x::SVector{3})
    c_val = sound_speed(model, x)
    dc    = grad_c(model, x)
    return -(1.0 / c_val^3) * dc
end
function ray_rhs(u, env, tau)
    x = SVector(u[1], u[2], u[3]) 
    p = SVector(u[4], u[5], u[6]) 
    q_theta = SVector(u[7], u[8], u[9]) 
    q_phi = SVector(u[10], u[11], u[12])
    P_theta = SVector(u[13], u[14], u[15])
    P_phi = SVector(u[16], u[17], u[18])
    c_val = sound_speed(env.ssp, x)
    dc    = grad_c(env.ssp, x)
    
    c3 = c_val^3
    dx = p
    dp = slowness_force(env.ssp, x)
    dq_theta = P_theta
    dq_phi= P_phi
    J_F = ForwardDiff.jacobian(pos -> slowness_force(env.ssp, pos), x)
    dP_theta = J_F * q_theta
    dP_phi  = J_F * q_phi
    return vcat(dx, dp, dq_theta, dq_phi, dP_theta, dP_phi)  
end

function launch(env, x0, theta, phi)
    c_src = sound_speed(env.ssp, x0)
    t_hat(a, b) = SVector(cos(a) * cos(b) , cos(a)*sin(b), sin(a))
    p0    = t_hat(theta, phi) / c_src
    dp_th = ForwardDiff.derivative(a -> t_hat(a, phi), theta) / c_src
    dp_ph = ForwardDiff.derivative(b -> t_hat(theta, b), phi) / c_src
    return SVector{18}(vcat(x0, p0, zeros(3), zeros(3), dp_th, dp_ph))
end

seafloor = FlatBottom(200.0)

linear_model = LinearSoundSpeed(1480.0, 0.016)
x0 = SVector(0.0, 0.0, 20.0)

theta, phi = 0.05, 0.0

env_A = Environment(linear_model, seafloor)
u0 = launch(env_A, x0, theta, phi)

tspan = (0, 1.4e7)

prob_A = ODEProblem(ray_rhs, u0, tspan, env_A) 



sol = solve(prob_A, Tsit5(); saveat = 2e4, reltol = 1e-10, abstol = 1e-12)


# ------ Solution verifications ------
# Exact solution 
g  = linear_model.g
R  = c_src / (g * cos(theta))          # circle radius
rc = R * sin(theta)                    # centre, horizontal range from source
zc = x0[3] - R * cos(theta)            # centre depth (equals -c0/g)

# horizontal range along the launch azimuth, and depth
rng(s) = s[1, :] .* cos(phi) .+ s[2, :] .* sin(phi)

function circle_err(s)
    r, z = rng(s), s[3, :]
    return maximum(abs.(sqrt.((r .- rc).^2 .+ (z .- zc).^2) .- R)) / R
end
println("1. circle residual (relative): ", circle_err(sol))


ph = sol[4, :] .* cos(phi) .+ sol[5, :] .* sin(phi)
pp = -sol[4, :] .* sin(phi) .+ sol[5, :] .* cos(phi)
println("2. horizontal slowness drift: ", maximum(abs.(ph .- ph[1])) / abs(ph[1]))
println("   sideways slowness:         ", maximum(abs.(pp)))


eik = [norm(u[4:6]) * sound_speed(linear_model, SVector(u[1], u[2], u[3])) - 1 for u in sol.u]
println("3. max | |p|c - 1 |: ", maximum(abs, eik))


# Only valid when the ray does not reach the bottom
z    = sol[3, :]
i    = argmax(z)
println("4. turning depth: sim = ", z[i], ", exact = ", zc + R)
println("   turning range: sim = ", rng(sol)[i], ", exact = ", rc)

println("5. tolerance sweep")
for tol in (1e-6, 1e-8, 1e-10, 1e-12)
    s = solve(prob_A, Tsit5(); saveat = 2e4, reltol = tol, abstol = tol * 1e-2)
    println("tol = $tol  circle residual = ", circle_err(s))
end

# Plot final solution
# plot(sol[1, :], sol[2, :], sol[3, :], zflip=true, xlabel="X (m)", ylabel="Y (m)", zlabel="Depth Z (m)")

# Plot A 
A = [dot(u[4:6], cross(u[7:9], u[10:12])) for u in sol.u]
A_free = @. sol.t^2 * cos(theta) / c_src^3      # same tau, homogeneous medium
ratio  = abs.(A[2:end]) ./ A_free[2:end]
tau    = sol.t[2:end]

plot(sol[1, 2:end], sol[3, 2:end];
     line_z = 10 .* log10.(ratio), lw = 3, yflip = true,
     xlabel = "x (m)", ylabel = "depth (m)", colorbar_title = "dB re free space")
hline!([env_A.topo.max_depth], color = :black)

