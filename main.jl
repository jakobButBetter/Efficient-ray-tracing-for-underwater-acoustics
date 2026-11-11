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

function ray_rhs(u::SVector{18}, env, t)
    x       = SVector(u[1],  u[2],  u[3])
    p       = SVector(u[4],  u[5],  u[6])
    q_theta = SVector(u[7],  u[8],  u[9])
    q_phi   = SVector(u[10], u[11], u[12])
    P_theta = SVector(u[13], u[14], u[15])
    P_phi   = SVector(u[16], u[17], u[18])

    c  = sound_speed(env.ssp, x)
    dc = grad_c(env.ssp, x)
    Hc = hess_c(env.ssp, x)

    dx = c^2 * p
    dp = -dc / c

    dq_theta = 2 * c * dot(dc, q_theta) * p + c^2 * P_theta
    dq_phi   = 2 * c * dot(dc, q_phi)   * p + c^2 * P_phi

    dP_theta = -(Hc * q_theta) / c + (dot(dc, q_theta) / c^2) * dc
    dP_phi   = -(Hc * q_phi)   / c + (dot(dc, q_phi)   / c^2) * dc

    return vcat(dx, dp, dq_theta, dq_phi, dP_theta, dP_phi)
end

function launch(env, x0, theta, phi)
    c_src = sound_speed(env.ssp, x0)

    t_hat    = SVector(cos(theta) * cos(phi), cos(theta) * sin(phi), sin(theta))
    dt_theta = SVector(-sin(theta) * cos(phi), -sin(theta) * sin(phi), cos(theta))
    dt_phi   = SVector(-cos(theta) * sin(phi),  cos(theta) * cos(phi), 0.0)

    p0       = t_hat / c_src
    q_theta0 = SVector(0.0, 0.0, 0.0)
    q_phi0   = SVector(0.0, 0.0, 0.0)
    P_theta0 = dt_theta / c_src
    P_phi0   = dt_phi / c_src

    return vcat(x0, p0, q_theta0, q_phi0, P_theta0, P_phi0)
end

seafloor = FlatBottom(200.0)
linear_model = LinearSoundSpeed(1480.0, 0.016)
x0 = SVector(0.0, 0.0, 20.0)

theta, phi = 0.05, 0.0

env_A = Environment(linear_model, seafloor)
c_src = sound_speed(env_A.ssp, x0)

u0 = launch(env_A, x0, theta, phi)

tspan   = (0.0, 6.4)
dt_save = 1e-2

prob_A = ODEProblem(ray_rhs, u0, tspan, env_A)
sol = solve(prob_A, Tsit5(); saveat = dt_save, reltol = 1e-10, abstol = 1e-12)

# ------ Solution verifications ------
g  = linear_model.g
R  = c_src / (g * cos(theta))
rc = R * sin(theta)
zc = x0[3] - R * cos(theta)

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

z = sol[3, :]
i = argmax(z)
println("4. turning depth: sim = ", z[i], ", exact = ", zc + R)
println("   turning range: sim = ", rng(sol)[i], ", exact = ", rc)

println("5. tolerance sweep")
for tol in (1e-6, 1e-8, 1e-10, 1e-12)
    s = solve(prob_A, Tsit5(); saveat = dt_save, reltol = tol, abstol = tol * 1e-2)
    println("tol = $tol  circle residual = ", circle_err(s))
end

# Plot A (u[4:6] = p, u[7:9] = q_theta, u[10:12] = q_phi)
A = [dot(u[4:6], cross(u[7:9], u[10:12])) for u in sol.u]
A_free = @. c_src * sol.t^2 * cos(theta)
ratio  = abs.(A[2:end]) ./ A_free[2:end]

plot(sol[1, 2:end], sol[3, 2:end];
     line_z = 10 .* log10.(ratio), lw = 3, yflip = true,
     xlabel = "x (m)", ylabel = "depth (m)", colorbar_title = "dB re free space")
hline!([env_A.topo.max_depth], color = :black)