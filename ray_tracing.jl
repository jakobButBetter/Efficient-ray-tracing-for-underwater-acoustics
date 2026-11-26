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

@inline function sound_speed(model::LinearSoundSpeed, x::SVector{3})
    z = x[3]
    return model.c0 + model.g * (z^1)
end

@inline function grad_c(model::AbstractSoundSpeed, x::SVector{3})
    return ForwardDiff.gradient(pos -> sound_speed(model, pos), x)
end

@inline function hess_c(model::AbstractSoundSpeed, x::SVector{3})
    return ForwardDiff.hessian(pos -> sound_speed(model, pos), x)
end

@inline function ray_rhs(u::SVector{20}, env, t)# inline it since it is going to be called a lot
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
    dp = -dc * c * norm(p)^2

    dq_theta = 2 * c * dot(dc, q_theta) * p + c^2 * P_theta
    dq_phi   = 2 * c * dot(dc, q_phi)   * p + c^2 * P_phi

    dP_theta = -(Hc * q_theta) / c + (dot(dc, q_theta) / c^2) * dc
    dP_phi   = -(Hc * q_phi)   / c + (dot(dc, q_phi)   / c^2) * dc
    return vcat(dx, dp, dq_theta, dq_phi, dP_theta, dP_phi, SVector(0.0, 0.0))
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
    return vcat(x0, p0, q_theta0, q_phi0, P_theta0, P_phi0, SVector(0.0, 0.0))
end

function build_ensemble(prob_base, x0, angles)
    pf = let x0 = x0, angles = angles
        (prob, ctx) -> begin
            th, ph = angles[ctx.sim_id]
            u0_i = launch(prob.p, x0, th, ph)
            remake(prob, u0 = u0_i)
        end
    end
    return EnsembleProblem(prob_base; prob_func = pf, safetycopy = false)
end

@inline function reflect_vec(v::SVector{3}, n::SVector{3})
    return v - 2.0 * dot(v, n) * n
end

@inline function reflect_perturbed_slowness(
    P::SVector{3},
    q::SVector{3},
    x_dot::SVector{3},
    n::SVector{3},
    c::Real,
    dc::SVector{3}
)
    dt_dalpha = -dot(n, q) / dot(n, x_dot)
    return reflect_vec(P, n) + (2.0 * (dot(n, dc) / c) * dt_dalpha) * n
end

function plane_reflection_callback_constructor(normals, offsets; loss_parameter = 1.0)
    N = [normalize(SVector{3}(n)) for n in normals]
    
    callbacks = map(zip(N, offsets)) do (n, offset)
        @inline condition(u, t, integrator) = dot(n, SVector(u[1], u[2], u[3])) - offset
        
        @inline function affect!(integrator)
            u       = integrator.u
            env     = integrator.p
            x       = SVector(u[1],  u[2],  u[3])
            p       = SVector(u[4],  u[5],  u[6])
            q_theta = SVector(u[7],  u[8],  u[9])
            q_phi   = SVector(u[10], u[11], u[12])
            P_theta = SVector(u[13], u[14], u[15])
            P_phi   = SVector(u[16], u[17], u[18])
            number_of_reflections = u[19]
            number_of_reflections += 1
            number_of_caustics = u[20]
            c     = sound_speed(env.ssp, x)
            dc    = grad_c(env.ssp, x)
            x_dot = (c^2) * p

            p_new       = reflect_vec(p, n)
            q_theta_new = reflect_vec(q_theta, n)
            q_phi_new   = reflect_vec(q_phi, n)
            P_theta_new = reflect_perturbed_slowness(P_theta, q_theta, x_dot, n, c, dc)
            P_phi_new   = reflect_perturbed_slowness(P_phi,   q_phi,   x_dot, n, c, dc)

            integrator.u = vcat(x, p_new, q_theta_new, q_phi_new, P_theta_new, P_phi_new, SVector(number_of_reflections, number_of_caustics))
        end
        
        ContinuousCallback(condition, affect!, nothing)
    end
    
    return CallbackSet(callbacks...)
end
function flat_reflection_callback(g; surface_tol = 1e-5)
    @inline condition(u, t, integrator) = g(SVector(u[1], u[2], u[3]))

    @inline function affect!(integrator)
        u       = integrator.u
        env     = integrator.p
        x       = SVector(u[1],  u[2],  u[3])
        p       = SVector(u[4],  u[5],  u[6])
        q_theta = SVector(u[7],  u[8],  u[9])
        q_phi   = SVector(u[10], u[11], u[12])
        P_theta = SVector(u[13], u[14], u[15])
        P_phi   = SVector(u[16], u[17], u[18])

        n_surface = abs(x[3]) < surface_tol ? u[19] + 1.0 : u[19]
        n_total   = u[20] + 1.0

        n     = normalize(ForwardDiff.gradient(g, x))
        c     = sound_speed(env.ssp, x)
        dc    = grad_c(env.ssp, x)
        x_dot = (c^2) * p

        p_new       = reflect_vec(p, n)
        q_theta_new = reflect_vec(q_theta, n)
        q_phi_new   = reflect_vec(q_phi, n)
        P_theta_new = reflect_perturbed_slowness(P_theta, q_theta, x_dot, n, c, dc)
        P_phi_new   = reflect_perturbed_slowness(P_phi,   q_phi,   x_dot, n, c, dc)

        integrator.u = vcat(x, p_new, q_theta_new, q_phi_new, P_theta_new, P_phi_new, n_surface, n_total)
    end

    return ContinuousCallback(condition, affect!, affect!)
end
function compute_phase_shifts(sol)
    n          = length(sol.u)
    n_surface  = Vector{Int}(undef, n)
    n_caustics = Vector{Int}(undef, n)
    phase      = Vector{Float64}(undef, n)

    prev_sign = 0.0
    c_count   = 0

    for k in 1:n
        u       = sol.u[k]
        n_surf  = round(Int, u[19])
        n_total = round(Int, u[20])

        if k >= 2  # skip point source at t = 0 where A == 0
            p       = SVector(u[4],  u[5],  u[6])
            q_theta = SVector(u[7],  u[8],  u[9])
            q_phi   = SVector(u[10], u[11], u[12])

            A_raw  = dot(p, cross(q_theta, q_phi))
            A_cont = isodd(n_total) ? -A_raw : A_raw

            s = sign(A_cont)
            if s != 0.0
                if prev_sign != 0.0 && s != prev_sign
                    c_count += 1
                end
                prev_sign = s
            end
        end

        n_surface[k]  = n_surf
        n_caustics[k] = c_count
        phase[k]      = n_surf * pi - c_count * (pi / 2) # e^(-iωt) convention
    end

    return (; n_surface, n_caustics, phase)
end
