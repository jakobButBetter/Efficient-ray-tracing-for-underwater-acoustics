using StaticArrays
using ForwardDiff
using Interpolations
using DifferentialEquations
using Plots
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

seafloor = FlatBottom(200.0)
u0       = SVector(0.0, 0.0, 20.0, 1.0/1480.0, 0.0, 0.0)
tspan    = (100, 1.4e7)

linear_model = LinearSoundSpeed(1480.0, 0.016)
env_A        = Environment(linear_model, seafloor)

prob_A = ODEProblem(ray_rhs, u0, tspan, env_A) 
sol = solve(prob_A, Tsit5(); saveat = 2e4)
plot(sol[1, :], sol[2, :], sol[3, :], zflip=true, xlabel="X (m)", ylabel="Y (m)", zlabel="Depth Z (m)")