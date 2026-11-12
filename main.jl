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

function ray_rhs(u::SVector{18}, env, t)# inline it since it is going to be called a lot
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

function reflect_vec(v::SVector{3}, n::SVector{3})
    v - 2.0 * dot(v,n) * n
end

function plane_reflection_callback_constructor(normals, offsets; loss_parameter = 1.0)
    N = [normalize(SVector{3}(n)) for n in normals]
    
    callbacks = map(zip(N, offsets)) do (n, offset)
        condition(u, t, integrator) = dot(n, SVector(u[1], u[2], u[3])) - offset
        
        function affect!(integrator)
            u       = integrator.u
            x       = SVector(u[1],  u[2],  u[3])
            p       = SVector(u[4],  u[5],  u[6])
            q_theta = SVector(u[7],  u[8],  u[9])
            q_phi   = SVector(u[10], u[11], u[12])
            P_theta = SVector(u[13], u[14], u[15])
            P_phi   = SVector(u[16], u[17], u[18])

            p_new = reflect_vec(p, n)
            integrator.u = vcat(x, p_new, q_theta, q_phi, P_theta, P_phi)
        end
        
        ContinuousCallback(condition, affect!, nothing)
    end
    
    return CallbackSet(callbacks...)
end

if abspath(PROGRAM_FILE) == @__FILE__() || isinteractive() #Only runs when this is main   
    seafloor     = FlatBottom(200.0)
    linear_model = LinearSoundSpeed(1480.0, 0.016)
    x0           = SVector(0.0, 0.0, 150.0)

    theta, phi = 0.05, 0.0

    env_A = Environment(linear_model, seafloor)
    c_src = sound_speed(env_A.ssp, x0)
    normals = [
        SVector(0.0, 0.0, -1.0), 
        SVector(0.0, 0.0,  1.0)  
    ]
    offsets = [0.0, env_A.topo.max_depth]
    
    cb = plane_reflection_callback_constructor(normals, offsets)
    u0      = launch(env_A, x0, theta, phi)
    tspan   = (0.0, 6.4)
    dt_save = 1e-2

    prob_A = ODEProblem(ray_rhs, u0, tspan, env_A; callback = nothing)
    sol    = solve(prob_A, Tsit5(); saveat = dt_save, reltol = 1e-10, abstol = 1e-12)

    thetas = range(0.0, 0.05, length = 5)
    phis   = [0.0]

    angle_pairs = vec(collect(Iterators.product(thetas, phis)))
    n_rays      = length(angle_pairs)

    u0_base   = launch(env_A, x0, angle_pairs[1][1], angle_pairs[1][2])
    prob_base = ODEProblem(ray_rhs, u0_base, tspan, env_A; callback = nothing)

    ens_prob = build_ensemble(prob_base, x0, angle_pairs)

    ens_sol = solve(
        ens_prob,
        Tsit5(),
        EnsembleThreads();
        trajectories = n_rays,
        saveat = dt_save,
        reltol = 1e-10,
        abstol = 1e-12
    )

    # ------  Single-Ray Solution Verifications ------
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


    # ------ 4. Plot Single Ray (p1) and Ensemble Ray Fan (p2) ------
    A = [dot(u[4:6], cross(u[7:9], u[10:12])) for u in sol.u]
    A_free = @. c_src * sol.t^2 * cos(theta)
    ratio  = abs.(A[2:end]) ./ A_free[2:end]

    p1 = plot(sol[1, 2:end], sol[3, 2:end];
            line_z = 10 .* log10.(ratio), lw = 3, yflip = true,
            xlabel = "x (m)", ylabel = "depth (m)",
            colorbar_title = "dB re free space", title = "Single Ray", legend = false)
    hline!(p1, [env_A.topo.max_depth], color = :black, lw = 2)

    p2 = plot(xlabel = "x (m)", ylabel = "depth (m)",
            colorbar_title = "dB re free space", yflip = true,
            legend = false, title = "Ensemble Ray Fan ($n_rays rays)")

    for (idx, sol_i) in enumerate(ens_sol.u)

        th, _ = angle_pairs[idx]
        A_i = [dot(u[4:6], cross(u[7:9], u[10:12])) for u in sol_i.u]
        A_free_i = @. c_src * sol_i.t^2 * cos(th)
        
        # Ensure lengths match before plotting
        min_len = min(length(A_i) - 1, length(A_free_i) - 1)

        ratio_i = abs.(A_i[2:end]) ./ A_free_i[2:end]

        plot!(p2, sol_i[1, 2:min_len+1], sol_i[3, 2:min_len+1];
            line_z = 10 .* log10.(ratio_i[1:min_len]), lw = 1.5)
    end
    hline!(p2, [env_A.topo.max_depth], color = :black, lw = 2)

    display(plot(p1, p2, layout = (2, 1), size = (800, 700)))


    println("5. tolerance sweep")
    tols       = [1e-6, 1e-8, 1e-10, 1e-12]
    circ_errs  = Float64[]
    for tol in tols
        s = solve(prob_A, Tsit5(); saveat = dt_save, reltol = tol, abstol = tol * 1e-2)
        err = circle_err(s)
        push!(circ_errs, err)
        println("tol = $tol  circle residual = ", err)
    end

    p3 = plot(tols, max.(circ_errs, eps());   # floor avoids zeros on a log axis
            xscale = :log10, yscale = :log10,
            marker = :circle, lw = 2, label = "circle residual",
            xlabel = "reltol",
            ylabel = "max relative circle error",
            title  = "Circle error vs. solver tolerance",
            xflip  = true,            # tighter tolerance to the right
            legend = :topright)

    # reference line: error proportional to tol (slope 1)
    plot!(p3, tols, circ_errs[1] .* (tols ./ tols[1]);
        ls = :dash, color = :gray, label = "proportional to tol")
    display(p3)

    # ------ Eikonal residual vs. ray parameter (time) ------
    n_of(u)     = 1 / sound_speed(linear_model, SVector(u[1], u[2], u[3]))   # n(x) = 1/c
    eik_abs(u)  = abs(norm(SVector(u[4], u[5], u[6])) - n_of(u))             # | |grad phi| - n |
    eik_rel(u)  = eik_abs(u) / n_of(u)                                       # = ||p|c - 1|

    res_abs = eik_abs.(sol.u)
    res_rel = eik_rel.(sol.u)

    p4 = plot(sol.t, max.(res_rel, eps());      # floor so zeros don't break the log axis
            yscale = :log10, lw = 2, label = "relative: ||p|c - 1|",
            xlabel = "ray parameter t (s)",
            ylabel = "eikonal residual",
            title  = "Eikonal residual: ||grad phi| - n(x)|",
            legend = :bottomright)
    plot!(p4, sol.t, max.(res_abs, eps());
        lw = 2, ls = :dash, label = "absolute: ||p| - n|")
    display(p4)


end