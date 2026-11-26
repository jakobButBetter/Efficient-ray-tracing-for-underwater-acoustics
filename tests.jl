include("ray_tracing.jl")



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
    g_waveguide(x) = min(x[3], env_A.topo.max_depth - x[3])
    cb_func = flat_reflection_callback(g_waveguide)
    u0      = launch(env_A, x0, theta, phi)
    tspan   = (0.0, 6.4)
    # tspan   = (0.0, 0.8)
    dt_save = 1e-2

    prob_A = ODEProblem(ray_rhs, u0, tspan, env_A; callback = cb_func)
    sol    = solve(prob_A, Tsit5(); saveat = dt_save, reltol = 1e-10, abstol = 1e-12)

    thetas = range(0.0, 0.05, length = 500)
    phis   = [0.0]

    angle_pairs = vec(collect(Iterators.product(thetas, phis)))
    n_rays      = length(angle_pairs)

    u0_base   = launch(env_A, x0, angle_pairs[1][1], angle_pairs[1][2])
    prob_base = ODEProblem(ray_rhs, u0_base, tspan, env_A; callback = cb_func)

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

    function exact_position(t, theta)
        alpha = 2 * atan( tanh((g * t / 2) - atanh(tan(theta / 2))) )

        r = rc + R * sin(alpha)
        return SVector(r * cos(phi), r * sin(phi), zc + R * cos(alpha))
    end

 
    # Absolute position error (m) at time t for a solve with relative tolerance tol
    function circle_err(tol, t)
        prob  = remake(prob_A; tspan = (0.0, t))
        s     = solve(prob, Tsit5(); reltol = tol, abstol = tol * 1e-2, save_everystep = false)
        x_num = SVector(s.u[end][1], s.u[end][2], s.u[end][3])
        return norm(x_num - exact_position(t, theta))
    end

    t_check = 0.5   # must be before the first bottom reflection (about 0.77 s)

    # Absolute position error (m) at the single time point t, for relative tolerance tol
    function circle_err(tol, t)
        s = solve(prob_A, Tsit5();
                  reltol = tol, abstol = tol * 1e-2,
                  tstops = [t], saveat = [t],
                  save_start = false, save_end = false)
        x_num = SVector(s.u[1][1], s.u[1][2], s.u[1][3])
        return norm(x_num - exact_position(t, theta))
    end

    println("1. position error at t = $t_check s (m): ", circle_err(1e-10, t_check))

    
    eik = [norm(u[4:6]) * sound_speed(linear_model, SVector(u[1], u[2], u[3])) - 1 for u in sol.u]
    println("2. max | |p|c - 1 |: ", maximum(abs, eik))


    # ------ 4. Plot Single Ray (p1) and Ensemble Ray Fan (p2) ------
    A = [dot(u[4:6], cross(u[7:9], u[10:12])) for u in sol.u]
    A_free = @. c_src * sol.t^2 * cos(theta)
    ratio  = abs.(A[2:end]) ./ A_free[2:end]

    p1 = plot(sol[1, 2:end], sol[3, 2:end];
            line_z = 10 .* log10.(ratio), lw = 3, yflip = true,
            xlabel = "x (m)", ylabel = "depth (m)",
            colorbar_title = "dB re free space", title = "Single Ray", legend = false, dpi=600)
    hline!(p1, [env_A.topo.max_depth], color = :black, lw = 2)
    savefig("ray.png")
    p2 = plot(xlabel = "x (m)", ylabel = "depth (m)",
            colorbar_title = "dB re free space", yflip = true,
            legend = false, title = "Ensemble Ray Fan ($n_rays rays)", dpi=600)
    
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
    savefig("rays.png")
    display(plot(p1, p2, layout = (2, 1), size = (800, 700)))


    println("3. tolerance sweep (t = $t_check s)")
    tols      = [1e-2, 1e-4, 1e-6, 1e-8]
    circ_errs = Float64[]
    for tol in tols
        err = circle_err(tol, t_check)
        push!(circ_errs, err)
        println("tol = $tol  position error (m) = ", err)
    end

    p3 = plot(tols, max.(circ_errs, eps());
            xscale = :log10, yscale = :log10,
            marker = :circle, lw = 2, label = "position error",
            xlabel = "reltol",
            ylabel = "position error at t = $t_check s (m)",
            title  = "Position error vs. solver tolerance",
            xflip  = true,
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