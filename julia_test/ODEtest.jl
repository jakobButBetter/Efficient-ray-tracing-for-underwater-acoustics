using DifferentialEquations
using Plots

u0 = [10.0, 5.0]
alpha = 1.5
beta = 1.0
delta = 1.0
gamma = 3.0
tspan = (0, 15)
p = (alpha, beta, delta, gamma)

function predator_prey!(du, u, p, t)
    
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = p[3] * u[1] * u[2] - p[4] * u[2]
end
prob = ODEProblem(predator_prey!, u0, tspan, p)
sol = solve(prob, Tsit5())
plot(sol)