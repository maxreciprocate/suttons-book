using ReinforcementLearningAnIntroduction
using Statistics
using Plots
using StatsBase: sample
using LinearAlgebra
using StaticArrays
using ProgressMeter

function evaluate!(create_env::Function, Q, n = 10)
    env = create_env()
    evaluate!(env, Q, n)
end

function evaluate!(env::AbstractEnv, Q, n = 10)
    R = 0.

    for episode = 1:n
        reset!(env)
        state = observe(env).state

        for t = 1:1000
            env(argmax(Q[:, state]))
            obs = observe(env)

            R += obs.reward
            obs.terminal && break
            state = obs.state
        end
    end
    reset!(env)

    R / n
end


# ■ Dyna Q+ v Dyna Q
ε = 0.1
α = 0.1
γ = 0.9
# revisiting/exploration coefficient
κ = 0.01

n_updates = 100
n_trials, n_timesteps = 10, 20_000
timestep_of_change = 5000

function DynaQ(create_env::Function; κ = 0, change_env = false)
    env = create_env(change_env)
    na, nobs = length(get_action_space(env)), length(get_observation_space(env))

    rewards = zeros(n_timesteps, n_trials)

    @showprogress for trial = 1:n_trials
        Q = rand(na, nobs) * 0.01

        # S x A -> S' x R
        Model = Dict{Int, Dict{Int, Tuple{Int, Float64}}}()

        # the last timestep for which a particular action was chosen at the given state
        τ = zeros(Int, (na, nobs))

        reset!(env)
        s = observe(env).state

        for t = 1:n_timesteps
            action = if rand() < ε
                rand(1:na)
            else
                argmax(Q[:, s] .+ κ * sqrt.(t .- τ[:, s]))
            end

            env(action)
            obs = observe(env)

            Q[action, s] += α * (obs.reward + γ * maximum(Q[:, obs.state]) - Q[action, s])

            if !haskey(Model, s)
                Model[s] = Dict()
            end

            Model[s][action] = (obs.state, obs.reward)
            τ[action, s] = t

            for update in 1:n_updates
                s = rand(keys(Model))
                a = rand(keys(Model[s]))
                s_, r = Model[s][a]

                Q[a, s] += α * (r + γ * maximum(Q[:, s_]) - Q[a, s])
            end

            rewards[t, trial] = evaluate!(create_env, Q)

            if obs.terminal
                # force the agent to reexplore
                if change_env && t > timestep_of_change
                    env = create_env()
                end

                reset!(env)
                s = observe(env).state
            else
                s = obs.state
            end
        end
    end

    # return an average cummulative reward over all trials
    mean(mapslices(xs -> accumulate(+, xs), rewards, dims=1), dims=2)
end

# ■
function random_maze(change_env = false)
    if change_env
        MazeEnv() * rand(2:5)
    else
        MazeEnv()
    end
end

plot(DynaQ(random_maze; change_env = true, κ), label="Q+")
plot!(DynaQ(random_maze, change_env = true), label="Q-")

# ■ Trajectory sampling

function evaluate7(M, Q)
    s = 1
    R = 0.

    while true
        a = argmax(Q[:, s])

        future = filter(x -> x[1] != 0, M[:, a, s])
        isempty(future) && break

        s, r  = rand(future)
        R += r

        rand() < 0.1 && break
    end

    R
end

ns = 10000
na = 2
nb = 3

env = BranchMDPEnv(ns, na, nb)
# ■

function trajectory_sampling(sampling=:trajectory)
    Q = rand(na, ns) * 0.1
    # M : S x A -> S', R
    M = Array{Tuple{Int, Float64}, 3}(undef, nb, na, ns)
    fill!(M, (0, 0.))

    ε = 0.1
    γ = 1
    α = 0.1

    reset!(env)
    s = observe(env).state
    V = Float64[]

    n_compute_steps = 200_000

    while n_compute_steps > 0
        a = rand() < ε ? rand(1:na) : argmax(Q[:, s])

        env(a)
        obs = observe(env)

        if obs.terminal
            reset!(env)
            s = observe(env).state
            continue
        end

        for branch in 1:nb
            # already stored transition
            M[branch, a, s][1] == obs.state && break

            if M[branch, a, s][1] == 0
                M[branch, a, s] = (obs.state, obs.reward)
                break
            end
        end

        Q[a, s] += α * (obs.reward + γ * maximum(Q[:, s]) - Q[a, s])

        if sampling == :none
            n_compute_steps -= 1
            if n_compute_steps % 2_000 == 0
                push!(V, mean(evaluate7(M, Q) for _ = 1:100))
            end
        end

        if sampling == :uniform
            for s in 1:ns, a in 1:na
                future = filter(x -> x[1] != 0, M[:, a, s])
                # einstein for today
                isempty(future) && continue

                p = 1 / length(future)

                EQ = sum(p * (r + maximum(Q[:, s′])) for (s′, r) in future)
                Q[a, s] += α * (EQ - Q[a, s])

                n_compute_steps -= 1
                if n_compute_steps % 2_000 == 0
                    push!(V, mean(evaluate7(M, Q) for _ = 1:100))
                end
            end
        end

        if sampling == :trajectory
            ŝ = 1

            # counteracting uniform sampling's limited interaction with the environment
            for playtime = 1:20
                â = rand() < 0.1 ? rand(1:na) : argmax(Q[:, ŝ])
                # â = argmax(Q[:, ŝ])

                future = filter(x -> x[1] != 0, M[:, â, ŝ])

                if isempty(future)
                    ŝ = 1
                    continue
                end

                ŝ′, r̂ = rand(future)
                Q[â, ŝ] += α * (r̂ + maximum(Q[:, ŝ′]) - Q[â, ŝ])

                n_compute_steps -= 1
                n_compute_steps % 2_000 == 0 && push!(V, mean(evaluate7(M, Q) for _ = 1:100))

                ŝ = rand() < 0.1 ? 1 : ŝ′
            end
        end

        s = obs.state
    end

    V
end
# ■

trajectored, uniformed, noned = [], [], []

@showprogress for _ = 1:10
    push!(trajectored, trajectory_sampling(:trajectory))
    push!(uniformed, trajectory_sampling(:uniform))
    push!(noned, trajectory_sampling(:none))
end
# ■

# in the order of the least interactions with the environment
plot(mean(uniformed), label="uniform")
plot!(mean(trajectored), label="trajectory")
plot!(mean(noned), label="none")
