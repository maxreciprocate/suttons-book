using ReinforcementLearningAnIntroduction
using Statistics
using BlackBoxOptim, Random
using Plots
using LinearAlgebra

# ■
env = CliffWalkingEnv()

γ = 0.9
α = 0.1
ε = 0.1

n_space, n_actions = length(get_observation_space(env)), length(get_action_space(env))
Q = rand(n_space, n_actions)

for episode = 1:1e3
    reset!(env)
    s = observe(env).state

    a = rand() < ε ? rand(1:n_actions) : argmax(Q[s, :])

    for t = 1:100
        env(a)

        obs = observe(env)
        # what do you get if you get lazy
        s_, r, terminal = obs.state, obs.reward, obs.terminal

        a_ = rand() < ε ? rand(1:n_actions) : argmax(Q[s_, :])

        # adhoc policy probabilities
        π = zeros(n_actions) .+ (ε / n_actions)
        π[argmax(Q[s_, :])] = 1 - ε * (n_actions - 1) / n_actions

        # sarsa update
        # Q[s, a] += α * (r + γ * Q[s_, a_] - Q[s, a])

        # Q-learning update
        Q[s, a] += α * (r + γ * maximum(Q[s_, :]) - Q[s, a])

        # expected sarsa update
        # Q[s, a] += α * (r + γ * dot(Q[s_, :], π) - Q[s, a])

        terminal && break

        s, a = s_, a_
    end
end

reset!(env)
s = observe(env).state
G = 0

for t = 1:100
    a = argmax(Q[s, :])
    env(a)
    obs = observe(env)
    s_, r, terminal = obs.state, obs.reward, obs.terminal
    G += r

    terminal && break

    s = s_
end

G
# ■ E6.4 compare MC, TD(0) for different α

env = RandomWalkEnv(leftreward=0)

function costMC(params)
    α = first(params)
    V = zeros(length(get_observation_space(env)))
    N = zeros(size(V))

    truthV = [i/6 for i = 1:5]

    γ = 1
    reset!(env)

    for episode = 1:1000
        trajectory = []
        reset!(env)
        s = env.state

        for t = 1:100
            a = rand(1:2)
            env(a)

            s_, terminal, r = observe(env)

            push!(trajectory, (s, a, r))
            terminal && break
            s = s_
        end

        G = 0
        for (s, a, r) in reverse(trajectory)
            G = γ * G + r
            # try out the straightforward update instead
            # V[s] = (V[s] * N[s] + G) / (N[s] + 1)
            # N[s] += 1

            V[s] += α * (G - V[s])
        end
    end

    sqrt(mean((V[2:end-1] .- truthV) .^2))
end

t = bboptimize(costMC, SearchRange=(0.0, 1.0), NumDimensions=1)
α = first(best_candidate(t))
mean(costMC(α) for i = 1:100)

# ■

function costTD(params)
    α = first(params)

    V = zeros(length(get_observation_space(env)))
    truthV = [i/6 for i = 1:5]

    γ = 1

    for episode = 1:100
        reset!(env)
        s = env.state

        for t = 1:100
            a = rand(1:2)
            env(a)

            s_, terminal, r = observe(env)

            V[s] += α * (r + γ * V[s_] - V[s])
            terminal && break
            s = s_
        end
    end

    sqrt(mean((V[2:end-1] .- truthV) .^2))
end

t = bboptimize(costTD, SearchRange=(0.0, 1.0), NumDimensions=1)
α = first(best_candidate(t))
mean(costTD(α) for i = 1:1000)
# ■ E6.5

V = zeros(length(get_observation_space(env))) .+ 1
truthV = [i/6 for i = 1:5]

α = 0.1
γ = 1

rms_errors = []

for episode = 1:10
    reset!(env)
    s = env.state

    for t = 1:100
        a = rand(1:2)
        env(a)

        s_, terminal, r = observe(env)

        V[s] += α * (r + γ * V[s_] - V[s])

        terminal && break
        s = s_
    end

    push!(rms_errors, sqrt(mean((V[2:end-1] .- truthV) .^2)))
end

plot(rms_errors)

# ■ double Q-learning

env = MaximizationBiasEnv()
n_space, n_actions = length(get_observation_space(env)), length(get_action_space(env))

γ = 0.9
α = 0.01
ε = 0.1

function maximizebias(; double=false)
    lefts_per_episode = zeros(10000, 500)

    for trial = 1:10000
        Q = zeros(n_space, n_actions)
        Q_2 = deepcopy(Q)

        for episode = 1:500
            reset!(env)
            s = observe(env).state

            actions = Int[]

            for t = 1:10
                if s == 1
                    if double
                        a = rand() < ε ? rand(1:2) : argmax(Q[s, 1:2] .+ Q_2[s, 1:2])
                    else
                        a = rand() < ε ? rand(1:2) : argmax(Q[s, 1:2])
                    end

                    # record actions from state A
                    push!(actions, a)
                else
                    if double
                        a = rand() < ε ? rand(1:n_actions) : argmax(Q[s, :] .+ Q_2[s, :])
                    else
                        a = rand() < ε ? rand(1:n_actions) : argmax(Q[s, :])
                    end
                end

                env(a)
                obs = observe(env)
                s_, r, terminal = obs.state, obs.reward, obs.terminal

                if double
                    if rand() < 0.5
                        Q[s, a] += α * (r + γ * Q_2[s_, argmax(Q[s_, :])] - Q[s, a])
                    else
                        Q_2[s, a] += α * (r + γ * Q[s_, argmax(Q_2[s_, :])] - Q_2[s, a])
                    end
                else
                    Q[s, a] += α * (r + γ * maximum(Q[s_, :]) - Q[s, a])
                end

                terminal && break
                s_ = s
            end

            lefts_per_episode[trial, episode] = count(isequal(1), actions) / length(actions)
        end
    end

    mean(lefts_per_episode, dims=1)'
end

plot(maximizebias(), label="q-learning")
plot!(maximizebias(double=true), label="double q")
plot!([0.05 for _ = 1:500], label="0.05 ε")
# ■

reset!(env)
s = observe(env).state
G = 0

for t = 1:100
    a = argmax(Q[s, :])
    println(a)
    env(a)

    obs = observe(env)
    s_, r, terminal = obs.state, obs.reward, obs.terminal
    G += r

    terminal && break

    s = s_
end

G
