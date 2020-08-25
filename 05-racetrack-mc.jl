include("05-racetrack.jl")
using ReinforcementLearningCore

softmax(xs) = exp.(xs) / sum(exp.(xs))

env = Racetrack()
# ■ arbitrary evaluation

Q = rand(env.griddims..., length(get_actions(env)))
C = zeros(size(Q))
π = rand(size(Q)...)

for state in CartesianIndices(env.griddims)
    π[state, :] = softmax(π[state, :])
end

random_policy(env::AbstractEnv) = rand(get_actions(env))
behave_policy = random_policy

for episode = 1:1e5
    trajectory = []

    reset!(env)

    for t = 1:20
        state = get_state(env)
        action = behave_policy(env)
        env(action)

        get_terminal(env) && break
        push!(trajectory, (state, action, env.reward))
    end

    G = 0
    W = 1
    γ = 1

    for (s, a, r) in reverse(trajectory)
        G = γ * G + r

        C[s..., a] += W
        Q[s..., a] += W / C[s..., a] * (G - Q[s..., a])
        # / b(a | s)
        W *= π[s..., a] / (1 / length(get_actions(env)))
    end
end

# ■ eval

for state in CartesianIndices(env.griddims)
    π[state, :] = softmax(Q[state, :])
end

reset!(env)

for t in 1:100
    state = get_state(env)
    action = argmax(π[state..., :])
    env(action)

    get_terminal(env) && break
end

println(env.reward)

# ■ proper offline

Q = rand(env.griddims..., length(get_actions(env)))
C = zeros(size(Q))
π = last.(Tuple.(argmax(Q, dims=3)))

behave_policy = random_policy

for episode = 1:1e5
    trajectory = []

    reset!(env)

    for t = 1:20
        state = get_state(env)
        action = behave_policy(env)
        env(action)

        get_terminal(env) && break
        push!(trajectory, (state, action, env.reward))
    end

    G = 0
    W = 1
    γ = 0.8

    for (s, a, r) in reverse(trajectory)
        G = γ * G + r

        C[s..., a] += W
        Q[s..., a] += W / C[s..., a] * (G - Q[s..., a])
        π[s...] = argmax(Q[s..., :])

        π[s...] != a && continue

        # cleary too elaborative
        W *= 1 / (1 / length(get_actions(env)))
    end
end

# ■ eval

reset!(env)

for t in 1:100
    state = get_state(env)
    action = π[state...]
    env(action)

    get_terminal(env) && break
end

env.reward
