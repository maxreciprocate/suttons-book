using ReinforcementLearningAnIntroduction
using Statistics
using Plots
using LinearAlgebra
using StaticArrays

# ■ TD(n)
env = RandomWalkEnv(N=19, leftreward=-1)
na, nobs = length(get_action_space(env)), length(get_observation_space(env))

b = zeros(nobs-2); b[1] = -1/2; b[end] = 1/2
M = SymTridiagonal(zeros(nobs-2), ones(nobs-3) ./ 2)

# Mx + b = x
truthV = (M - I) \ -b

function walkTD_N(env; α = 0.1, n = 1)
    V = zeros(nobs)
    γ = 1

    for episode = 1:10
        reset!(env)
        T = Int(1e12)

        R = Float64[]
        S = Int[get_state(env)]

        for t = 1:100
            if t < T
                env(rand(1:2))

                s, terminal, r = observe(env)
                push!(S, s)
                push!(R, r)

                if terminal
                    T = t + 1
                end
            end

            # t for the next update of V
            τ = t - n

            if 1 <= τ < T
                # up to how far sum real rewards
                horizon = min(τ + n, T - 1)

                # in the case n = 3
                # G = (1, 2) (2, 3) (3, 4) (4, 5) + V(5)
                G = sum(R[idx] * γ^(idx - τ) for idx = τ:horizon)

                # add the last approximation
                if τ + n < T
                    G += γ^n * V[S[τ + n + 1]]
                end

                V[S[τ]] += α * (G - V[S[τ]])
            end
        end
    end

    @show V[2:end-1]
    sqrt(mean((V[2:end-1] .- truthV) .^ 2))
end

# ■ Figure 7.2

learning_rates = [0:0.025:1;]

plot()
for n = (2^i for i = 0:7)
    RMS_per_alpha = [mean(walkTD_N(env; α, n) for _ in 1:100) for α = learning_rates]

    plot!(learning_rates, RMS_per_alpha, label="n = $n")
end
plot!(size=(800, 600))

# ■ n-step tree backup

env = RandomWalkEnv(N=25, leftreward=-1)
na, nobs = length(get_action_space(env)), length(get_observation_space(env))

Q = rand(na, nobs) * 0.01
π = zeros(size(Q))
π[argmax(Q, dims=1)] .= 1

α = 0.4
γ = 0.9
n = 7

for episode = 1:20
    reset!(env)

    S = [get_state(env)]
    A = Int[rand(1:2)]
    R = Float64[]
    T = Int(1e12)

    for t = 1:200
        if t < T
            env(A[t])

            observation = observe(env)
            push!(S, observation.state)
            push!(R, observation.reward)

            if observation.terminal
                T = t + 1
            else
                push!(A, rand(1:2))
            end
        end

        τ = t - n

        if 1 <= τ < T
            G = if t >= T
                last(R)
            else
                R[t] + γ * π[:, S[t]] ⋅ Q[:, S[t]]
            end

            for time in min(T-2, t-1):-1:τ
                G = R[time] + γ * π[A[time], S[time]] * G

                for action = get_action_space(env)
                    action == A[time] && continue

                    G += γ * π[action, S[time]] * Q[action, S[time]]
                end
            end

            Q[A[τ], S[τ]] += α * (G - Q[A[τ], S[τ]])

            fill!(π, 0)
            π[argmax(Q, dims=1)] .= 1
        end
    end
end

(Q[:, end-8:end-1]) |> display
(Q[1, 2:end-1] .> Q[2, 2:end-1]) |> display

reset!(env)

state = get_state(env)

for t = 1:20
    action = argmax(π[:, state])
    env(action)
    observation = observe(env)

    if observation.terminal
        println(t, ' ', observation.reward)
        break
    end

    state = observation.state
end
