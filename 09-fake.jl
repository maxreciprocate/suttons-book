using Plots
using ProgressMeter
using StaticArrays
using Flux
using Flux.Optimise: update!
using LinearAlgebra
using ReinforcementLearningBase
using ReinforcementLearningEnvironments: MountainCarEnv
# ■

env = MountainCarEnv(max_steps=Int(1e12))
na, nobs = length(get_actions(env)), 2

model = Chain(
    Dense(nobs, 16, relu),
    Dense(16, 16, relu),
    Dense(16, na)
)

println("#params = ", length.(params(model)) |> sum)

Q(s) = model(s)
Q(s, a) = Q(s)[a]

γ, α = 0.999, 4e-3
ε = 0.1
frameskip = 10
opt = RMSProp(α)
terminals = 0

@showprogress for episode = 1:50e3
    reset!(env)
    s = copy(get_state(env))

    T = 0
    δ = 0

    for t = 1:1000 ÷ frameskip
        a = rand() < ε ? rand(1:na) : argmax(Q(s))
        r = 0

        for _ = 1:frameskip
            env(a)
            get_terminal(env) && break

            r += get_reward(env)
        end

        if get_terminal(env)
            r += 100

            T = 1
            terminals += 1
        end

        a′ = rand() < ε ? rand(1:na) : argmax(Q(s))
        s′ = copy(get_state(env))

        δ = r + (1 - T) * γ * Q(s′, a′) - Q(s, a)
        ∇ = gradient(() -> -δ * Q(s, a), params(model))

        update!(opt, params(model), ∇)

        if T == 1
            break
        end

        s = s′
        a = a′
    end
end

@show terminals

# ■
reset!(env)

G = 0
for t = 1:500 ÷ frameskip
    s = copy(get_state(env))
    a = argmax(Q(s))

    for _ = 1:frameskip
        env(a)
        get_terminal(env) && break
        G += get_reward(env)
    end

    println(a, " ", round.(s; digits=3), " ", round.(Q(s); digits=3))

    if get_terminal(env)
        @show t
        break
    end
end

G
# ■
using DataStructures: CircularBuffer

println("#params = ", length.(params(model)) |> sum)

Q(s) = model(s)
Q(s, a) = Q(s)[a]

γ, α = 0.9, 4e-3
ε = 0.1
frameskip = 5
opt = Descent(α)
terminals = 0
n = 5

S = CircularBuffer{Vector{Float64}}(n+1)
A = CircularBuffer{Int64}(n)
R = CircularBuffer{Int64}(n)

@showprogress for episode = 1:50e3
    reset!(env)
    reset!(S)
    reset!(A)
    reset!(R)

    s = copy(get_state(env))
    push!(S, s)

    for t = 1:1000 ÷ frameskip
        a = rand() < ε ? rand(1:na) : argmax(Q(s))
        r = 0

        for _ = 1:frameskip
            env(a)
            r += get_reward(env)
        end

        s = copy(get_state(env))
        push!(S, s)
        push!(A, a)

        if get_terminal(env)
            push!(R, r + 100)
            terminals += 1

            break
        else
            push!(R, r)
        end

        if length(A) == n
            G = sum(R[idx] * γ^(idx - 1) for idx = 1:n)
            G += γ^n * maximum(Q(s))

            δ = G - Q(S[1], A[1])
            ∇ = gradient(() -> -δ * Q(S[1], A[1]), params(model))

            update!(opt, params(model), ∇)
        end
    end

    # eat away the chips
    for τ = 1:n
        G = sum(R[idx] * γ^(idx - 1) for idx = τ:n)
        δ = G - Q(S[τ], A[τ])

        ∇ = gradient(() -> -δ * Q(S[τ], A[τ]), params(model))
        update!(opt, params(model), ∇)
    end
end

@show terminals
# ■
reset!(env)

G = 0
for t = 1:500 ÷ frameskip
    s = get_state(env)
    a = argmax(Q(s))

    for _ = 1:frameskip
        env(a)
        G += get_reward(env)
    end

    println(a, " ", round.(s; digits=3), " ", round.(Q(s); digits=3))

    if get_terminal(env)
        @show t
        break
    end
end

G
