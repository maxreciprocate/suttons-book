using Plots
using ProgressMeter
ProgressMeter.ijulia_behavior(:clear)
using Flux
using Flux: glorot_uniform
using Flux.Optimise: update!
using LinearAlgebra
using ReinforcementLearningBase
using Statistics: mean
using CUDA
using Random
using Zygote
# ■

include("left-right.jl")

env = RandomWalkEnv(N = 50, leftreward=-1)
na, nobs = 2, 1

M = SymTridiagonal(zeros(env.N), ones(env.N) ./ 2)
b = zeros(env.N); b[1] = -1/2; b[end] = 1/2
truthV = (M - I) \ -b

# ■ offline λ-return

# naive way
function G_λ(t::Int, T::Int)
    G = 0

    for k = 1:T-t
        G_tk = sum(γ^p * rewards[t+p] for p = 0:k-1)

        if t+k <= T
            G_tk += γ^k * V(states[t+k])
        end

        G += λ^(k-1) * G_tk
    end

    (1 - λ) * G + λ^(T-t+1) * sum(γ^p * rewards[t+p] for p = 0:T-t)
end

initW = glorot_uniform
model = Chain(
    Dense(nobs, 32, relu; initW),
    Dense(32, 32, relu; initW),
    Dense(32, 1; initW)
) |> cpu

V(s) = model([s])[1]

γ, α = 0.99, 3e-4
ε = 0.112
λ = 0.8
frameskip = 1
opt = ADAM()
terminals = 0

maxtimestep = 5000 ÷ frameskip
states = zeros(Float64, nobs, maxtimestep)
actions = zeros(Int, maxtimestep)
rewards = zeros(Float64, maxtimestep)
maxepisodes = 1000
T = 0

@showprogress for episode = 1:maxepisodes
    reset!(env)
    T = 0
    for t = 1:maxtimestep
        s = copy(get_state(env))

        a = rand(1:na)
        env(a)

        states[1, t] = s
        actions[t] = a
        rewards[t] = get_reward(env)

        if get_terminal(env) || t == maxtimestep
            T = t
            break
        end
    end

    ## testing convergence of MC v. TD(1)
    # for ridx in T-1:-1:1
    #     rewards[ridx] += γ * rewards[ridx+1]
    # end

    ## naive G_λ
    # for idx in 1:T
    #     rewards[idx] = G_λ(idx, T)
    # end

    ## recursive G_λ
    for t in T-1:-1:1
        rewards[t] += λ * γ * rewards[t+1] + (1 - λ) * V(states[t+1])
    end

    G = Flux.unsqueeze(rewards[1:T], 1)
    S = @view states[:, 1:T]

    ∇ = gradient(params(model)) do
        mean((model(S) - G) .^ 2)
    end

    update!(opt, params(model), ∇)
end

outV = model(accumulate(+, ones(env.N))' ./ env.N)'
println.(outV);
sqrt(mean((outV .- truthV) .^ 2))

# ■ TD(λ)

function add!(z1::Zygote.Grads, z2::Zygote.Grads)
    for p in z1.params
        z1.grads[p] .+= z2.grads[p]
    end
end

import LinearAlgebra.mul!
function mul!(z::Zygote.Grads, a)
    for p in z.params
        z.grads[p] .*= a
    end
end

model = Chain(
    Dense(nobs, 32, relu; initW),
    Dense(32, 32, relu; initW),
    Dense(32, 1; initW)
) |> cpu

V(s) = model([s])[1]

γ, α = 0.9, 0.001
ε = 0.112
λ = 0.8
frameskip = 1
opt = ADAM()
terminals = 0

maxepisodes = 10000
maxtimestep = 500 ÷ frameskip

@showprogress for episode = 1:maxepisodes
    reset!(env)
    z = gradient(() -> V(get_state(env)), params(model))
    for p in z.params
        fill!(z.grads[p], 0)
    end

    for t = 1:maxtimestep
        s = get_state(env)

        a = rand(1:na)
        env(a)
        r = get_reward(env)
        s′ = get_state(env)

        ∇ = gradient(() -> V(s), params(model))
        mul!(z, γ * λ)
        add!(z, ∇)

        δ = r + γ * V(s′) - V(s)

        for w in params(model)
            w .+= α .* δ .* z.grads[w]
        end

        if get_terminal(env) || t == maxtimestep
            break
        end
    end
end

outV = model(accumulate(+, ones(env.N))' ./ env.N)'
println.(outV);
sqrt(mean((outV .- truthV) .^ 2))

# ■
