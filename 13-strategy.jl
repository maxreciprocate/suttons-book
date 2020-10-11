using Plots
using ProgressMeter
ProgressMeter.ijulia_behavior(:clear)
using Flux
using Flux: glorot_uniform
using Flux.Optimise: update!
using LinearAlgebra
using ReinforcementLearningBase
using ReinforcementLearningEnvironments: MountainCarEnv
using Statistics: mean
using CUDA
using Distributions: Categorical
using Random
using DataStructures: CircularBuffer
# ■

env = MountainCarEnv(max_steps=Int(1e12))
na, nobs = length(get_actions(env)), 2

function evaluate(model, frameskip=1)
    reset!(env)
    G = 0

    for t = 1:1000 ÷ frameskip
        s = copy(get_state(env))

        a = rand(Categorical(model(s) |> softmax))
        @show s, a

        for _ = 1:frameskip
            env(a)
            G += get_reward(env)
        end

        get_terminal(env) && break
    end

    G
end

# ■ REINFORCE

initW = glorot_uniform
model = Chain(
    Dense(nobs, 32, relu; initW),
    Dense(32, 32, relu; initW),
    Dense(32, na; initW)
)

println("#params = ", length.(params(model)) |> sum)

γ, α = 0.999, 3e-4
ε = 0.112
frameskip = 10
opt = ADAM()
terminals = 0

maxtimestep = 5000 ÷ frameskip
states = zeros(nobs, maxtimestep)
actions = zeros(Int, maxtimestep)
rewards = zeros(Float64, maxtimestep)
maxepisodes = 10e3

starttime = time()
@showprogress for episode = 1:maxepisodes
    reset!(env)
    T = 0

    for t = 1:maxtimestep
        s = copy(get_state(env))
        r = 0

        a = rand() < ε ? rand(1:na) : argmax(model(s) |> softmax)

        for _ = 1:frameskip
            env(a)
            r += get_reward(env)
        end

        if get_terminal(env)
            r += 100
            terminals += 1
        end

        states[:, t] = s
        actions[t] = a
        rewards[t] = r

        if get_terminal(env) || t == maxtimestep
            T = t
            break
        end
    end

    for idx in T-1:-1:1
        rewards[idx] += γ * rewards[idx+1]
    end

    S = @view states[:, 1:T]
    A = CartesianIndex.(view(actions, 1:T), 1:T)
    G = @view rewards[1:T]

    ∇ = gradient(params(model)) do
        logprob = (model(S) |> logsoftmax)[A]

        -mean(G .* logprob)
    end

    update!(opt, params(model), ∇)
end

@show time() - starttime, terminals / maxepisodes
mean(evaluate(model, 10) for _ = 1:10)
# ■ on GPU

initW = glorot_uniform

model = Chain(
    Dense(nobs, 32, relu; initW),
    Dense(32, 32, relu; initW),
    Dense(32, na; initW)
) |> cpu

modelg = model |> gpu

println("#params = ", length.(params(model)) |> sum)

γ, α = 0.999, 3e-4
ε = 0.112
frameskip = 10
opt = ADAM()

maxepisodes = 50e3
maxtimestep = 5000 ÷ frameskip
numtrajectories = 32

states = zeros(nobs, maxtimestep, numtrajectories)
actions = zeros(Int, maxtimestep, numtrajectories)
rewards = zeros(Float64, maxtimestep, numtrajectories)
terminals = zeros(Int, numtrajectories)

@showprogress for episode = 0:maxepisodes
    ntrajectory = Int(episode % numtrajectories) + 1
    reset!(env)

    for t = 1:maxtimestep
        s = copy(get_state(env))
        r = 0

        a = rand(Categorical(model(s) |> softmax))

        for _ = 1:frameskip
            env(a)
            r += get_reward(env)
        end

        if get_terminal(env)
            r += 100
        end

        states[:, t, ntrajectory] = s
        actions[t, ntrajectory] = a
        rewards[t, ntrajectory] = r

        if get_terminal(env) || t == maxtimestep
            terminals[ntrajectory] = t
            break
        end
    end

    for idx in terminals[ntrajectory]-1:-1:1
        rewards[idx, ntrajectory] += γ * rewards[idx+1, ntrajectory]
    end

    if ntrajectory == numtrajectories
        modelg = model |> gpu

        for idx in Random.shuffle(1:numtrajectories)
            T = terminals[idx]
            S = states[:, 1:T, idx] |> cu
            A = CartesianIndex.(actions[1:T, idx], 1:T)
            G = rewards[1:T, idx] |> cu

            ∇ = gradient(params(modelg)) do
                logprob = (modelg(S) |> logsoftmax)[A]

                -mean(G .* logprob)
            end

            update!(opt, params(modelg), ∇)
        end

        model = modelg |> cpu
    end
end

mean(evaluate(model, 10) for _ = 1:10)
# ■ With baseline

initW = glorot_uniform

model = Chain(
    Dense(nobs, 32, relu; initW),
    Dense(32, 32, relu; initW),
    Dense(32, na; initW)
) |> cpu

base = Chain(
    Dense(nobs, 32, relu; initW),
    Dense(32, 32, relu; initW),
    Dense(32, 1; initW)
) |> gpu

γ, α = 0.999, 3e-4
ε = 0.112
frameskip = 10
opt = ADAM()
opt_base = ADAM()

maxtimestep = 5000 ÷ frameskip
numtrajectories = 32
states = zeros(nobs, maxtimestep, numtrajectories)
actions = zeros(Int, maxtimestep, numtrajectories)
rewards = zeros(Float64, maxtimestep, numtrajectories)
terminals = zeros(Int, numtrajectories)
maxepisodes = 50e3

starttime = time()
@showprogress for episode = 0:maxepisodes
    ntrajectory = Int(episode % numtrajectories) + 1
    reset!(env)

    for t = 1:maxtimestep
        s = copy(get_state(env))
        r = 0

        a = rand(Categorical(model(s) |> softmax))

        for _ = 1:frameskip
            env(a)
            r += get_reward(env)
        end

        if get_terminal(env)
            r += 100
        end

        states[:, t, ntrajectory] = s
        actions[t, ntrajectory] = a
        rewards[t, ntrajectory] = r

        if get_terminal(env) || t == maxtimestep
            terminals[ntrajectory] = t
            break
        end
    end

    for idx in terminals[ntrajectory]-1:-1:1
        rewards[idx, ntrajectory] += γ * rewards[idx+1, ntrajectory]
    end

    modelg = model |> gpu

    if ntrajectory == numtrajectories
        for idx in Random.shuffle(1:numtrajectories)
            T = terminals[idx]
            S = states[:, 1:T, idx] |> cu
            A = CartesianIndex.(actions[1:T, idx], 1:T)
            G = Flux.unsqueeze(rewards[1:T, idx], 1) |> cu

            ∇_base = gradient(params(base)) do
                mean((G - base(S)) .^ 2)
            end

            update!(opt_base, params(base), ∇_base)

            ∇ = gradient(params(modelg)) do
                logprob = (modelg(S) |> logsoftmax)[A]

                -mean((G - base(S)) .* logprob)
            end

            update!(opt, params(modelg), ∇)
        end
    end

    model = modelg |> cpu
end

mean(evaluate(model, 10) for _ = 1:10)
# ■ With TD-δ baseline

initW = glorot_uniform

model = Chain(
    Dense(nobs, 32, relu; initW),
    Dense(32, 32, relu; initW),
    Dense(32, na; initW)
) |> cpu

base = Chain(
    Dense(nobs, 32, relu; initW),
    Dense(32, 32, relu; initW),
    Dense(32, 1; initW)
) |> gpu

γ = 0.99f0
frameskip = 10
opt = ADAM()
opt_base = ADAM()

replaysize = 512
states = zeros(Float32, nobs, replaysize)
nextstates = zeros(Float32, nobs, replaysize)
actions = zeros(Int, replaysize)
rewards = zeros(Float32, replaysize)
terminals = zeros(Float32, replaysize)

maxtimestep = 5000 ÷ frameskip
maxepisodes = 50e3

batchsize = 32
idx = 1
nterminals = 0

progressmeter = Progress(Int(maxepisodes / 1000))
episodelengths = []
for episode = 1:maxepisodes
    reset!(env)

    startepisode = idx
    T = 0
    for t = 1:maxtimestep
        s = copy(get_state(env))
        r = 0

        a = rand(Categorical(model(s) |> softmax))

        for _ = 1:frameskip
            env(a)
            r += get_reward(env)
        end

        if get_terminal(env)
            r += 100
            nterminals += 1
        end

        s′ = copy(get_state(env))

        states[:, idx] = s
        actions[idx] = a
        rewards[idx] = r
        nextstates[:, idx] = s′

        if get_terminal(env) || t == maxtimestep
            terminals[idx] = 1
            T = idx
            push!(episodelengths, t)
            break
        end

        s = s′
        idx += 1
    end

    for ridx in T-1:-1:startepisode
        rewards[ridx] += γ * rewards[ridx+1]
    end

    if idx + maxtimestep > replaysize
        modelg = model |> gpu

        for partition in Iterators.partition(Random.shuffle(1:idx), batchsize)
            # grained copies here are definitely not good
            S′ = nextstates[:, partition] |> cu
            S = states[:, partition] |> cu
            A = CartesianIndex.(actions[partition], 1:length(partition))
            T = Flux.unsqueeze(terminals[partition], 1) |> cu
            R = Flux.unsqueeze(rewards[partition], 1) |> cu

            G = R .+ (1 .- T) .* γ .* base(S′)

            gbase = gradient(params(base)) do
                mean((G - base(S)) .^ 2)
            end

            update!(opt_base, params(base), gbase)

            ∇ = gradient(params(modelg)) do
                logprob = (modelg(S) |> logsoftmax)[A]

                -mean((G - base(S)) .* logprob)
            end

            update!(opt, params(modelg), ∇)
        end

        model = modelg |> cpu

        idx = 1
        fill!(terminals, 0.0)
    end

    if episode % 1000 == 0
        next!(progressmeter; showvalues = [(:terminals, nterminals), (:episodelen, mean(episodelengths))])
        nterminals = 0
        empty!(episodelengths)
    end
end

@show maximum(evaluate(model, 10) for _ = 1:10)
device_reset!()
