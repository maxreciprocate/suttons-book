# ■ {Sutton 2018} E2.5-10
using Statistics, Distributions, StatsPlots

# ■ k-bandit arms
function flesharms(howmany::Int; moving=false)
    elbows = [randn() for _ in 1:howmany]

    function pull(whicharm::Int)
        whicharm > howmany && return -Inf

        if moving
            # Move the mean of the arm
            elbows .+= rand(Normal(0, 0.01), howmany)
        end

        randn() + elbows[whicharm]
    end

    # elbows to track optimal action
    pull, Ref(elbows)
end

arms, _ = flesharms(10, moving=true)
violin([arms(i) for _ in 1:10^5, i in 1:10], leg=false, fillcolor=:grey, xlabel="arms")

# ■ actual gambler
function gambler(N=10, steps=1000; c=0, ε=0.1, α=0, optimism=0, introspect=false)
    arms, elbows = flesharms(N)

    # reward each step
    Rₜ = Vector{Float64}(undef, steps)
    # Q* -estimate
    Q = zeros(N) .+ optimism
    # was that an optimal action?
    Oₜ = zeros(Int, steps)
    # times each arm was pulled
    Nₜ = zeros(Int, N)

    for t in 1:steps
        if c > 0
            # upper bound on confidence choice
            a = argmax(Q .+ c * sqrt.(log(t) ./ Nₜ))
        else
            # ε-greedy choice
            a = rand() > ε ? argmax(Q) : rand(1:N)
        end

        # pulling the arm
        Rₜ[t] = r = arms(a)
        Nₜ[a] += 1

        # α is supplied factor's constant, otherwise it's the sample average
        factor = α > 0 ? α : 1 / Nₜ[a]

        Q[a] += factor * (r - Q[a])

        # That was actually an optimal action
        if a == argmax(elbows[])
            Oₜ[t] = 1
        end

        introspect && @show a r Q
    end

    Rₜ, Oₜ
end

# ■ E2.5 1)
meangambler(ε) = mean(cumsum(first(gambler(10, 10^4, ε=ε))) ./ collect(1:10^4) for _ in 1:2000)

plot(meangambler(0), label="epsilon = 0")
plot!(meangambler(0.01), label="epsilon = 0.01")
plot!(meangambler(0.1), label="epsilon = 0.1")

# ■ E2.5 2)
plot(meangambler(0.1), label="sample average")
plot!(mean(cumsum(first(gambler(10, 10^4, ε=0.1, α=0.1))) ./ collect(1:10^4) for _ in 1:2000), label="constant")

# ■ E2.6
gambler(10, 100, ε=0, optimism=5, introspect=true); nothing

# ■ gradient gambler
function sample(probs::Vector{Float64})
    U = rand()
    findfirst(x -> x > U || isnan(x), cumsum(probs))
end

function gradientgambler(N=10, steps=1000; α=1/4, introspect=false)
    arms, elbows = flesharms(N)

    # reward each step
    Rₜ = Vector{Float64}(undef, steps)
    # total rewards
    R̂ = 0.0
    # Y* -estimate
    Yₜ = zeros(N)
    # was that an optimal action?
    Oₜ = zeros(Int, steps)

    πₜ(a) = exp(Yₜ[a]) / sum(exp(γ) for γ in Yₜ)

    for t in 1:steps
        # sampling action
        probs = πₜ.(1:N)
        a = sample(probs)

        # pulling the arm
        R̂ += Rₜ[t] = r = arms(a)

        # updating preference estimates
        factor = α * (r - R̂ / t)
        Yₜ -= factor * probs
        Yₜ[a] += factor

        # optimal action
        if a == argmax(elbows[])
            Oₜ[t] = 1
        end
    end

    Rₜ, Oₜ
end

plot(mean(cumsum(last(gradientgambler(10, 10^4, α=1/4)))
          ./ collect(1:10^4) for _ in 1:10), label="gradient")

# ■ E2.11
L = 200_000
params = [1//128, 1//64, 1//32, 1//16, 1//8, 1//4, 1//2, 1, 2, 4, 6, 8]

macro bench(fn, param)
    # average reward from last 100000 steps over 100 runs over the parameter space
    :([mean(mean(first($fn(10, L, $param=ξ))[end-L÷2+1:end]) for _ in 1:50) for ξ in params])
end

plot(params, @bench(gambler, ε), label="greedy", xscale=:log, xticks=(params, map(string, params)), legend=:bottomleft)
plot!(params, @bench(gambler, α), label="greedy w/ constant")
plot!(params, @bench(gambler, c), label="confident")
plot!(params, @bench(gambler, optimism), label="optimistic")
plot!(params, @bench(gradientgambler, α), label="gradient")
