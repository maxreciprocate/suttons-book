using Test, Random, Plots

function sampler(probs, xs)
    cdf = cumsum(probs)

    return function sample()
        U = rand()
        idx = findfirst(x -> x > U || isnan(x), cdf)

        xs[idx]
    end
end

# ■ Value iteration with Q-convergence
N = 48
kernel = [[0, 1], [0, -1], [1, 0], [-1, 0]]
block  = [[i, j] for i in 1:N, j in 1:N]

# state actions
A = map(x -> filter(c -> 1 <= c[1] <= N && 1 <= c[2] <= N, map(cell -> cell + x, kernel)), block)
# ways out
A[N ÷ 2 + 1, N ÷ 2 + 1] = A[1, 1] = A[N, N] = A[1, N] = A[N, 1] = []
# random policy
P = map(x -> if isempty(x) nothing else rand(x) end, A)
# empty values
V = zeros(Float64, N, N)
Q = map(x -> zeros(length(x)), A)
# discount
γ = .97

while true
    Δ = 1
    while Δ > 0.1
        Δ = 0

        for i in CartesianIndices(A)
            isempty(A[i]) && continue

            _V = V[i]

            V[i] = -1 + γ * V[P[i]...]

            Δ = max(Δ, abs(_V - V[i]))
        end

    end

    # check convergence
    _Q = deepcopy(Q)

    for i in CartesianIndices(A)
        isempty(A[i]) && continue

        # recompute Q-values
        Q[i] = map(s -> -1.0 + γ * V[s...], A[i])

        # update policy
        P[i] = A[i][argmax(Q[i])]
    end

    # approx
    _Q == Q && break

    @show maximum(Q - _Q)
end

display(P)
display(Q)
display(V)

heatmap(V, c=:matter, yflip=true, aspect_ratio=1)
