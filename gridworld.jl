using Test, Random, Plots

function littleshufle(zs::AbstractVector)
    if length(zs) > 9
        error("'tis a jest")
    end

    y = BigInt(10)^rand(15:19)

    idx = digits(round(Int, π * y % y)) |> inspect |>
        xs -> filter(x -> x > 0, xs) |>
        xs -> map(x -> x > length(zs) ? 10 - x : x, xs) |> unique

    zs[idx]
end

function sampler(probs, what)
    pdf = cumsum(probs)

    return function sample()
        U = rand()
        idx = findfirst(x -> x > U || isnan(x), pdf)

        what[idx]
    end
end

function pp(d::Dict)
    for (k, v) in d
        v == nothing && continue

        println("$k -> $v")
    end
end

# ■ Value iteration with Q convergence
N = 15
kernel = [[0, 1], [0, -1], [1, 0], [-1, 0]]
block  = [[i, j] for i in 1:N, j in 1:N]

A = map(x -> filter(c -> 1 <= c[1] <= N && 1 <= c[2] <= N, map(cell -> cell + x, kernel)), block)
A[N, N] = []
A[1, 1] = []

V = zeros(Float64, N, N)
P = map(x -> if isempty(x) nothing else rand(x) end, A)
Q = map(x -> zeros(length(x)), A)
γ = 0.9

while true
    Δ = 1
    while Δ > 0.1
        Δ = 0

        for i in CartesianIndices(A)
            isempty(A[i]) && continue

            _V = V[i]

            V[i] = -1.0 + γ * V[P[i]...]

            Δ = max(Δ, abs(_V - V[i]))
        end

    end

    _Q = deepcopy(Q)

    for i in CartesianIndices(A)
        isempty(A[i]) && continue

        Q[i] = map(s -> -1.0 + γ * V[s...], A[i])

        P[i] = A[i][argmax(Q[i])]
    end

    # approx
    _Q == Q && break

    @show maximum(Q - _Q)
end

display(P)
display(V)
display(Q)

heatmap(V, c=:matter, yflip=true, aspect_ratio=1)

# ■ Q iteration
N = 15
kernel = [[0, 1], [0, -1], [1, 0], [-1, 0]]
block  = [[i, j] for i in 1:N, j in 1:N]

A = map(x -> filter(c -> 1 <= c[1] <= N && 1 <= c[2] <= N, map(cell -> cell + x, kernel)), block)
A[N, N] = []
A[1, 1] = []

P = map(x -> if isempty(x) nothing else rand(x) end, A)
Q = map(x -> zeros(length(x)), A)
γ = 0.9

map(s_ -> -1.0 + γ * maximum(Q[s_...]), A[3, 3])

while true
    Δ = 1
    while Δ > 0.1
        Δ = 0

        for i in CartesianIndices(A)
            if isempty(A[i]) continue end

            _Q = deepcopy(Q)
            Q[i] = map(s_ -> -1.0 + γ * maximum(Q[s_...]), A[i])

            V[i] = -1.0 + γ * V[P[i]...]

            Δ = max(Δ, abs(_V - V[i]))
        end

    end

    _Q = deepcopy(Q)

    for i in CartesianIndices(A)
        if isempty(A[i]) continue end

        Q[i] = map(s -> -1.0 + γ * V[s...], A[i])

        P[i] = A[i][argmax(Q[i])]
    end

    _Q == Q && break

    @show maximum(Q - _Q)
end

display(P)
display(V)
display(Q)

heatmap(V, c=:matter, yflip=true, aspect_ratio=1)
