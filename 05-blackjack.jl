using Plots

# ■ Solving blackjack

mutable struct Hand
    cards::Vector{Int}
    aces::Int

    Hand() = new([], 0)
end

function hit!(hand::Hand)
    card = rand([2:10..., 10, 10, 10, 11])

    if card == 11
        hand.aces += 1
    end

    push!(hand.cards, card)
end

function getscore(hand::Hand)::Int
    s = sum(hand.cards)

    aces = hand.aces

    # using soft aces
    while s > 21 && aces > 0
        s -= 10
        aces -= 1
    end

    s
end

function savestep!(trajectory::Vector{Any}, hand::Hand, dealer::Hand, action=0)
    state = (getscore(hand) - 1, Int(hand.aces > 0) + 1, first(dealer.cards) - 1)

    if action != 0
        push!(trajectory, (state, action))
    else
        push!(trajectory, state)
    end
end

function improve!(Q::Array{Float64, 4}, N::Array{Int64, 4}, trajectory::Vector{Any}; reward::Float64, γ=1)
    G = reward

    for (state, action) in reverse(trajectory)
        Q[state..., action] = (Q[state..., action] * N[state..., action] + G) / (N[state..., action] + 1)
        N[state..., action] += 1

        G = γ * G
    end
end

function improve!(V::Array{Float64, 3}, N::Array{Int64, 3}, trajectory::Vector{Any}; reward::Float64, γ=1)
    for state in reverse(trajectory)
        V[state...] = (V[state...] * N[state...] + reward) / (N[state...] + 1)
        N[state...] += 1
    end
end


# ■ 5.1 MC state-value function for the simple policy

V = zeros(20, 2, 10)
N = zeros(Int, size(V)...)

for episode = 1:1e6
    trajectory = []

    # seeding starting state
    dealer = Hand()
    hit!(dealer); hit!(dealer)

    hand = Hand()
    hit!(hand)

    # evaluating policy "hit until the score is bigger than 19"
    while getscore(hand) < 19
        hit!(hand)

        savestep!(trajectory, hand, dealer)
    end

    if getscore(hand) > 21
        # we've already lost at this point
        pop!(trajectory)

        improve!(V, N, trajectory, reward=-1.0)
        continue
    end

    if getscore(hand) == 21 && getscore(dealer) != 21
        improve!(V, N, trajectory, reward=1.0)
        continue
    end

    # dealer's policy is "hit until the score is bigger than 17"
    while getscore(dealer) < 17
        hit!(dealer)
    end

    reward = if getscore(hand) == getscore(dealer)
        0.0
    elseif getscore(hand) > getscore(dealer) || getscore(dealer) > 21
        1.0
    else
        -1.0
    end

    improve!(V, N, trajectory; reward)
end

access(dealercard, score) = V[score, 1, dealercard]
plot(1:10, 11:20, access, st=:surface, camera=(40, 50))

# ■ 5.3 MC with exploring starts

Q = zeros(20, 2, 10, 2)
N = zeros(Int, size(Q)...)

for episode = 1:1e6
    trajectory = []

    dealer = Hand()
    hit!(dealer); hit!(dealer)

    hand = Hand()
    hit!(hand); hit!(hand)

    # starting with a random action
    action = rand(1:2)

    savestep!(trajectory, hand, dealer, action)

    while action == 1 && getscore(hand) <= 21
        hit!(hand)
        savestep!(trajectory, hand, dealer, action)

        action = argmax(Q, dims=4)
    end

    if getscore(hand) > 21
        # we've already lost at this point
        pop!(trajectory)

        improve!(Q, N, trajectory, reward=-1.0)
        continue
    end

    if getscore(hand) == 21 && getscore(dealer) != 21
        improve!(Q, N, trajectory, reward=1.0)
        continue
    end

    while getscore(dealer) < 17
        hit!(dealer)
    end

    reward = if getscore(hand) == getscore(dealer)
        0.0
    elseif getscore(hand) > getscore(dealer) || getscore(dealer) > 21
        1.0
    else
        -1.0
    end

    improve!(Q, N, trajectory; reward)
end

Qₐ = maximum(Q, dims=4)

access(dealercard, score) = Qₐ[score, 1, dealercard]
plot(1:10, 11:20, access, st=:surface, camera=(40, 50))
# ■
