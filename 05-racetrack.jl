using ReinforcementLearningBase

# increase in velocity in a given direction
const action_space = [
    (-1, 0)
  , (1, 0)
  , (0, 1)
  , (0, -1)
]

mutable struct Racetrack <: AbstractEnv
    startline::Vector{Vector{Int}}
    finishline::Vector{Vector{Int}}
    walls::Vector{Vector{Int}}
    griddims::Tuple{Int, Int}
    position::Vector{Int}
    velocity::Vector{Int}
    reward::Float64

    Racetrack(s, f, w, gd) = new(s, f, w, gd, copy(rand(s)), zeros(2), -1)
end

RLBase.get_action_space(::Racetrack) = DiscreteSpace(4)
RLBase.get_reward(env::Racetrack) = env.reward
RLBase.get_terminal(env::Racetrack) = env.position in env.finishline
RLBase.get_state(env::Racetrack) = copy(env.position)
RLBase.get_observation_space(env::Racetrack) = DiscreteSpace(1:prod(env.griddims))

function RLBase.observe(env::Racetrack)
    return ( state = copy(env.position),
             reward = env.reward,
             terminal = env.position in env.finishline )
end

function RLBase.reset!(env::Racetrack)
    fill!(env.velocity, 0)

    env.position = copy(rand(env.startline))
end

function (env::Racetrack)(action::Int)
    accel_x, accel_y = action_space[action]
    speed_x, speed_y = env.velocity

    # update the actual velocity from the acceleration for the next step
    env.velocity[1] = min(max(speed_x + accel_x, -3), 3)
    env.velocity[2] = min(max(speed_y + accel_y, -3), 3)

    while true
        if env.position ∈ env.walls
            env.position = copy(rand(env.startline))
            fill!(env.velocity, 0)

            break
        end

        if env.position ∈ env.finishline
            break
        end

        # incremental updates to the position, easier to check for the barriers
        if speed_x != 0
            env.position[1] += sign(speed_x)
            speed_x -= sign(speed_x)

        elseif speed_y != 0
            env.position[2] += sign(speed_y)
            speed_y -= sign(speed_y)
        else
            break
        end

        # racing is hard
        if rand() < 0.1
            fill!(env.velocity, 0)
        end
    end

    nothing
end

function Base.display(env::Racetrack)
    NX, NY = env.griddims

    for y in NY:-1:0
        for x in 0:NX+1
            p = [x, y]

            if p == env.position
                print("A")
            elseif p in env.startline
                print("0")
            elseif p in env.finishline
                print("F")
            elseif p in env.walls
                print("x")
            else
                print(" ")
            end
        end

        println()
    end
end

function Racetrack()
    NY = 20
    NX = 10

    # trivial setup for starters
    startline = [[i, 1] for i = 1:NX]
    finishline = [[i, NY] for i = 1:NX]

    walls = vcat(
        [[0, i] for i = 1:NY],
        [[i, 0] for i = 1:NX], # safety net
        [[NX+1, i] for i = 1:NY]
    )

    Racetrack(startline, finishline, walls, (NX, NY))
end
