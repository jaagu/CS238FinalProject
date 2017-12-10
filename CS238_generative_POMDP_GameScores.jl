#Pkg.add("POMDPs")
using POMDPs
#POMDPs.add_all() # this may take a few minutes
importall POMDPs
using DataFrames
POMDPs.add("ParticleFilters")
using ParticleFilters
POMDPs.add("BasicPOMCP")
using BasicPOMCP
using POMDPModels
using POMDPToolbox
using ParticleFilters
#Pkg.add("JLD")
using JLD
import ParticleFilters: obs_weight

#POMCP.uses_states_from_planner(::Array{Int64,1}) = true

game_data = readtable("game_data_new.csv", header = true)
score_data = readtable("score_data_new.csv", header = true)
numIter = 1
numSave = 0
reward_stream = Int64[]

type team_POMDP <: POMDP{Array{Int64,1}, Array{Int64,1}, Array{Int64,1}} # POMDP{State, Action, Observation} all parametarized by Vectors of Int64s
    num_teams::Int64
end
team_POMDP() = team_POMDP(30)

function generate_s(pomdp::team_POMDP, state::Array{Int64,1}, action::Array{Int64,1}, rng::AbstractRNG)
    global numIter
    global score_data
    numIter = rand(1:size(score_data)[1])
    return vec(convert(Array{Int64},score_data[numIter, :]))
end

function generate_o(pomdp::team_POMDP, state::Array{Int64,1}, action::Array{Int64,1}, next_state::Array{Int64,1}, rng::AbstractRNG)
    obs = copy(next_state)
    indices = find(obs)
    obs[indices[1]] == 1
    obs[indices[2]] = 1
    return obs
end

state_index_map = Dict{Array{Int64,1},Int64}()

function POMDPs.states(mdp::team_POMDP)
    s = Array{Int64,1}[] # initialize an array
    # loop over all our states
    num = 0
    for x = 1:mdp.num_teams, y = 1:mdp.num_teams
        if x == y
            continue
        end
        for z = 50:155, w = 50:155
            num += 1
            nextState = zeros(Int64, 1, 2*mdp.num_teams)
            nextState[x] = z
            nextState[mdp.num_teams + y] = w
            push!(s, vec(nextState))
            state_index_map[vec(nextState)] = num
        end
    end
    return s
end;

function state_index(mdp::team_POMDP, state::Array{Int64,1})
    global state_index_map
    _ = POMDPs.states(mdp)
    return state_index_map[state]
end

n_observations(mdp::team_POMDP) = 30*29

function n_states(mdp::team_POMDP)
    return length(POMDPs.states(mdp))
end

function POMDPs.actions(mdp::team_POMDP, state::Array{Int64,1})
    s = Vector{Int64}[] # initialize an array
    for z = 50:5:150, w = 50:5:150
        push!(s, [z,w])
    end
    return s
end

action_index_map = Dict{Array{Int64,1},Int64}()

function index_actions()
    global action_index_map
    num_teams = 30
    num = 0
    for z = 50:5:150, w = 50:5:150
        num += 1
        action_index_map[[z,w]] = num
    end
end

function POMDPs.actions(mdp::team_POMDP)
    s = Vector{Int64}[] # initialize an array
    for z = 50:5:150, w = 50:5:150
        push!(s, [z,w])
    end
    return s
end

function action_index(mdp::team_POMDP, action::Array{Int64,1})
    global action_index_map
    index_actions()
    return action_index_map[action]
end

function n_actions(mdp::team_POMDP)
    return length(POMDPs.actions(mdp))
end

function reward(pomdp::team_POMDP, state::Array{Int64,1}, action::Array{Int64,1})
    global reward_stream
    global numSave
    numSave += 1

    reward = 0
    indices = find(state)
    reward -= (action[1] - state[indices[1]])*(action[1] - state[indices[1]])
    reward -= (action[2] - state[indices[2]])*(action[2] - state[indices[2]])

    push!(reward_stream,reward)

    if(numSave % 100000 == 0)
        print("HERE!\n")
        writedlm("test.txt", reward_stream)
    end

    return reward
end

function initial_state(p::team_POMDP, rng::AbstractRNG)
    global numIter
    global score_data
    numIter = rand(1:size(score_data)[1])
    return vec(convert(Array{Int64},score_data[numIter, :]))
end

function initial_state_distribution(mdp::team_POMDP)
    s = Array{Int64,1}[] # initialize an array
    # loop over all our states
    num = 0
    for x = 1:mdp.num_teams, y = 1:mdp.num_teams
        if x == y
            continue
        end
        for z = 50:155, w = 50:155
            num += 1
            nextState = zeros(Int64, 1, 2*mdp.num_teams)
            nextState[x] = z
            nextState[mdp.num_teams + y] = w
            push!(s, vec(nextState))
            state_index_map[vec(nextState)] = num
        end
    end
    return s
end

function isterminal(p::team_POMDP, s::Array{Int64,1})
    return false
end

function obs_weight(pomdp::team_POMDP, state::Array{Int64,1}, action::Array{Int64,1}, nextState::Array{Int64,1}, obs::Array{Int64,1})
    return 1
end

discount(p::team_POMDP) = 1

pomdp = team_POMDP()
solver = POMCPSolver()
policy = solve(solver, pomdp)
filter = SIRParticleFilter(pomdp, 10000)
hist = simulate(HistoryRecorder(max_steps=1000000), pomdp, policy, filter)
print(hist)

println("reward: $(discounted_reward(hist))")
