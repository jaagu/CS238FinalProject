#Pkg.add("POMDPs")
using POMDPs
#POMDPs.add_all() # this may take a few minutes
importall POMDPs
using DataFrames
POMDPs.add("POMCPOW")
using POMCPOW
POMDPs.add("ParticleFilters")
using ParticleFilters
POMDPs.add("BasicPOMCP")
using BasicPOMCP
using POMDPModels
using POMDPToolbox
using ParticleFilters
using JLD
import ParticleFilters: obs_weight

game_data = readtable("game_data.csv", header = true)
score_data = readtable("score_data.csv", header = true)
numIter = 1
numSave = 0
reward_stream = Int64[]

type team_POMDP <: POMDP{Array{Int64,1}, Int64, Array{Int64,1}} # POMDP{State, Action, Observation} all parametarized by Vectors of Int64s
    num_teams::Int64
end
team_POMDP() = team_POMDP(30)

function generate_s(pomdp::team_POMDP, state::Array{Int64,1}, action::Int64, rng::AbstractRNG)
    global numIter
    global game_data
    global score_data
    numIter = rand(1:size(score_data)[1])
    next_state = vec(convert(Array{Int64},copy(game_data[numIter, :])))
    indices = find(next_state)
    state_score = vec(convert(Array{Int64},copy(score_data[numIter, :])))
    # print(state_score[indices[1]])
    if state_score[indices[1]]>state_score[indices[2]]
        next_state[indices[2]] = -1
    else
        next_state[indices[1]] = -1
    end
    return next_state
end

function generate_o(pomdp::team_POMDP, state::Array{Int64,1}, action::Int64, next_state::Array{Int64,1}, rng::AbstractRNG)
    obs = copy(next_state)
    indices = find(obs)
    if obs[indices[1]] == -1
        obs[indices[1]] = 1
    else
        obs[indices[2]] = 1
    end
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
        num += 1
        nextState_1 = zeros(Int64, 1, 2*mdp.num_teams)
        nextState_1[x] = 1
        nextState_1[mdp.num_teams + y] = -1
        push!(s, vec(nextState_1))
        state_index_map[vec(nextState_1)] = num
        num += 1
        nextState_2 = zeros(Int64, 1, 2*mdp.num_teams)
        nextState_2[x] = -1
        nextState_2[mdp.num_teams + y] = 1
        push!(s, vec(nextState_2))
        state_index_map[vec(nextState_2)] = num
    end
    return s
end;

function state_index(mdp::team_POMDP, state::Array{Int64,1})
    global state_index_map
    return state_index_map[state]
end

n_observations(mdp::team_POMDP) = 30*29

n_states(mdp::team_POMDP) = 30*29*2

function POMDPs.actions(mdp::team_POMDP, state::Array{Int64,1})
    return[-1,1]
end

function POMDPs.actions(mdp::team_POMDP)
    return [-1,1]
end

function action_index(mdp::team_POMDP, action::Int64)
    if action == -1
        return 1
    end
    return 2
end

n_actions(mdp::team_POMDP) = 2

function reward(pomdp::team_POMDP, state::Array{Int64,1}, action::Int64)
    global reward_stream
    global numSave
    numSave += 1

    reward = 1
    indices = find(state)
    if state[indices[1]] == 1
        if action == -1
            reward = -1
        end
    else
        if action == 1
            reward = -1
        end
    end

    push!(reward_stream,reward)

    if(numSave % 100000 == 0)
        print("HERE!\n")
        writedlm("test2.txt", reward_stream)
    end
    return reward
end

function initial_state(p::team_POMDP, rng::AbstractRNG)
    global numIter
    global game_data
    global score_data
    numIter = rand(1:size(score_data)[1])
    next_state = vec(convert(Array{Int64},copy(game_data[numIter, :])))
    indices = find(next_state)
    print(score_data[numIter][indices[1]])
    if score_data[numIter][indices[1]]>score_data[numIter][indices[2]]
        next_state[indices[2]] = -1
    else
        next_state[indices[1]] = -1
    end
    return next_state
end

function initial_state_distribution(mdp::team_POMDP)
    global numIter
    global game_data
    global score_data
    s = Array{Int64,1}[] # initialize an array
    num = 0
    for x = 1:mdp.num_teams, y = 1:mdp.num_teams
        if x == y
            continue
        end
        num += 1
        nextState_1 = zeros(Int64, 1, 2*mdp.num_teams)
        nextState_1[x] = 1
        nextState_1[mdp.num_teams + y] = -1
        push!(s, vec(nextState_1))
        state_index_map[vec(nextState_1)] = num
        num += 1
        nextState_2 = zeros(Int64, 1, 2*mdp.num_teams)
        nextState_2[x] = -1
        nextState_2[mdp.num_teams + y] = 1
        push!(s, vec(nextState_2))
        state_index_map[vec(nextState_2)] = num
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
