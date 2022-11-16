using LinearAlgebra
using Plots
using DataFrames, CSV, Tables
using StatsBase
using GLMakie

dim = 20
γ = 0.95
State = zeros(20,20,20)
𝒮 = LinearIndices(State)
𝒜 = [1,2,3,4,5,6]
fire1 = 𝒮[18,3,2]
fire2 = 𝒮[17,17,6]
fire3 = 𝒮[3,18,17]
fire = [fire1, fire2, fire3]
global fire_all = [fire1,fire1+400,fire1+800,fire2,fire2+400,fire2+800,fire3,fire3+400,fire3+800]
base = ones(Int,20,20)*5
mountain1 = [10 12 11 12 12
            10 10 10 10 10
            9 9 9 10 8
            9 8 8 9 7
            6 6 6 6 5]

mountain2 = [6 6 7 6 6
            6 9 10 9 6
            5 11 14 11 5 
            6 9 10 9 6
            7 7 8 7 7]

mountain3 = [7 8 7 6 6
            7 10 8 7 6
            7 10 9 7 6
            7 10 8 7 6
            7 7 6 6 6]

mountain4 = [6 8 11 13 13 12
            6 9 11 14 13 12
            6 10 16 16 11 8
            7 9 11 12 14 15
            7 8 9 9 15 15
            6 6 6 6 8 8]

basin1 = [4 4 4 4 4
        4 1 1 1 4
        4 1 1 1 4
        4 2 1 3 4
        4 4 4 4 4]

base[11:15,1:5] = mountain1
base[11:15,6:10] = mountain2
base[16:20,1:5] = basin1
base[16:20,6:10] = mountain3
base[1:6,15:20] = mountain4
reward_matrix = copy(State)
for i in 1:20
    for j in 1:20
        height = base[i,j]
        reward_matrix[i,j,1:height] .= -10.0
    end
end

## Bridge
reward_matrix[14,4:5,10:11] .= -10
reward_matrix[15,5:6,10:11] .= -10
reward_matrix[16,5:7,11:12] .= -10
reward_matrix[17,7,11] = -10

## Fire
reward_matrix[[fire1,fire1+400,fire1+800]] .= 10
reward_matrix[[fire2,fire2+400,fire2+800]] .= 8
reward_matrix[[fire3,fire3+400,fire3+800]] .= 8
# figure = heatmap(reward_matrix,show=true)
# display(figure)
# reward_matrix_2d = reshape(reward_matrix,20,400)
# CSV.write("landscape.csv", Tables.table(reward_matrix_2d), writeheader=false)

R(s,a) =reward_matrix[s]

mutable struct MDP
    γ
    𝒮
    𝒜
    T
    R
    TR
end

mutable struct POMDP
    γ
    𝒮
    𝒜
    𝒪
    T
    R
    O
    TRO
end

struct ValueIteration
    convergence
end

struct MonteCarlo
end

struct SARSA
end

###########################################
# Value Iteration
###########################################
function dist_prob(dim,fire,s_x,s_y,s_z)
    fire_y, fire_x, fire_z = CartesianIndices((dim,dim,dim))[fire][1],CartesianIndices((dim,dim,dim))[fire][2], CartesianIndices((dim,dim,dim))[fire][3]
    # compute distance between fire and agent
    # different level of turbulence as a piecewise function of distance
    dist = floor(norm([fire_x, fire_y, fire_z]-[s_x,s_y,s_z])*2)/2
    # for a specific range of dist, there is ω probability the agent will move in a random direction,
    # each direction with ω/4 probability. There is an additional 1-ω probability the drone will move
    # in the desired position.
    if dist <= 0.5
        ω = 0.9
    elseif dist <= 1
        ω = 0.8
    elseif dist <= 1.5
        ω = 0.6
    elseif dist <= 2
        ω = 0.4
    elseif dist <= 2.5
        ω = 0.2
    else
        ω = 0
    end
    return ω
end

function trans_prob(type::ValueIteration,dim,fire,s,a,s′)
    # extract cartesian index of fire and agent from linear index
    s_y, s_x, s_z = CartesianIndices((dim,dim,dim))[s][1],CartesianIndices((dim,dim,dim))[s][2],CartesianIndices((dim,dim,dim))[s][3]
    s′_y, s′_x, s′_z = CartesianIndices((dim,dim,dim))[s′][1],CartesianIndices((dim,dim,dim))[s′][2], CartesianIndices((dim,dim,dim))[s′][3]
    ω = []
    for fire_i in fire append!(ω, dist_prob(dim,fire_i,s_x,s_y,s_z)) end
    ω = maximum(ω)
    # print(ω)

    if (s′_x, s′_y, s′_z) == state_action_trans(type,dim,s,a)
        T = 1-ω+ω/6
    elseif norm([s′_x, s′_y, s′_z]-[s_x, s_y, s_z]) == 1
        T = ω/6
    else
        T = 0
    end
    # print(T)
    return T
end

function state_action_trans(type::ValueIteration,dim,s,a)
    # Given a grid dimension (int), a state (linear index), an action (int)
    # return corresponding s′ (Cartesian index) in the (dim,dim) grid world

    s_y, s_x, s_z = CartesianIndices((dim,dim,dim))[s][1],CartesianIndices((dim,dim,dim))[s][2],CartesianIndices((dim,dim,dim))[s][3]   # CartesianIndices return (y,x,z)

    # action 1,2,3,4 corresponds to front (y-1), back (y+1), left (x-1), right (x+1), up (z+1), down (z-1)
    # due to the nature of CartesianIndices
    if a == 1
        s′_x, s′_y, s′_z = s_x, s_y-1, s_z
    elseif a == 2
        s′_x, s′_y, s′_z = s_x, s_y+1, s_z
    elseif a == 3
        s′_x, s′_y, s′_z = s_x-1, s_y, s_z
    elseif a == 4
        s′_x, s′_y, s′_z = s_x+1, s_y, s_z
    elseif a == 5
        s′_x, s′_y, s′_z = s_x, s_y, s_z+1
    elseif a == 6
        s′_x, s′_y, s′_z = s_x, s_y, s_z-1
    end
    return (s′_x, s′_y, s′_z)
end

function transition(type::ValueIteration,𝒜,dim,fire,s,a)
    𝒜 = possible_action(s,𝒜,R)
    # print(𝒜)
    s′_all = [state_action_trans(type,dim,s,action) for action in 𝒜]
    s′_all = [LinearIndices((dim,dim,dim))[s′[2],s′[1],s′[3]] for s′ in s′_all]
    # print(s′_all)
    ω = [trans_prob(type,dim,fire,s,a,s′) for s′ in s′_all]
    # print(ω)
    s′ = sample(s′_all,Weights(ω))
    # print(s′)
    return s′
end

function reachable_state(s)
    S = []
    s_y, s_x, s_z = CartesianIndices((dim,dim,dim))[s][1],CartesianIndices((dim,dim,dim))[s][2],CartesianIndices((dim,dim,dim))[s][3]
    if s_y != 1 append!(S, LinearIndices((dim, dim, dim))[s_y-1,s_x,s_z]) end
    if s_y != 20 append!(S, LinearIndices((dim, dim, dim))[s_y+1,s_x,s_z]) end
    if s_x != 1 append!(S, LinearIndices((dim, dim, dim))[s_y,s_x-1,s_z]) end
    if s_x != 20 append!(S, LinearIndices((dim, dim, dim))[s_y,s_x+1,s_z]) end
    if s_z != 1 append!(S, LinearIndices((dim, dim, dim))[s_y,s_x,s_z-1]) end
    if s_z != 20 append!(S, LinearIndices((dim, dim, dim))[s_y,s_x,s_z+1]) end
    return S
end

function remove_fire(s,fire,fire_all,reward_matrix)
    fire_all_index = findfirst(x->x==s, fire_all)
    fire_index = ceil(Int,fire_all_index/3)
    recover = fire_all[fire_index*3-2:fire_index*3]
    reward_matrix[recover] .= 0
    fire = deleteat!(fire,fire_index)
    fire_all = deleteat!(fire_all,fire_index*3-2:fire_index*3)
    return fire, fire_all, reward_matrix
end

function extract_policy(U)
    policy = []
    for z in 1:dim
        for x in 1:dim
            for y in 1:dim
                values = ones(6)*-10
                if checkindex(Bool,1:dim,y-1) values[1] = U[y-1,x,z] end
                if checkindex(Bool,1:dim,y+1) values[2] = U[y+1,x,z] end
                if checkindex(Bool,1:dim,x-1) values[3] = U[y,x-1,z] end
                if checkindex(Bool,1:dim,x+1) values[4] = U[y,x+1,z] end
                if checkindex(Bool,1:dim,z+1) values[5] = U[y,x,z+1] end
                if checkindex(Bool,1:dim,z-1) values[6] = U[y,x,z-1] end

                action = argmax(values)
                if values[action] == -10 action = 5 end
                
                direc = []
                if action == 1 direc = Vec3f(-1.0,0.0,0.0) end
                if action == 2 direc = Vec3f(1.0,0.0,0.0)end
                if action == 3 direc = Vec3f(0.0,-1.0,0.0) end
                if action == 4 direc = Vec3f(0.0,1.0,0.0) end
                if action == 5 direc = Vec3f(0.0,0.0,1.0) end
                if action == 6 direc = Vec3f(0.0,0.0,-1.0) end
                append!(policy,[direc])
            end
        end
    end
    return policy
end

function plot_actual_path(path::Vector{Int64})
    x = []
    y = []
    z = []
    # traj = []
    for point in path
        s_y, s_x, s_z = CartesianIndices((dim,dim,dim))[point][1],CartesianIndices((dim,dim,dim))[point][2],CartesianIndices((dim,dim,dim))[point][3]
        append!(y,s_y)
        append!(x,s_x)
        append!(z,s_z)
        # push!(traj,[Vec3f(point[2],point[1],point[3])])
    end
    return x,y,z
    # return traj
end

function plot_optimal_path(policy,s,goal)
    s_y, s_x, s_z = CartesianIndices((dim,dim,dim))[s][1],CartesianIndices((dim,dim,dim))[s][2],CartesianIndices((dim,dim,dim))[s][3]
    x = [s_x]
    y = [s_y]
    z = [s_z]
    while s != goal
        action = policy[s]
        s_y += Int(action[1])
        s_x += Int(action[2])
        s_z += Int(action[3])
        append!(x,s_x)
        append!(y,s_y)
        append!(z,s_z)
        s = LinearIndices((dim,dim,dim))[s_y,s_x,s_z]
        print(s)
    end
    return x,y,z
end

###########################################
# MonteCarlo
###########################################
function trans_prob(type::MonteCarlo,dim,fire,s,a,s′)
    s_y, s_x, s_z = CartesianIndices((dim,dim,dim))[s][1],CartesianIndices((dim,dim,dim))[s][2],CartesianIndices((dim,dim,dim))[s][3]
    s′_y, s′_x, s′_z = CartesianIndices((dim,dim,dim))[s′][1],CartesianIndices((dim,dim,dim))[s′][2], CartesianIndices((dim,dim,dim))[s′][3]
    ω = []
    for fire_i in fire append!(ω, dist_prob(dim,fire_i,s_x,s_y,s_z)) end
    ω = maximum(ω)

    if s′ == state_action_trans(type,dim,s,a)
        T = 1-ω+ω/6
    elseif norm([s′_x, s′_y, s′_z]-[s_x, s_y, s_z]) == 1
        T = ω/6
    else
        T = 0
    end
    return T
end

function state_action_trans(type::MonteCarlo,dim,s,a)
    # Given a grid dimension (int), a state (linear index), an action (int)
    # return corresponding s′ (Cartesian index) in the (dim,dim) grid world

    s_y, s_x, s_z = CartesianIndices((dim,dim,dim))[s][1],CartesianIndices((dim,dim,dim))[s][2],CartesianIndices((dim,dim,dim))[s][3]   # CartesianIndices return (y,x)

    # action 1,2,3,4 corresponds to up (y-1), down (y+1), left (x-1), right (x+1)
    # due to the nature of CartesianIndices
    if a == 1
        s′_x, s′_y, s′_z = s_x, s_y-1, s_z
    elseif a == 2
        s′_x, s′_y, s′_z = s_x, s_y+1, s_z
    elseif a == 3
        s′_x, s′_y, s′_z = s_x-1, s_y, s_z
    elseif a == 4
        s′_x, s′_y, s′_z = s_x+1, s_y, s_z
    elseif a == 5
        s′_x, s′_y, s′_z = s_x, s_y, s_z+1
    elseif a == 6
        s′_x, s′_y, s′_z = s_x, s_y, s_z-1
    end
    if checkindex(Bool,1:dim,s′_x) && checkindex(Bool,1:dim,s′_y) && checkindex(Bool,1:dim,s′_z)
        return LinearIndices((dim,dim,dim))[s′_y,s′_x,s′_z]
    else
        return s
    end
end

function trans_reward(type::MonteCarlo,𝒜,dim,fire,fire_all,s,a)
    s′_all = [state_action_trans(type,dim,s,action) for action in 𝒜]
    # print(s′_all)
    ω = [trans_prob(type,dim,fire,s,a,s′) for s′ in s′_all]
    # print(ω)
    s′ = sample(s′_all,Weights(ω))
    r = any(s′.== fire_all) ? reward_matrix[s′] : 0
    s_y, s_x, s_z = CartesianIndices((dim,dim,dim))[s][1],CartesianIndices((dim,dim,dim))[s][2],CartesianIndices((dim,dim,dim))[s][3]   # CartesianIndices return (y,x)
    s′_y, s′_x, s′_z = CartesianIndices((dim,dim,dim))[s′][1],CartesianIndices((dim,dim,dim))[s′][2], CartesianIndices((dim,dim,dim))[s′][3]
    fire_y,fire_x,fire_z = CartesianIndices((dim,dim,dim))[fire[1]][1],CartesianIndices((dim,dim,dim))[fire[1]][2],CartesianIndices((dim,dim,dim))[fire[1]][3]
    if norm([s_x,s_y,s_z]-[fire_x,fire_y,fire_z]) > norm([s′_x, s′_y, s′_z]-[fire_x,fire_y,fire_z]) r += 0.2 end 
    # r = reward_matrix[s′]
    return s′, r
end

###########################################
# Sarsa Lambda
###########################################
function trans_prob(type::SARSA,dim,fire,s,a,s′)
    # extract cartesian index of fire and agent from linear index
    fire1,fire2,fire3 = fire
    s_y, s_x, s_z = CartesianIndices((dim,dim,dim))[s][1],CartesianIndices((dim,dim,dim))[s][2],CartesianIndices((dim,dim,dim))[s][3]
    s′_y, s′_x, s′_z = CartesianIndices((dim,dim,dim))[s′][1],CartesianIndices((dim,dim,dim))[s′][2], CartesianIndices((dim,dim,dim))[s′][3]
    ω1 = dist_prob(dim,fire1,s_x,s_y,s_z)
    ω2 = dist_prob(dim,fire2,s_x,s_y,s_z)
    ω3 = dist_prob(dim,fire3,s_x,s_y,s_z)
    ω = max(ω1,ω2,ω3)

    if s′ == state_action_trans(type,dim,s,a)
        T = 1-ω+ω/6
    elseif norm([s′_x, s′_y, s′_z]-[s_x, s_y, s_z]) == 1
        T = ω/6
    else
        T = 0
    end
    return T
end

function state_action_trans(type::SARSA,dim,s,a)
    # Given a grid dimension (int), a state (linear index), an action (int)
    # return corresponding s′ (Cartesian index) in the (dim,dim) grid world

    s_y, s_x, s_z = CartesianIndices((dim,dim,dim))[s][1],CartesianIndices((dim,dim,dim))[s][2],CartesianIndices((dim,dim,dim))[s][3]   # CartesianIndices return (y,x)
    # action 1,2,3,4 corresponds to up (y-1), down (y+1), left (x-1), right (x+1)
    # due to the nature of CartesianIndices
    if a == 1
        s′_x, s′_y, s′_z = s_x, s_y-1, s_z
    elseif a == 2
        s′_x, s′_y, s′_z = s_x, s_y+1, s_z
    elseif a == 3
        s′_x, s′_y, s′_z = s_x-1, s_y, s_z
    elseif a == 4
        s′_x, s′_y, s′_z = s_x+1, s_y, s_z
    elseif a == 5
        s′_x, s′_y, s′_z = s_x, s_y, s_z+1
    elseif a == 6
        s′_x, s′_y, s′_z = s_x, s_y, s_z-1
    end
    if checkindex(Bool,1:dim,s′_x) && checkindex(Bool,1:dim,s′_y) && checkindex(Bool,1:dim,s′_z)
        return LinearIndices((dim,dim,dim))[s′_y,s′_x,s′_z]
    else
        return s
    end
end

function trans_reward(type::SARSA,𝒜,R,dim,fire,s,a)
    𝒜 = possible_action(s,𝒜,R)
    s′_all = [state_action_trans(type,dim,s,action) for action in 𝒜]
    # print(s,s′_all,"\n")
    ω = [trans_prob(type,dim,fire,s,a,s′) for s′ in s′_all]
    # print(s,a,ω,"\n")
    s′ = sample(s′_all,Weights(ω))
    r = any(s′ .== fire) ? reward_matrix[s′] : 0
    # print(s,a,s′,"\n")
    return s′, r
end

function possible_action(s,𝒜,R)
    # if R(s,0) == -10 return [5] end
    a = copy(𝒜)
    # if mod(s,dim) != 1 && R(s-1,0) == -10 a = filter!(x->x≠1,a) end
    if mod(s,dim) != 0 && R(s+1,0) == -10 a = filter!(x->x≠2,a) end
    # if (mod(s,dim^2) > 20 || mod(s,dim^2) == 0) && R(s-20,0) == -10 a = filter!(x->x≠3,a) end
    # if (0 < mod(s,dim^2) < 381) && R(s+20,0) == -10 a = filter!(x->x≠4,a) end
    # if s < 7600 && R(s+400,0) == -10 a = filter!(x->x≠5,a) end
    if s > 400 && R(s-400,0) == -10 a = filter!(x->x≠6,a) end
    if mod(s,dim) == 1 a = filter!(x->x≠1,a) end
    if mod(s,dim) == 0 a = filter!(x->x≠2,a) end
    if 1 <= mod(s,dim^2) <= 20 a = filter!(x->x≠3,a) end
    if mod(s,dim^2) >= 381 || mod(s,dim^2) == 0 a = filter!(x->x≠4,a) end
    if s >= dim^3-dim^2+1 a = filter!(x->x≠5,a) end
    if s <= dim^2 a = filter!(x->x≠6,a) end
    return a
end

#####################
# MonteCarlo Histroy
#####################
function observation_func(s)
    s_y, s_x, s_z = CartesianIndices((dim,dim,dim))[s][1],CartesianIndices((dim,dim,dim))[s][2],CartesianIndices((dim,dim,dim))[s][3]
    prob = []
    state = []
    off_bound = 0
    if checkindex(Bool,1:dim,s_x+1) 
        append!(prob,0.025) 
        append!(state,LinearIndices((dim,dim,dim))[s_y,s_x+1,s_z]) 
    else off_bound += 0.025 end
    if checkindex(Bool,1:dim,s_x-1) 
        append!(prob,0.025) 
        append!(state,LinearIndices((dim,dim,dim))[s_y,s_x-1,s_z]) 
    else off_bound += 0.025 end
    if checkindex(Bool,1:dim,s_y+1) 
        append!(prob,0.025) 
        append!(state,LinearIndices((dim,dim,dim))[s_y+1,s_x,s_z]) 
    else off_bound += 0.025 end
    if checkindex(Bool,1:dim,s_y-1) 
        append!(prob,0.025) 
        append!(state,LinearIndices((dim,dim,dim))[s_y-1,s_x,s_z]) 
    else off_bound += 0.025 end
    if checkindex(Bool,1:dim,s_z+1) 
        append!(prob,0.1)   
        append!(state,LinearIndices((dim,dim,dim))[s_y,s_x,s_z+1]) 
    else off_bound += 0.1 end
    if checkindex(Bool,1:dim,s_z-1) 
        append!(prob,0.1)   
        append!(state,LinearIndices((dim,dim,dim))[s_y,s_x,s_z-1]) 
    else off_bound += 0.1 end
    append!(prob,0.7+off_bound)
    append!(state,s)
    b = zeros(dim^3)
    b[state] .= prob
    return b
end

function trans_reward_obs(type::MonteCarlo,𝒜,dim,fire,fire_all,s,a)
    s′, r = trans_reward(type,𝒜,dim,fire,fire_all,s,a)
    b = observation_func(s)
    o = sample(π.𝒫.𝒮,Weights(b))
    return s′, r, o
end
