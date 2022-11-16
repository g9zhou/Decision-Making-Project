include("setup.jl")

struct HistoryMonteCarloTreeSearch
    𝒫
    N
    Q
    d
    m
    c
    U
end

function explore(π::HistoryMonteCarloTreeSearch,s, h)
    𝒜, N, Q, c = π.𝒫.𝒜, π.N, π.Q, π.c
    𝒜 = possible_action(s,𝒜,R)
    Nh = sum(get(N, (h,a), 0) for a in 𝒜)
    # print("state:",s,"\n")
    # print("𝒜:",𝒜,"\n")
    # print("argmax:",argmax(a->Q[(h,a)]+c*bonus(N[(h,a)],Nh),𝒜),"\n")
    # print("history:",h,"\n")
    return argmax(a->Q[(h,a)]+c*bonus(N[(h,a)],Nh),𝒜)
end

function simulate(π::HistoryMonteCarloTreeSearch,s,h,d)
    if d <= 0
        return 0.0
    end
    𝒫, N, Q, c = π.𝒫, π.N, π.Q, π.c
    𝒮, 𝒜, TRO, γ = 𝒫.𝒮, 𝒫.𝒜, 𝒫.TRO, 𝒫.γ
    # 𝒜 = possible_action(s,𝒜,R)
    if !haskey(N, (h,first(𝒜)))
        for a in 𝒜
            N[(h,a)] = 0
            Q[(h,a)] = 0.0
        end
        return rollout(𝒫,s,π,d)
    end
    a = explore(π, s, h)
    s′, r, o = TRO(s,a)
    q = r+γ*simulate(π,s′,vcat(h,(a,o)), d-1)
    N[(h,a)] += 1
    Q[(h,a)] += (q-Q[(h,a)])/N[(h,a)]
    return q
end

bonus(Nsa,Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)

function (π::HistoryMonteCarloTreeSearch)(b,𝒜,h=[])
    s = 0
    for i in 1:π.m
        s = sample(π.𝒫.𝒮,Weights(b))
        # s = rand(SetCategorical(π.𝒫.𝒮,b))
        simulate(π,s,h,π.d)
    end
    𝒜 = possible_action(argmax(b),𝒜,R)
    action = argmax(a->π.Q[(h,a)],𝒜)
    print(π.Q[(h,action)])
    return action
end

function rollout(𝒫,s,π,d)
    if d ≤ 0
        return 0.0
    end
    a = rollout_policy(s,𝒫.𝒜)
    s′, r, o = TRO(s,a)
    return r + 𝒫.γ*rollout(𝒫,s′,π,d-1)
end

function rollout_policy(s,𝒜)
    𝒜 = possible_action(s,𝒜,R)
    return sample(𝒜,Weights(ones(length(𝒜))))
end

type = MonteCarlo()
𝒪 = copy(𝒮)
fire, fire_all, reward_matrix = remove_fire(7543,fire,fire_all,reward_matrix)
fire, fire_all, reward_matrix = remove_fire(3137,fire,fire_all,reward_matrix)
R(s,a) = reward_matrix[s]
T(s,a,s′) = trans_prob(type,dim,fire,s,a,s′)
O(s) = observation_func(s)
TRO(s,a) = trans_reward_obs(type,𝒜,dim,fire,fire_all,s,a)
𝒫 = POMDP(γ,𝒮,𝒜,𝒪,T,R,O,TRO)
N = Dict{Tuple{Vector{Any},Int64},Int64}()
Q = Dict{Tuple{Vector{Any},Int64},Float64}()
# N[([],3)] = 0
# Q[([],3)] = 0.0
d = 60
m = 500
c = 2
π = HistoryMonteCarloTreeSearch(𝒫,N,Q,d,m,c,nothing)
function propagate()
    s = 2001
    i = 1
    path1 = [s]
    while !any(s .== fire_all)
        b = O(s)
        a = π(b,𝒜)
        s, _ = trans_reward(type,𝒜,dim,fire,fire_all,s,a)
        append!(path1,s)
        i += 1
        print(i,",",a,",",s,"\n")
        # if i == 2 break end
    end
    return path1
end
path1 = propagate()