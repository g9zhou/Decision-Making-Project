include("setup.jl")

struct HistoryMonteCarloTreeSearch
    š«
    N
    Q
    d
    m
    c
    U
end

function explore(Ļ::HistoryMonteCarloTreeSearch,s, h)
    š, N, Q, c = Ļ.š«.š, Ļ.N, Ļ.Q, Ļ.c
    š = possible_action(s,š,R)
    Nh = sum(get(N, (h,a), 0) for a in š)
    # print("state:",s,"\n")
    # print("š:",š,"\n")
    # print("argmax:",argmax(a->Q[(h,a)]+c*bonus(N[(h,a)],Nh),š),"\n")
    # print("history:",h,"\n")
    return argmax(a->Q[(h,a)]+c*bonus(N[(h,a)],Nh),š)
end

function simulate(Ļ::HistoryMonteCarloTreeSearch,s,h,d)
    if d <= 0
        return 0.0
    end
    š«, N, Q, c = Ļ.š«, Ļ.N, Ļ.Q, Ļ.c
    š®, š, TRO, Ī³ = š«.š®, š«.š, š«.TRO, š«.Ī³
    # š = possible_action(s,š,R)
    if !haskey(N, (h,first(š)))
        for a in š
            N[(h,a)] = 0
            Q[(h,a)] = 0.0
        end
        return rollout(š«,s,Ļ,d)
    end
    a = explore(Ļ, s, h)
    sā², r, o = TRO(s,a)
    q = r+Ī³*simulate(Ļ,sā²,vcat(h,(a,o)), d-1)
    N[(h,a)] += 1
    Q[(h,a)] += (q-Q[(h,a)])/N[(h,a)]
    return q
end

bonus(Nsa,Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)

function (Ļ::HistoryMonteCarloTreeSearch)(b,š,h=[])
    s = 0
    for i in 1:Ļ.m
        s = sample(Ļ.š«.š®,Weights(b))
        # s = rand(SetCategorical(Ļ.š«.š®,b))
        simulate(Ļ,s,h,Ļ.d)
    end
    š = possible_action(argmax(b),š,R)
    action = argmax(a->Ļ.Q[(h,a)],š)
    print(Ļ.Q[(h,action)])
    return action
end

function rollout(š«,s,Ļ,d)
    if d ā¤ 0
        return 0.0
    end
    a = rollout_policy(s,š«.š)
    sā², r, o = TRO(s,a)
    return r + š«.Ī³*rollout(š«,sā²,Ļ,d-1)
end

function rollout_policy(s,š)
    š = possible_action(s,š,R)
    return sample(š,Weights(ones(length(š))))
end

type = MonteCarlo()
šŖ = copy(š®)
fire, fire_all, reward_matrix = remove_fire(7543,fire,fire_all,reward_matrix)
fire, fire_all, reward_matrix = remove_fire(3137,fire,fire_all,reward_matrix)
R(s,a) = reward_matrix[s]
T(s,a,sā²) = trans_prob(type,dim,fire,s,a,sā²)
O(s) = observation_func(s)
TRO(s,a) = trans_reward_obs(type,š,dim,fire,fire_all,s,a)
š« = POMDP(Ī³,š®,š,šŖ,T,R,O,TRO)
N = Dict{Tuple{Vector{Any},Int64},Int64}()
Q = Dict{Tuple{Vector{Any},Int64},Float64}()
# N[([],3)] = 0
# Q[([],3)] = 0.0
d = 60
m = 500
c = 2
Ļ = HistoryMonteCarloTreeSearch(š«,N,Q,d,m,c,nothing)
function propagate()
    s = 2001
    i = 1
    path1 = [s]
    while !any(s .== fire_all)
        b = O(s)
        a = Ļ(b,š)
        s, _ = trans_reward(type,š,dim,fire,fire_all,s,a)
        append!(path1,s)
        i += 1
        print(i,",",a,",",s,"\n")
        # if i == 2 break end
    end
    return path1
end
path1 = propagate()