include("setup.jl")

struct MonteCarloTreeSearch
    š«
    N
    Q
    d
    m
    c
    U
end

function(Ļ::MonteCarloTreeSearch)(s)
    for k in 1:Ļ.m
        simulate!(Ļ,s)
    end
    š = possible_action(s,Ļ.š«.š,Ļ.š«.R)
    # print(š)
    action = argmax(a->Ļ.Q[(s,a)],š)
    print(Ļ.Q[(s, action)])
    # if Ļ.Q[(s, action)] == 0.0
    #     if length(š) > 1
    #         # if š[3] == 4 return 6 end
    #         return š[3]
    #     end
    #     return š[1]
    # end
    return argmax(a->Ļ.Q[(s,a)],š)
end

function simulate!(Ļ::MonteCarloTreeSearch, s, d=Ļ.d)
    if d ā¤ 0
        return 0.0
    end
    š«, N, Q, c = Ļ.š«, Ļ.N, Ļ.Q, Ļ.c
    š, R, TR, Ī³ = š«.š, š«.R, š«.TR, š«.Ī³
    # print(š«.š,"\n")
    # š = possible_action(s,š,R)
    if !haskey(N, (s,first(š)))
        for a in š
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return Ļ.U(s)
    end
    a = explore(Ļ,s)
    sā², r = TR(s,a)
    q = r + Ī³*simulate!(Ļ,sā²,d-1)
    N[(s,a)] += 1
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]
    # if R(s,0) == -10 return 0.0 end
    return q
end

bonus(Nsa,Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)

function explore(Ļ::MonteCarloTreeSearch,s)
    š, R, N, Q, c = Ļ.š«.š, Ļ.š«.R, Ļ.N, Ļ.Q, Ļ.c
    š = possible_action(s,š,R)
    Ns = sum(N[(s,a)] for a in š)
    return argmax(a->Q[(s,a)] + c*bonus(N[(s,a)], Ns), š)
end

type = MonteCarlo()
T(s,a,sā²) = trans_prob(type,dim,fire,s,a,sā²)
TR(s,a) = trans_reward(type,š,dim,fire,fire_all,s,a)
š« = MDP(Ī³,š®,š,T,R,TR)
N = Dict{Tuple{Int64,Int64},Int64}()
Q = Dict{Tuple{Int64,Int64},Float64}()
d = 50
m = 2000
c = 2

function value_fcn(s,š,R)
    š = possible_action(s,š,R)
    action = argmax(a->Ļ.Q[(s,a)], š)
    return Ļ.Q[(s, action)]
end

# U(s) = value_fcn(s,š,š«.R)
# Ļ = MonteCarloTreeSearch(š«,N,Q,d,m,c,U)

function propagate(s, fire, fire_all, reward_matrix)
    i = 1
    print("Fire", fire_all)
    path = [s]
    while !any(s .== fire_all)
        a = Ļ(s)
        s, _ = trans_reward(type,š,dim,fire,fire_all,s,a)
        append!(path, s)
        i += 1
        print(i,",",a,",",s,"\n")
        # if i == 2 break end
    end
    return s, path
end

# s = 2001
# s, path1 = propagate(s, fire, fire_all, reward_matrix)

# fire, fire_all, reward_matrix = remove_fire(s, fire, fire_all, reward_matrix)
# R(s,a) = reward_matrix[s]
# T(s,a,sā²) = trans_prob(type,dim,fire,s,a,sā²)
# TR(s,a) = trans_reward(type,š,dim,fire,fire_all,s,a)
# š« = MDP(Ī³,š®,š,T,R,TR)
# N = Dict{Tuple{Int64,Int64},Int64}()
# Q = Dict{Tuple{Int64,Int64},Float64}()
# U(s) = value_fcn(s, š, š«.R) 
# Ļ = MonteCarloTreeSearch(š«,N,Q,d,m,c,U)

# s, path2 = propagate(s, fire, fire_all, reward_matrix)

fire = [fire3]
fire_all = [fire3,fire3+400,fire3+800]
reward_matrix[[fire1,fire1+400,fire1+800]] .= 0
reward_matrix[[fire2,fire2+400,fire2+800]] .= 0
s = 2337
# fire, fire_all, reward_matrix = remove_fire(s, fire, fire_all, reward_matrix)
R(s,a) = reward_matrix[s]
T(s,a,sā²) = trans_prob(type,dim,fire,s,a,sā²)
TR(s,a) = trans_reward(type,š,dim,fire,fire_all,s,a)
š« = MDP(Ī³,š®,š,T,R,TR)
N = Dict{Tuple{Int64,Int64},Int64}()
Q = Dict{Tuple{Int64,Int64},Float64}()
U(s) = value_fcn(s, š, š«.R)
Ļ = MonteCarloTreeSearch(š«,N,Q,d,m,c,U)

s, path3 = propagate(s, fire, fire_all, reward_matrix)
# path = [path1, path2, path3]

# CSV.write("MonteCarlo.csv", Tables.table(path), writeheader=false)