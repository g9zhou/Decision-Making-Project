include("setup.jl")

struct MonteCarloTreeSearch
    𝒫
    N
    Q
    d
    m
    c
    U
end

function(π::MonteCarloTreeSearch)(s)
    for k in 1:π.m
        simulate!(π,s)
    end
    𝒜 = possible_action(s,π.𝒫.𝒜,π.𝒫.R)
    # print(𝒜)
    action = argmax(a->π.Q[(s,a)],𝒜)
    print(π.Q[(s, action)])
    # if π.Q[(s, action)] == 0.0
    #     if length(𝒜) > 1
    #         # if 𝒜[3] == 4 return 6 end
    #         return 𝒜[3]
    #     end
    #     return 𝒜[1]
    # end
    return argmax(a->π.Q[(s,a)],𝒜)
end

function simulate!(π::MonteCarloTreeSearch, s, d=π.d)
    if d ≤ 0
        return 0.0
    end
    𝒫, N, Q, c = π.𝒫, π.N, π.Q, π.c
    𝒜, R, TR, γ = 𝒫.𝒜, 𝒫.R, 𝒫.TR, 𝒫.γ
    # print(𝒫.𝒜,"\n")
    # 𝒜 = possible_action(s,𝒜,R)
    if !haskey(N, (s,first(𝒜)))
        for a in 𝒜
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return π.U(s)
    end
    a = explore(π,s)
    s′, r = TR(s,a)
    q = r + γ*simulate!(π,s′,d-1)
    N[(s,a)] += 1
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]
    # if R(s,0) == -10 return 0.0 end
    return q
end

bonus(Nsa,Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)

function explore(π::MonteCarloTreeSearch,s)
    𝒜, R, N, Q, c = π.𝒫.𝒜, π.𝒫.R, π.N, π.Q, π.c
    𝒜 = possible_action(s,𝒜,R)
    Ns = sum(N[(s,a)] for a in 𝒜)
    return argmax(a->Q[(s,a)] + c*bonus(N[(s,a)], Ns), 𝒜)
end

type = MonteCarlo()
T(s,a,s′) = trans_prob(type,dim,fire,s,a,s′)
TR(s,a) = trans_reward(type,𝒜,dim,fire,fire_all,s,a)
𝒫 = MDP(γ,𝒮,𝒜,T,R,TR)
N = Dict{Tuple{Int64,Int64},Int64}()
Q = Dict{Tuple{Int64,Int64},Float64}()
d = 50
m = 2000
c = 2

function value_fcn(s,𝒜,R)
    𝒜 = possible_action(s,𝒜,R)
    action = argmax(a->π.Q[(s,a)], 𝒜)
    return π.Q[(s, action)]
end

# U(s) = value_fcn(s,𝒜,𝒫.R)
# π = MonteCarloTreeSearch(𝒫,N,Q,d,m,c,U)

function propagate(s, fire, fire_all, reward_matrix)
    i = 1
    print("Fire", fire_all)
    path = [s]
    while !any(s .== fire_all)
        a = π(s)
        s, _ = trans_reward(type,𝒜,dim,fire,fire_all,s,a)
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
# T(s,a,s′) = trans_prob(type,dim,fire,s,a,s′)
# TR(s,a) = trans_reward(type,𝒜,dim,fire,fire_all,s,a)
# 𝒫 = MDP(γ,𝒮,𝒜,T,R,TR)
# N = Dict{Tuple{Int64,Int64},Int64}()
# Q = Dict{Tuple{Int64,Int64},Float64}()
# U(s) = value_fcn(s, 𝒜, 𝒫.R) 
# π = MonteCarloTreeSearch(𝒫,N,Q,d,m,c,U)

# s, path2 = propagate(s, fire, fire_all, reward_matrix)

fire = [fire3]
fire_all = [fire3,fire3+400,fire3+800]
reward_matrix[[fire1,fire1+400,fire1+800]] .= 0
reward_matrix[[fire2,fire2+400,fire2+800]] .= 0
s = 2337
# fire, fire_all, reward_matrix = remove_fire(s, fire, fire_all, reward_matrix)
R(s,a) = reward_matrix[s]
T(s,a,s′) = trans_prob(type,dim,fire,s,a,s′)
TR(s,a) = trans_reward(type,𝒜,dim,fire,fire_all,s,a)
𝒫 = MDP(γ,𝒮,𝒜,T,R,TR)
N = Dict{Tuple{Int64,Int64},Int64}()
Q = Dict{Tuple{Int64,Int64},Float64}()
U(s) = value_fcn(s, 𝒜, 𝒫.R)
π = MonteCarloTreeSearch(𝒫,N,Q,d,m,c,U)

s, path3 = propagate(s, fire, fire_all, reward_matrix)
# path = [path1, path2, path3]

# CSV.write("MonteCarlo.csv", Tables.table(path), writeheader=false)