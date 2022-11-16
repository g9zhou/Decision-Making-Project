include("setup.jl")

mutable struct SarsaLambda
    𝒮
    𝒜
    γ
    Q
    N
    α
    λ
    𝓁
end


function sampling(𝒫::MDP,s)
    s_origin = copy(s)
    M = Matrix{Int}(undef,1,4)
    for i in 1:25
        print(i)
        while !any(s .== fire_all)
            𝒜 = possible_action(s,𝒫.𝒜,𝒫.R)
            a = sample(𝒜,Weights(ones(length(𝒜))))
            s′, r = 𝒫.TR(s,a)
            M = cat(M,[s a s′ r],dims=1)
            s = s′
        end
        s = s_origin
    end
    return M[2:end,:]
end
        

lookahead(model::SarsaLambda,s,a) = model.Q[s,a]

function update!(model::SarsaLambda,s,a,r,s′)
    if model.𝓁 ≠ nothing
        γ, λ, Q, α, 𝓁 = model.γ, model.λ, model.Q, model.α, model.𝓁
        model.N[𝓁.s,𝓁.a] += 1
        δ = 𝓁.r + γ*Q[s,a] - Q[𝓁.s,𝓁.a]
        model.Q .+= α*δ*model.N
        model.N .*= γ*λ
    else
        model.N[:,:] .= 0.0
    end
    model.𝓁 = (s=s,a=a,r=r)
    return model
end

type = SARSA()
T(s,a,s′) = trans_prob(type,dim,fire,s,a,s′)
TR(s,a) = trans_reward(type,𝒜,R,dim,fire,s,a)
𝒫 = MDP(γ,𝒮,𝒜,T,R,TR)
Q = zeros(length(𝒫.𝒮),length(𝒫.𝒜))
N = zeros(length(𝒫.𝒮),length(𝒫.𝒜))
α = 0.1 #learning rate
λ = 0.9
model = SarsaLambda(𝒫.𝒮, 𝒫.𝒜, 𝒫.γ, Q, N, α, λ, nothing)

function compute(model)
    M = sampling(𝒫,2001)
    ndata = size(M,1)
    converge = false
    i = 0
    s, a, r, s′ = M[:,1], M[:,2], M[:,3], M[:,4]
    while !converge
        i += 1
        print(i,"\n")
        Q_prev = copy(model.Q)
        for row in 1:ndata
            model = update!(model,s[row],a[row],r[row],s′[row])
        end
        print(maximum(abs.(model.Q-Q_prev)),"\n")
        if maximum(abs.(model.Q-Q_prev)) < 10^(-3)
            converge = true
        end
    end
    return model
end
model = compute(model)