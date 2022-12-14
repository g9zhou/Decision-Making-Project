include("setup.jl")

mutable struct SarsaLambda
    š®
    š
    Ī³
    Q
    N
    Ī±
    Ī»
    š
end


function sampling(š«::MDP,s)
    s_origin = copy(s)
    M = Matrix{Int}(undef,1,4)
    for i in 1:25
        print(i)
        while !any(s .== fire_all)
            š = possible_action(s,š«.š,š«.R)
            a = sample(š,Weights(ones(length(š))))
            sā², r = š«.TR(s,a)
            M = cat(M,[s a sā² r],dims=1)
            s = sā²
        end
        s = s_origin
    end
    return M[2:end,:]
end
        

lookahead(model::SarsaLambda,s,a) = model.Q[s,a]

function update!(model::SarsaLambda,s,a,r,sā²)
    if model.š ā  nothing
        Ī³, Ī», Q, Ī±, š = model.Ī³, model.Ī», model.Q, model.Ī±, model.š
        model.N[š.s,š.a] += 1
        Ī“ = š.r + Ī³*Q[s,a] - Q[š.s,š.a]
        model.Q .+= Ī±*Ī“*model.N
        model.N .*= Ī³*Ī»
    else
        model.N[:,:] .= 0.0
    end
    model.š = (s=s,a=a,r=r)
    return model
end

type = SARSA()
T(s,a,sā²) = trans_prob(type,dim,fire,s,a,sā²)
TR(s,a) = trans_reward(type,š,R,dim,fire,s,a)
š« = MDP(Ī³,š®,š,T,R,TR)
Q = zeros(length(š«.š®),length(š«.š))
N = zeros(length(š«.š®),length(š«.š))
Ī± = 0.1 #learning rate
Ī» = 0.9
model = SarsaLambda(š«.š®, š«.š, š«.Ī³, Q, N, Ī±, Ī», nothing)

function compute(model)
    M = sampling(š«,2001)
    ndata = size(M,1)
    converge = false
    i = 0
    s, a, r, sā² = M[:,1], M[:,2], M[:,3], M[:,4]
    while !converge
        i += 1
        print(i,"\n")
        Q_prev = copy(model.Q)
        for row in 1:ndata
            model = update!(model,s[row],a[row],r[row],sā²[row])
        end
        print(maximum(abs.(model.Q-Q_prev)),"\n")
        if maximum(abs.(model.Q-Q_prev)) < 10^(-3)
            converge = true
        end
    end
    return model
end
model = compute(model)