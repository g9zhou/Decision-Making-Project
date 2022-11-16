include("setup.jl")

data = CSV.read("SARSA_Q1_motion_restricted.csv",DataFrame,header=false)
D = Matrix{Float64}(data)
idx = findall(x->x==0,D)
D[idx] .= -1000

function value_extract_policy(Q)
    policy = []
    value = []
    for row in 1:8000
        action = argmax(Q[row,:])
        
        if Q[row,action] == -1000 action = 5 end
        
        direc = []
        if action == 1 direc = Vec3f(-1.0,0.0,0.0) end
        if action == 2 direc = Vec3f(1.0,0.0,0.0)end
        if action == 3 direc = Vec3f(0.0,-1.0,0.0) end
        if action == 4 direc = Vec3f(0.0,1.0,0.0) end
        if action == 5 direc = Vec3f(0.0,0.0,1.0) end
        if action == 6 direc = Vec3f(0.0,0.0,-1.0) end
        append!(policy,[direc])
        append!(value,Q[row,action])
    end
    return policy
end

ps = [Point3f(y,x,z) for z in 1:1:20 for x in 1:1:20 for y in 1:1:20]
ns_1 = value_extract_policy(D)
arrows(ps,0.4*ns_1,fxaa=true,color=length_1,linewidth=0.1,arrowsize = Vec3f(0.2,0.2,0.2),align = :center, axis=(type=Axis3,))

function plot_path(path,s)
    s_y, s_x, s_z = CartesianIndices((dim,dim,dim))[s][1],CartesianIndices((dim,dim,dim))[s][2],CartesianIndices((dim,dim,dim))[s][3]
    x = [s_x]
    y = [s_y]
    z = [s_z]
    for i in 1:20
        s_y, s_x, s_z = CartesianIndices((dim,dim,dim))[path[i]][1],CartesianIndices((dim,dim,dim))[path[i]][2],CartesianIndices((dim,dim,dim))[path[i]][3]
        append!(x,s_x)
        append!(y,s_y)
        append!(z,s_z)
    end
    return x,y,z
end
