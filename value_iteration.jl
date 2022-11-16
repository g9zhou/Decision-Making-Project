include("setup.jl")

struct ValueFunctionPolicy
    𝒫
    U
end

function lookahead(𝒫::MDP,U,s,a)
    𝒮,T,R,γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    reachable_𝒮 = reachable_state(s)
    return R(s,a) + γ*sum(T(s,a,s′)*U[s′] for (i,s′) in enumerate(reachable_𝒮))
end

function backup(𝒫::MDP, U, s)
    return maximum(lookahead(𝒫,U,s,a) for a in 𝒫.𝒜)
end

function greedy(𝒫::MDP,U,s)
    u,a = findmax(a->lookahead(𝒫,U,s,a),𝒫.𝒜)
    return (a=a,u=u)
end

(π::ValueFunctionPolicy)(s) = greedy(π.𝒫,π.U,s).a

function solve(M::ValueIteration,𝒫::MDP)
    U = [0.0 for s in 𝒫.𝒮]
    U_prev = copy(U)
    converge = false
    i = 0
    while !converge
        i += 1
        print(i,"\n")
        U = [backup(𝒫,U,s) for s in 𝒫.𝒮]
        if maximum(abs.(U-U_prev)) < M.convergence
            converge = true
        end
        # if i == 1 break end
        # plot1 = heatmap(U,show=true)
        # display(plot1)
        print(maximum(abs.(U-U_prev)),"\n")
        U_prev = copy(U)
    end
    return ValueFunctionPolicy(𝒫,U)
end


###############
M = ValueIteration(10^(-3))
T(s,a,s′) = trans_prob(M,dim,fire,s,a,s′)
𝒫 = MDP(γ,𝒮,𝒜,T,R,nothing)
Result = solve(M,𝒫)

function trajectory_real(Result,fire,fire_all,reward_matrix)
    i = 0
    s = 2001
    path = [s]
    value = []
    append!(value,[Result])
    while true
        a = Result(s)
        s = transition(M,𝒜,dim,fire,s,a)
        append!(path,s)
        if any(s .== fire_all)
            i += 1
            # print(i,"\n")
            if i == 3 break end
            fire, fire_all, reward_matrix = remove_fire(s,fire,fire_all,reward_matrix)
            # print(fire,"\n")
            R(s,a) = reward_matrix[s]
            T(s,a,s′) = trans_prob(M,dim,fire,s,a,s′)
            𝒫 = MDP(γ,𝒮,𝒜,T,R,nothing)
            Result = solve(M,𝒫)
            append!(value,[Result])
        end
    end
    path_xyz = []
    for point in path
        print(point)
        push!(path_xyz,CartesianIndices((dim,dim,dim))[point])
    end
    return path_xyz, value
end

path, value = trajectory_real(Result,fire,fire_all,reward_matrix)

ps = [Point3f(y,x,z) for z in 1:1:20 for x in 1:1:20 for y in 1:1:20]
ns_1 = extract_policy(value[1].U)
length_1 = reshape(value[1].U,(8000))
arrows(ps,0.4*ns_1,fxaa=true,color=length_1,linewidth=0.1,arrowsize = Vec3f(0.2,0.2,0.2),align = :center, axis=(type=Axis3,))
x,y,z = plot_actual_path(path[1:32])
x_op,y_op,z_op = plot_optimal_path(ns_1,2001,1258)
plot3d(y,x,z,lw=2,camera = (45,60),label = "Actual Trajectory")
plot3d(y_op,x_op,z_op,lw=2, camera = (45,45), label = "First Trajectory")
plot3d!(xlims=(0,20),ylims=(0,20),zlims=(0,20),aspect_ratio=:equal,xlable="x",ylabel="y",zlabel="z",title="First Trajectory")

ns_2 = extract_policy(value[2].U)
length_2 = reshape(value[2].U,(8000))
arrows(ps,0.4*ns_2,fxaa=true,color=length_2,linewidth=0.1,arrowsize = Vec3f(0.2,0.2,0.2),align = :center, axis=(type=Axis3,))
x,y,z = plot_actual_path(path[32:55])
x_op,y_op,z_op = plot_optimal_path(ns_2,1258,3137)
plot3d(y,x,z,lw=2,camera = (45,60),label = "Actual Trajectory")
plot3d!(y_op,x_op,z_op,lw=2,camera = (45,60), label = "Second Trajectory")
plot3d!(xlims=(0,20),ylims=(0,20),zlims=(0,20),aspect_ratio=:equal,xlable="x",ylabel="y",zlabel="z",title="Second Trajectory")

ns_3 = extract_policy(value[3].U)
length_3 = reshape(value[3].U,(8000))
arrows(ps,0.4*ns_3,fxaa=true,color=length_3,linewidth=0.1,arrowsize = Vec3f(0.2,0.2,0.2),align = :center, axis=(type=Axis3,))
x,y,z = plot_actual_path(path[55:end])
x_op,y_op,z_op = plot_optimal_path(ns_3,3137,7543)
plot3d(y,x,z,lw=2,camera = (45,60),label = "Actual Trajectory",legend=:bottomright)
plot3d!(y_op,x_op,z_op,lw=2, camera = (45,60), label = "Third Trajectory",legend=:bottomright)
plot3d!(xlims=(0,20),ylims=(0,20),zlims=(0,20),aspect_ratio=:equal,xlable="x",ylabel="y",zlabel="z",title="Third Trajectory")
# CSV.write("ValueIteration.csv", Tables.table(Result.U), writeheader=false)