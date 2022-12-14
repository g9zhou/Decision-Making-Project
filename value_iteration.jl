include("setup.jl")

struct ValueFunctionPolicy
    š«
    U
end

function lookahead(š«::MDP,U,s,a)
    š®,T,R,Ī³ = š«.š®, š«.T, š«.R, š«.Ī³
    reachable_š® = reachable_state(s)
    return R(s,a) + Ī³*sum(T(s,a,sā²)*U[sā²] for (i,sā²) in enumerate(reachable_š®))
end

function backup(š«::MDP, U, s)
    return maximum(lookahead(š«,U,s,a) for a in š«.š)
end

function greedy(š«::MDP,U,s)
    u,a = findmax(a->lookahead(š«,U,s,a),š«.š)
    return (a=a,u=u)
end

(Ļ::ValueFunctionPolicy)(s) = greedy(Ļ.š«,Ļ.U,s).a

function solve(M::ValueIteration,š«::MDP)
    U = [0.0 for s in š«.š®]
    U_prev = copy(U)
    converge = false
    i = 0
    while !converge
        i += 1
        print(i,"\n")
        U = [backup(š«,U,s) for s in š«.š®]
        if maximum(abs.(U-U_prev)) < M.convergence
            converge = true
        end
        # if i == 1 break end
        # plot1 = heatmap(U,show=true)
        # display(plot1)
        print(maximum(abs.(U-U_prev)),"\n")
        U_prev = copy(U)
    end
    return ValueFunctionPolicy(š«,U)
end


###############
M = ValueIteration(10^(-3))
T(s,a,sā²) = trans_prob(M,dim,fire,s,a,sā²)
š« = MDP(Ī³,š®,š,T,R,nothing)
Result = solve(M,š«)

function trajectory_real(Result,fire,fire_all,reward_matrix)
    i = 0
    s = 2001
    path = [s]
    value = []
    append!(value,[Result])
    while true
        a = Result(s)
        s = transition(M,š,dim,fire,s,a)
        append!(path,s)
        if any(s .== fire_all)
            i += 1
            # print(i,"\n")
            if i == 3 break end
            fire, fire_all, reward_matrix = remove_fire(s,fire,fire_all,reward_matrix)
            # print(fire,"\n")
            R(s,a) = reward_matrix[s]
            T(s,a,sā²) = trans_prob(M,dim,fire,s,a,sā²)
            š« = MDP(Ī³,š®,š,T,R,nothing)
            Result = solve(M,š«)
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