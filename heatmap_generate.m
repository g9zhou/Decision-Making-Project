clear all;
close all;
clc;

[X,Y,Z] = meshgrid(1:20,1:20,1:20);
U = readtable("landscape.csv");
data = reshape(U{:,:},[20,20,20]);
mymap = [0 1 0;
        1 0 0;
        1 1 1]
colormap(mymap)
plt = slice(X,Y,Z,data,[1:20],[1:20],[1:20]);
alpha(plt,0.05)