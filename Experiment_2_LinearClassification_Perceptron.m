%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% THIS IS AN EXPERIMENT TO CREATE AND TRAIN A PERCEPTRON NETWORK 
% TO ADDRESS THE LINEAR CLASSIFICATION PROBLEM.
% BORROWED FROM : (source) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all; 
clear all; 
clc;

%%%%% CREATE INPUT DATA FOR LINEAR CLASSIFICATION PROBLEM %%%%%%%

% number of samples of each class
N = 20;
% define inputs and outputs
offset = 5; % offset for second class
x = [randn(2,N) randn(2,N)+offset]; % inputs
y = [zeros(1,N) ones(1,N)]; % outputs
% Plot input samples with PLOTPV (Plot perceptron input/target vectors)
figure(1)
plotpv(x,y);
grid on;
%%%%% DEFINE INPUT VECTOR %%%%%%%
p = [2 3];

%%%%% CREATE AND TRAIN THE PERCEPTRON NETWORK %%%%%%%

net = perceptron;
net = train(net,x,y);
view(net);

%%%%% PLOT THE OUTPUT DECISION BOUNDARY %%%%%%%
figure(1)
plotpc(net.IW{1},net.b{1});