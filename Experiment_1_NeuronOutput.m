%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% THIS IS AN EXPERIMENT TO DESIGN A NEURON AND TEST ITS OUTPUT 
% OVER A RANGE OF INPUT VALUES.
% BORROWED FROM : (source) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all; 
clear all; 
clc;

%%%%% DEFINE NEURON PARAMETERS %%%%%%%
% Neuron weights
w = [4 -2];
% Neuron bias
b = -3;
% Activation function
func = 'tansig';
% func = 'purelin'
% func = 'hardlim'
% func = 'logsig'

%%%%% DEFINE INPUT VECTOR %%%%%%%
p = [2 3];

%%%%% CALCULATE NEURON OUTPUT FOR ONE INPUT VECTOR 'p' %%%%%%%
activation_potential = p*w'+b;
neuron_output = feval(func, activation_potential);
disp(strcat('THE NEURON OUTPUT IS --> ',num2str(neuron_output)));

%%%%% CALCULATE NEURON OUTPUT FOR A MESH OF INPUT %%%%%%%
[p1,p2] = meshgrid(-10:.25:10);
z = feval(func, [p1(:) p2(:)]*w'+b );
z = reshape(z,length(p1),length(p2));
plot3(p1,p2,z);
grid on;
xlabel('Input 1');
ylabel('Input 2');
zlabel('Neuron output');
