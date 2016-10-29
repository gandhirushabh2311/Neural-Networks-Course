%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% THIS IS AN EXPERIMENT TO CREATE AND TRAIN A ADALINE NETWORK 
% TO ADDRESS THE TIME SERIES PREDICTION PROBLEM.
% BORROWED FROM : (source) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all; 
clear all; 
clc;

%%%%% CREATE INPUT DATA FOR TIME SERIES PREDICTION PROBLEM %%%%%%%
% define segments of time vector
dt = 0.01; % time step [seconds]
t1 = 0 : dt : 3; % first time vector [seconds]
t2 = 3+dt : dt : 6; % second time vector [seconds]
t = [t1 t2]; % complete time vector [seconds]
% define signal
y = [sin(4.1*pi*t1) .8*sin(8.3*pi*t2)];
% plot signal
plot(t,y,'.-')
xlabel('Time [sec]');
ylabel('Target Signal');
grid on
ylim([-1.2 1.2])

%%%%% PREPARING DATA FOR NEURAL NETWORK %%%%%%%
% There are two basic types of input vectors: those that occur concurrently
% (at the same time, or in no particular time sequence), and those that
% occur sequentially in time. For concurrent vectors, the order is not
% important, and if there were a number of networks running in parallel,
% you could present one input vector to each of the networks. For
% sequential vectors, the order in which the vectors appear is important.
p = con2seq(y);

%%%%% DEFINE A ADALINE NEURAL NETWORK %%%%%%%
% The resulting network will predict the next value of the target signal
% using delayed values of the target.
inputDelays = 1:5; % delayed inputs to be used
learning_rate = 0.2; % learning rate
% define ADALINE
net = linearlayer(inputDelays,learning_rate);

%%%%% ADAPTIVE LEARNING OF THE ADALINE %%%%%%%
% Given an input sequence with N steps the network is updated as follows.
% Each step in the sequence of inputs is presented to the network one at
% a time. The network's weight and bias values are updated after each step,
% before the next step in the sequence is presented. Thus the network is
% updated N times. The output signal and the error signal are returned,
% along with new network.
[net,Y,E] = adapt(net,p,p);
% view network structure
view(net);
% check final network parameters
disp('Weights of the ADALINE after adaptation')
disp(net.IW{1}); 
disp('Bias of the ADALINE after adaptation')
disp(net.b{1});

%%%%%%%%% PLOT RESULTS %%%%%%

% transform result vectors
Y = seq2con(Y); Y = Y{1};
E = seq2con(E); E = E{1};
% start a new figure
figure;
% first graph
subplot(211);
plot(t,y,'b', t,Y,'r--');
legend('Original','Prediction');
grid on;
xlabel('Time [sec]');
ylabel('Target Signal');
ylim([-1.2 1.2]);
% second graph
subplot(212);
plot(t,E,'g');
grid on; 
legend('Prediction error');
xlabel('Time [sec]');
ylabel('Error');
ylim([-1.2 1.2]);