%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% THIS IS AN EXPERIMENT TO CREATE AND TRAIN A GENERALIZED REGRESSION 
% NEURAL NETWORK TO ADDRESS THE XOR CLASSIFICATION PROBLEM.
% BORROWED FROM : (source) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all; 
clear all; 
clc;

%%%%% GENERATE INPUT DATA %%%%%%%
% number of samples of each cluster
K = 100;
% offset of clusters
q = .6;
% define 2 groups of input data
A = [rand(1,K)-q rand(1,K)+q;
rand(1,K)+q rand(1,K)-q];
B = [rand(1,K)+q rand(1,K)-q;
rand(1,K)+q rand(1,K)-q];
% plot data
plot(A(1,:),A(2,:),'k+',B(1,:),B(2,:),'b*');
grid on;
hold on;

%%%%% DEFINE OUTPUT CODING %%%%%%%
% coding (+1/-1) for 2-class XOR problem
a = -1;
b = 1;

%%%%% PREPARE INPUT/OUTPUT FOR NETWORK TRAINING %%%%%%%
% define inputs (combine samples from all four classes)
P = [A B];
% define targets
T = [repmat(a,1,length(A)) repmat(b,1,length(B))];



%%%%% DEFINE GENERALIZED REGRESSIVE NEURAL NETWORK %%%%%%%
% choose a spread constant
spread = .2;
% create a neural network
net = newgrnn(P,T,spread);
% view network
view(net);


%%%%%%%%% EVALUATE NETWORK PERFORMANCE %%%%%%
% simulate GRNN on training data
Y = net(P);
% calculate [%] of correct classifications
correct = 100 * length(find(T.*Y > 0)) / length(T);
fprintf('\nSpread = %.2f\n',spread)
fprintf('Num of neurons = %d\n',net.layers{1}.size)
fprintf('Correct class = %.2f %%\n',correct)
% plot targets and network response
figure;
plot(T');
hold on;
grid on;
plot(Y','r');
ylim([-2 2]);
set(gca,'ytick',[-2 0 2]);
legend('Targets','Network response');
xlabel('Sample No.');


%%%%% PLOT CLASSIFICATION RESULTS %%%%%%
% generate a grid
span = -1:.025:2;
[P1,P2] = meshgrid(span,span);
pp = [P1(:) P2(:)]';
% simualte neural network on a grid
aa = sim(net,pp);
% plot classification regions based on MAX activation
figure(1)
ma = mesh(P1,P2,reshape(-aa,length(span),length(span))-5);
mb = mesh(P1,P2,reshape( aa,length(span),length(span))-5);
set(ma,'facecolor',[1 0.2 .7],'linestyle','none');
set(mb,'facecolor',[1 1.0 .5],'linestyle','none');
view(2)
% plot GRNN centers
figure; plot(net.iw{1}(:,1),net.iw{1}(:,2),'gs');

%%%%%%%%% 