%% Initialization
clear ; close all; clc

%% =============== Part 1: Creat and Visualizing Data ================
% 创建一个中心为（3，6）方差为1的正态分布点（10）个样本
% 创建一个中心为（6，3）方差为1的正态分布点（10）个样本
rng('default');
rng(1);
X1 = mvnrnd([3; 6], eye(2), 10);
Y1 = ones(10, 1);
rng(1);
X2 = mvnrnd([6; 3], eye(2), 10);
Y2 = zeros(10, 1);
X = [X1; X2];
y = [Y1; Y2];
figure()
plotData(X, y)
%% ==================== Part 2: Training Linear SVM ====================

C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
figure()
visualizeBoundaryLinear(X, y, model);
fprintf('Program paused. Press enter to continue.\n');

%% =============== Part 3: Implementing Gaussian Kernel ===============
rng('default');
rng(1);
X1 = mvnrnd([3; 6], eye(2), 100);
Y1 = ones(100, 1);
rng(1);
X2 = mvnrnd([6; 3], eye(2), 100);
Y2 = zeros(100, 1);
X = [X1; X2];
y = [Y1; Y2];
figure()
plotData(X, y)

% SVM Parameters
C = 1; sigma = 0.1;

model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
figure()
visualizeBoundary(X, y, model);
fprintf('Program paused. Press enter to continue.\n');
