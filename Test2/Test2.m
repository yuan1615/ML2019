% Test2
%% Initialization
clear ; close all; clc
%% ==================== Part 1: Plotting ====================
data = csvread('西瓜数据3.0a.csv');
X = data(:, [1, 2]); y = data(:, 3);

plotData(X, y);
% Put some labels 
hold on;
% Labels and Legend
xlabel('密度')
ylabel('含糖量')
legend('好瓜', '坏瓜', 'location', 'northwest')
hold off;

%% ============= Part 3: Optimizing 1 =============
%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);
% Add intercept term to x and X_test
X = [ones(m, 1) X];
initial_theta = zeros(n + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 100);
[theta, cost] = ...
	fmincg(@(t)(costFunction(t, X, y)), initial_theta, options);

% Plot Boundary
plotDecisionBoundary(theta, X, y);
% Put some labels 
hold on;
% Labels and Legend
xlabel('密度')
ylabel('含糖量')

%% ============= Part 4: Optimizing 2  =============
data = csvread('西瓜数据3.0a.csv');
X = data(:, [1, 2]); y = data(:, 3);

X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 0.0001;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fmincg(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('密度')
ylabel('含糖量')
legend('好瓜', '坏瓜', 'location', 'northwest')
hold off;
