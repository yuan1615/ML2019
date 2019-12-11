clear
clc
close all
% 随机生成预测值及真实值
% 简单例子
pred = [1 1 0 1 1 0 0 0 1 0]';
% 划分区间
pred = [ones(100,1); 
    round(rand(200, 1) ./ 2 + 0.45);
    round(rand(400, 1) ./ 2 + 0.25);
    round(rand(200, 1) ./2 + 0.05);
    zeros(100, 1)];
% 作业案例
% pred = [round(rand(500, 1) ./ 2 + 0.45);
%     round(rand(500, 1) ./ 2 + 0.1)];

% 画PR曲线
[P, R, F, bound] = PRCurve(pred);

% 画ROC曲线并且计算AUC
[TPR, FPR, AUC] = ROCCurve(pred);

