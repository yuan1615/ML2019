clear
clc
close all
% �������Ԥ��ֵ����ʵֵ
% ������
pred = [1 1 0 1 1 0 0 0 1 0]';
% ��������
pred = [ones(100,1); 
    round(rand(200, 1) ./ 2 + 0.45);
    round(rand(400, 1) ./ 2 + 0.25);
    round(rand(200, 1) ./2 + 0.05);
    zeros(100, 1)];
% ��ҵ����
% pred = [round(rand(500, 1) ./ 2 + 0.45);
%     round(rand(500, 1) ./ 2 + 0.1)];

% ��PR����
[P, R, F, bound] = PRCurve(pred);

% ��ROC���߲��Ҽ���AUC
[TPR, FPR, AUC] = ROCCurve(pred);

