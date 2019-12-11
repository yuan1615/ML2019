function [P, R, F, bound] = PRCurve(pred)
% 计算P R, 画出PR曲线(仅针对二分类问题)
% pred: 预测概率（根据概率大小已经排序）对应的真实值  列向量

% 返回结果: PR曲线
% 纵轴: 查准率 P = (TP)/(TP+FP). 真正例/（真正例 + 假正例）
% 横轴: 查全率 R = (TP)/(TP+FN). 真正例/（真正例 + 假反例）
m = size(pred, 1); % 样本个数
P = zeros(m, 1);
R = zeros(m, 1);

for i = 1:m
    % 顺序逐个把样本作为正例进行预测
    predi = [ones(i, 1); zeros(m-i, 1)];
    % 建立混淆矩阵计算 TP FP FN 
    [tab, order] = confusionmat(predi , pred);
    tab = tab(:);
    P(i) = tab(4) / sum(tab([2 4]));
    R(i) = tab(4) / sum(tab([3 4]));
end
F = 2 .* (P.*R) ./ (P + R);
[F bound] = max(F);

figure();
plot(R, P, 'r')
xlabel('查全率');
ylabel('查准率')
title('PR 曲线')
text(0.5, 0.5, strcat('F score = ', num2str(F)));
hold on;
% 插入平衡点
plot(0:0.01:1, 0:0.01:1, 'blue');
hold off;



