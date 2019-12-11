function [TPR, FPR, AUC] = ROCCurve(pred)
% 计算TPR FPR AUC, 画出ROC曲线(仅针对二分类问题)
% pred: 预测概率（根据概率大小已经排序）对应的真实值  列向量

% 返回结果: TPR FPR AUC
% 纵轴: 真正例率 TPR = (TP)/(TP+FN). 真正例/（真正例 + 假反例）
% 横轴: 假正例率 FPR = (FP)/(FP+TN). 假正例/（假正例 + 真反例）

m = size(pred, 1); % 样本个数
% m+1:表示过原点
TPR = zeros(m + 1, 1);
FPR = zeros(m + 1, 1);
AUC = 0;

for i = 1:m
    % 顺序逐个把样本作为正例进行预测
    predi = [ones(i, 1); zeros(m-i, 1)];
    % 建立混淆矩阵计算 TP FP FN 
    tab = confusionmat(predi , pred);
    tab = tab(:);
    TPR(i+1) = tab(4) / sum(tab([3 4]));
    FPR(i+1) = tab(2) / sum(tab([1 2]));
    % compute AUC
    AUC = AUC + (1/2) * (TPR(i+1) + TPR(i)) * (FPR(i+1) - FPR(i));
end
figure();
plot(FPR, TPR, 'r');
xlabel('假正例率');
ylabel('真正例率');
title('ROC 曲线');
text(0.5, 0.5, strcat('AUC = ', num2str(AUC)));
hold on;
% 插入标准线
plot(0:0.01:1, 0:0.01:1, 'blue');
hold off;



