function [TPR, FPR, AUC] = ROCCurve(pred)
% ����TPR FPR AUC, ����ROC����(����Զ���������)
% pred: Ԥ����ʣ����ݸ��ʴ�С�Ѿ����򣩶�Ӧ����ʵֵ  ������

% ���ؽ��: TPR FPR AUC
% ����: �������� TPR = (TP)/(TP+FN). ������/�������� + �ٷ�����
% ����: �������� FPR = (FP)/(FP+TN). ������/�������� + �淴����

m = size(pred, 1); % ��������
% m+1:��ʾ��ԭ��
TPR = zeros(m + 1, 1);
FPR = zeros(m + 1, 1);
AUC = 0;

for i = 1:m
    % ˳�������������Ϊ��������Ԥ��
    predi = [ones(i, 1); zeros(m-i, 1)];
    % ��������������� TP FP FN 
    tab = confusionmat(predi , pred);
    tab = tab(:);
    TPR(i+1) = tab(4) / sum(tab([3 4]));
    FPR(i+1) = tab(2) / sum(tab([1 2]));
    % compute AUC
    AUC = AUC + (1/2) * (TPR(i+1) + TPR(i)) * (FPR(i+1) - FPR(i));
end
figure();
plot(FPR, TPR, 'r');
xlabel('��������');
ylabel('��������');
title('ROC ����');
text(0.5, 0.5, strcat('AUC = ', num2str(AUC)));
hold on;
% �����׼��
plot(0:0.01:1, 0:0.01:1, 'blue');
hold off;



