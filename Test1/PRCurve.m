function [P, R, F, bound] = PRCurve(pred)
% ����P R, ����PR����(����Զ���������)
% pred: Ԥ����ʣ����ݸ��ʴ�С�Ѿ����򣩶�Ӧ����ʵֵ  ������

% ���ؽ��: PR����
% ����: ��׼�� P = (TP)/(TP+FP). ������/�������� + ��������
% ����: ��ȫ�� R = (TP)/(TP+FN). ������/�������� + �ٷ�����
m = size(pred, 1); % ��������
P = zeros(m, 1);
R = zeros(m, 1);

for i = 1:m
    % ˳�������������Ϊ��������Ԥ��
    predi = [ones(i, 1); zeros(m-i, 1)];
    % ��������������� TP FP FN 
    [tab, order] = confusionmat(predi , pred);
    tab = tab(:);
    P(i) = tab(4) / sum(tab([2 4]));
    R(i) = tab(4) / sum(tab([3 4]));
end
F = 2 .* (P.*R) ./ (P + R);
[F bound] = max(F);

figure();
plot(R, P, 'r')
xlabel('��ȫ��');
ylabel('��׼��')
title('PR ����')
text(0.5, 0.5, strcat('F score = ', num2str(F)));
hold on;
% ����ƽ���
plot(0:0.01:1, 0:0.01:1, 'blue');
hold off;



