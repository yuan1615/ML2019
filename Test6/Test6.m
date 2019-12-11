%������Psi��Ӧ��ά��˹���������е�(x-��)����ת�õ�˳���������ᵽ�ĸ�˹������ͬ��������Ϊ�˱�֤����ɳ��ԣ���Ӱ����.
%Gamma Ϊ��������ֵ��Gamma(i,j)�����i���������ڵ�j��ģ�͵ĸ��ʡ�
%MuΪ������SigmaΪЭ�������%PiΪ��ģ�͵�Ȩֵϵ��%����2����ά��̬����
%����2����ά��̬����
MU1 = [1 2];
SIGMA1 = [1 0; 0 0.5];
MU2 = [-1 -1];
SIGMA2 = [1 0; 0 1];
%����1000��2��(Ĭ��)��ֵΪmu��׼��Ϊsigma����̬�ֲ������
X = [mvnrnd(MU1, SIGMA1, 1000);mvnrnd(MU2, SIGMA2, 1000)];
%��ʾ
scatter(X(:,1),X(:,2),10,'.');
%====================
K=2;
[N,D]=size(X);
Gamma=zeros(N,K);
Psi=zeros(N,K);
Mu=zeros(K,D);
LM=zeros(K,D);
Sigma =zeros(D, D, K); 
Pi=zeros(1,K);
%ѡ�������������������Ϊ����������ֵ
Mu(1,:)=X(randi([1 N/2],1,1),:);
Mu(2,:)=X(randi([N/2 N],1,1),:); 
%�������ݵ�Э������ΪЭ�����ֵ
for k=1:K
  Pi(k)=1/K;
  Sigma(:, :, k)=cov(X);
end
LMu=Mu;
LSigma=Sigma;
LPi=Pi;
while true
%Estimation Step  
for k = 1:K
  Y = X - repmat(Mu(k,:),N,1);
  Psi(:,k) = (2*pi)^(-D/2)*det(Sigma(:,:,k))^(-1/2)*diag(exp(-1/2*Y/(Sigma(:,:,k))*(Y')));      %Psi��һ�д����һ����˹�ֲ����������ݵ�ȡֵ
end
Gamma_SUM=zeros(D,D);
for j = 1:N
  for k=1:K
  Gamma(j,k) = Pi(1,k)*Psi(j,k)/sum(Psi(j,:)*Pi');                                               %Psi�ĵ�һ�зֱ����������˹�ֲ��Ե�һ�����ݵ�ȡֵ
  end
end
%Maximization Step
for k = 1:K
%update Mu
  Mu_SUM= zeros(1,D);
  for j=1:N
     Mu_SUM=Mu_SUM+Gamma(j,k)*X(j,:);   
  end
  Mu(k,:)= Mu_SUM/sum(Gamma(:,k));
%update Sigma
 Sigma_SUM= zeros(D,D);
 for j = 1:N
   Sigma_SUM = Sigma_SUM+ Gamma(j,k)*(X(j,:)-Mu(k,:))'*(X(j,:)-Mu(k,:));
 end
 Sigma(:,:,k)= Sigma_SUM/sum(Gamma(:,k));
 %update Pi
 Pi_SUM=0;
 for j=1:N
   Pi_SUM=Pi_SUM+Gamma(j,k);
 end
 Pi(1,k)=Pi_SUM/N;
end

R_Mu=sum(sum(abs(LMu- Mu)));
R_Sigma=sum(sum(abs(LSigma- Sigma)));
R_Pi=sum(sum(abs(LPi- Pi)));
R=R_Mu+R_Sigma+R_Pi;
if (R<1e-10)
    disp('����');
    disp(Mu);
    disp('Э�������');
    disp(Sigma);
    disp('Ȩֵϵ��');
    disp(Pi);
    obj=gmdistribution(Mu,Sigma);
    figure,h = ezmesh(@(x,y)pdf(obj,[x,y]),[-8 6], [-8 6]);
    break;
end
 LMu=Mu;
 LSigma=Sigma;
 LPi=Pi;
end
%=====================


%%%%%%%%���н��%%%%%%%
%
% ����
%    1.0334    2.0479
%    -0.9703   -0.9879
% 
% Э�������
% 
% (:,:,1) =
% 
%     0.9604   -0.0177
%    -0.0177    0.4463
% 
% 
% (:,:,2) =
% 
%     0.9976    0.0667
%     0.0667    1.0249
% 
% Ȩֵϵ��
%     0.4935    0.5065
%%%%%%%%%%%%%%%%%%%%%%%%%
