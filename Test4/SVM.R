##### 支持向量机 #####

# 设置工作路径
setwd("D:/2018--2020-研究生/2、上海工程技术大学/机器学习/Year2019/Test4")

# 加载所需packages
options(stringsAsFactors = FALSE)
library(ggplot2)          # 绘图
library(easyGgplot2)      # 绘图
library(MASS)             # 生成数据
library(CVXR)             # CVX的R语言支持包
library(reshape2)

##### 1、生成数据 #####
# 随机生成以$[3,6]^T$为均值，以单位矩阵为协方差阵的20个样本，设置标签为-1；
# 随机生成以$[6,3]^T$为均值，以单位矩阵为协方差阵的20个样本，设置标签为1（确保两类样本线性可分）。
# 然后按照7：3的比例将样本分为训练集与测试集，并画出测试集图形。

CreatData <- function(n, mu1, mu2, Sigma, seed = 3, alha = 0.7){
  # 生成多元正态分布数据
  
  # Input:
      # n: 数据个数
      # mu1: 均值1
      # mu2: 均值2
      # Sigma: 协方差矩阵，1与2相同
      # seed: 随机数种子
      # alha: 训练集占比
  # Output:
      # 训练集与测试集（数据框格式）
  set.seed(seed)
  X1 <- mvrnorm(n, mu1, Sigma)
  X1 <- data.frame(x1 = X1[, 1], x2 = X1[, 2])
  Y1 <- rep(-1, n)
  set.seed(seed)
  X2 <- mvrnorm(n, mu2, Sigma)
  X2 <- data.frame(x1 = X2[, 1], x2 = X2[, 2])
  Y2 <- rep(1, n)
  X1.train <- X1[1:floor(n*alha), ]
  X1.test <- X1[(floor(n*alha) + 1):n, ]
  X2.train <- X2[1:floor(n*alha), ]
  X2.test <- X2[(floor(n*alha) + 1):n, ]
  Y1.train <- Y1[1:floor(n*alha)]
  Y1.test <- Y1[(floor(n*alha) + 1):n]
  Y2.train <- Y2[1:floor(n*alha)]
  Y2.test <- Y2[(floor(n*alha) + 1):n]
  X.train <- rbind(X1.train, X2.train)
  X.test <- rbind(X1.test, X2.test)
  Y.train <- data.frame(Y = as.factor(c(Y1.train, Y2.train)))
  Y.test <- data.frame(Y = as.factor(c(Y1.test, Y2.test)))
  return(list(data.train = cbind(X.train, Y.train),
              data.test = cbind(X.test, Y.test)))
}
#------ 1.1 线性可分数据及绘图------
data1 <- CreatData(20, c(3, 6), c(6, 3), diag(2))
data1.train <- data1$data.train
data1.test <- data1$data.test
ggplot(data = data1.train, aes(x = x1, y = x2, colour = Y)) + 
  geom_point(size = 2.0, shape = 16)

#------ 1.2 线性不可分数据及绘图------
data2 <- CreatData(100, c(3, 6), c(6, 3), diag(2), seed = 2)
data2.train <- data2$data.train
data2.test <- data2$data.test
ggplot(data = data2.train, aes(x = x1, y = x2, colour = Y)) + 
  geom_point(size = 2.0, shape = 16)


##### 2、SVM 基本形式求解（6.6）#####
X <- as.matrix(data1.train[, 1:2])
Y <- as.matrix(as.numeric(as.character(data1.train[, 3])))

SVMbase <- function(X, Y){
  # 求解SVM基本形式
  # Input：
      # X: 特征
      # Y: 正负1标签
  # Output:
      # W_res: w的解
      # b_res: b的解
  
  w <- Variable(2)
  b <- Variable(1)
  objective <- Minimize(1/2 * norm(w, "F")^2)
  constraints <- list(Y * (X %*% w + b) >= 1)
  prob <- Problem(objective, constraints)
  
  # Problem solution
  solution <- solve(prob)
  w_res <- solution$getValue(w)
  b_res <- solution$getValue(b)
  return(list(w_res = w_res, b_res = b_res))
}

modelbase <- SVMbase(X, Y)
w_res <- modelbase$w_res
b_res <- modelbase$b_res

ggplot(data = data1.train, aes(x = x1, y = x2, colour = Y)) + 
  geom_point(size = 2.0, shape = 16) + 
  geom_abline(slope = -w_res[1]/w_res[2], intercept = -b_res/w_res[2], colour = "red") + 
  labs(title="Train Data") + theme(plot.title = element_text(hjust = 0.5))

#------ 2.2 测试泛化能力---------

predictbase <- function(da, w_res, b_res){
  # 基础SVM预测函数
  # Input：
      # da: 需要预测的数据，包含X  Y
      # w_res,b_res: 求解得来的参数
  # Output：
      # 混淆矩阵与预测值
  
  X <- as.matrix(da[, 1:2])
  preY <- X %*% w_res + b_res
  preY <- ifelse(preY>=0, 1, -1)
  # print cof matrix
  TY <- da[, 3]
  print(table(preY, TY))
  
  return(preY)
}
preY <- predictbase(data1.test, w_res, b_res)

ggplot(data = data1.test, aes(x = x1, y = x2, colour = Y)) + 
  geom_point(size = 2.0, shape = 16) + 
  geom_abline(slope = -w_res[1]/w_res[2], intercept = -b_res/w_res[2], colour = "red") + 
  labs(title="Test Data") + theme(plot.title = element_text(hjust = 0.5))

##### 3、SVM 对偶问题 （6.11）#####
X <- as.matrix(data1.train[, 1:2])
Y <- as.matrix(as.numeric(as.character(data1.train[, 3])))

SVMdual <- function(X, Y){
  # 求解SVM对偶形式
  # Input：
      # X: 特征
      # Y: 正负1标签
  # Output:
      # W_res: w的解
      # b_res: b的解
  
  n <- ncol(X)
  m <- nrow(Y)
  A <- Variable(m)
  
  objective <- Minimize(-sum(A) + 1/2 * quad_form(Y * A, X %*%t(X)))
  constraints <- list(A >= 0, sum(Y * A) == 0)
  prob <- Problem(objective, constraints)
  
  # Problem solution
  solution <- solve(prob)
  A_res <- solution$getValue(A)
  
  w_res <- colSums(cbind(Y * A_res, Y * A_res) * X)
  # cal b_res(according to support vector)
  ind <- which(A_res > 0.00001)[1]
  b_res <- (1 - (Y[ind, 1])*(t(X[ind, ])%*%w_res)) / Y[ind, 1]
  
  # b_res <- Y[ind, 1] - sum(A_res * Y *  X %*% t(X[ind, , drop = FALSE]))
  return(list(w_res = w_res, b_res = b_res))
}

modeldual <- SVMdual(X, Y)
w_res <- modelbase$w_res
b_res <- modelbase$b_res

ggplot(data = data1.train, aes(x = x1, y = x2, colour = Y)) + 
  geom_point(size = 2.0, shape = 16) + 
  geom_abline(slope = -w_res[1]/w_res[2], intercept = -b_res/w_res[2], colour = "red") +
  labs(title="The dual solution") + theme(plot.title = element_text(hjust = 0.5))

##### 4、高斯核SVM #####

X <- as.matrix(data2.train[, 1:2])
Y <- as.matrix(as.numeric(as.character(data2.train[, 3])))

GaussianK <- function(xi, xj, Sigma = 1){
  # 计算高斯核
  # Input:
      # xi
      # xj
      # Sigma: 带宽
  # Output
      # 高斯核
  
  temp <- exp(-sum((xi - xj)^2)/(2 * Sigma^2))
  return(temp)
}

SVMGaussiank <- function(X, Y, Sigma = 1){
  # 求解高斯核SVM对偶形式
  # Input：
      # X: 特征
      # Y: 正负1标签
      # Sigma: 高斯核带宽
  # Output:
      # A_res: alpha的解
      # b_res: b的解
  
  n <- ncol(X)
  m <- nrow(Y)
  A <- Variable(m)
  
  KernelM <- matrix(0, m, m)
  for(i in 1:m){
    for(j in 1:m){
      KernelM[i, j] <- GaussianK(X[i, ], X[j, ], Sigma)
    }
  }
  # 
  objective <- Minimize(-sum(A) + 1/2 * quad_form(Y * A, KernelM))
  constraints <- list(A >= 0, sum(Y * A) == 0)
  prob <- Problem(objective, constraints)
  
  # Problem solution
  solution <- solve(prob)
  A_res <- solution$getValue(A)
  
  # cal b_res(according to support vector)
  ind <- which(A_res > 0.00001)[1]
  b_res <- Y[ind, 1] - sum(A_res * Y * KernelM[, ind])
  return(list(A_res = A_res, b_res = b_res, X = X, Y = Y, Sigma = Sigma))
}

modelGauss <- SVMGaussiank(X, Y, 1)

# predict

PredictGauss <- function(newda, modelGauss){
  # 高斯核SVM预测函数
  # Input：
      # newda: 需要预测的数据，包含X  Y
      # modelGuass: 求解得来的模型
  # Output：
      # preYvalue：预测值
      # preSign：预测标签
      # tb: 混淆矩阵
  
  A_res <- modelGauss$A_res
  b_res <- modelGauss$b_res
  X <- modelGauss$X
  Y <- modelGauss$Y
  Sigma <- modelGauss$Sigma
  #
  newX <- as.matrix(newda[, 1:2])
  newY <- as.matrix(as.numeric(as.character(newda[, 3])))
  newm <- nrow(newda)
  m <- nrow(X)
  KernelM <- matrix(0, m, newm)
  for(i in 1:m){
    for(j in 1:newm){
      KernelM[i, j] <- GaussianK(X[i, ], newX[j, ], Sigma)
    }
  }
  # 
  preYvalue <- colSums(matrix(rep(A_res * Y, newm), ncol = newm) * KernelM) + b_res
  preSign <- ifelse(preYvalue >= 0, 1, -1)
  tb <- table(preSign, newY)
  return(list(preYvalue = preYvalue, preSign = preSign, tb = tb))
}

PreGauss.train <- PredictGauss(data2.train, modelGauss)
# The Train confusion matrix
PreGauss.train$tb
PreGauss.test <- PredictGauss(data2.test, modelGauss)
# The Test confusion matrix
PreGauss.test$tb

Wherefx0 <- function(modelGauss){
  # 构建灰度矩阵，寻找fx=0边界
  # Input: 
      # modelGuass: 拟合模型
  # Output:
      # 灰度矩阵
  X <- modelGauss$X
  Sigma <- modelGauss$Sigma
  #
  x1 <- seq(min(X[, 1]), max(X[, 1]),0.05)
  x2 <- seq(min(X[, 2]), max(X[, 2]),0.05)
  newda <- outer(x1, x2)
  colnames(newda) <- x2
  rownames(newda) <- x1
  newda <- melt(newda)
  newda$value <- 0
  colnames(newda) <- c("x1", "x2", "Y")
  Pred <- PredictGauss(newda, modelGauss)
  newda$Y <- Pred$preYvalue
  return(newda)
}

fx0 <- Wherefx0(modelGauss)

ggplot(data = data2.train, aes(x = x1, y = x2, colour = Y)) + 
  geom_point(size = 2.0, shape = 16) + 
  labs(title="Gaussian Kernel") + theme(plot.title = element_text(hjust = 0.5)) +
  stat_contour(data = fx0, aes(x = x1, y = x2, z = Y, colour = "bond"), breaks=c(0))

##### 5、软间隔与正则化 #####

X <- as.matrix(data2.train[, 1:2])
Y <- as.matrix(as.numeric(as.character(data2.train[, 3])))

SVMSoftMargin <- function(X, Y, C = 1){
  # 软间隔支持向量机
  # Input:
      # X: 特征
      # Y: 标签
      # C: 超参数（错误样本重要程度）
  # OutPut：
      # w_res, b_res 模型参数
  
  n <- ncol(X)
  m <- nrow(Y)
  w <- Variable(n)
  b <- Variable(1)
  xi <- Variable(m)
  objective <- Minimize(1/2 * norm(w, "F")^2 + C*sum(xi))
  constraints <- list(xi >= 0, Y * (X %*% w + b) >= 1-xi)
  prob <- Problem(objective, constraints)
  
  # Problem solution
  solution <- solve(prob)
  w_res <- solution$getValue(w)
  b_res <- solution$getValue(b)
  return(list(w_res = w_res, b_res = b_res))
}

modelsoftmargin <- SVMSoftMargin(X, Y, C = 1)
w_res <- modelsoftmargin$w_res
b_res <- modelsoftmargin$b_res

ggplot(data = data2.train, aes(x = x1, y = x2, colour = Y)) + 
  geom_point(size = 2.0, shape = 16) + 
  geom_abline(slope = -w_res[1]/w_res[2], intercept = -b_res/w_res[2], colour = "red") + 
  labs(title="") + theme(plot.title = element_text(hjust = 0.5)) + 
  labs(title="Soft Margin SVM (Base)(Train)") + 
  theme(plot.title = element_text(hjust = 0.5))

#------ 5.2 测试泛化能力-----------
preY <- predictbase(data2.test, w_res, b_res)
ggplot(data = data2.test, aes(x = x1, y = x2, colour = Y)) + 
  geom_point(size = 2.0, shape = 16) + 
  geom_abline(slope = -w_res[1]/w_res[2], intercept = -b_res/w_res[2], colour = "red") + 
  labs(title="Soft Margin SVM (Base)(Test)") + theme(plot.title = element_text(hjust = 0.5))

#------ 5.3 软间隔对偶问题--------
X <- as.matrix(data2.train[, 1:2])
Y <- as.matrix(as.numeric(as.character(data2.train[, 3])))

SVMSoftMarginDual <- function(X, Y, C = 1){
  # 软间隔支持向量机
  # Input:
      # X: 特征
      # Y: 标签
      # C: 超参数（错误样本重要程度）
  # OutPut：
      # w_res, b_res 模型参数
  
  n <- ncol(X)
  m <- nrow(Y)
  A <- Variable(m)
  
  objective <- Minimize(-sum(A) + 1/2 * quad_form(Y * A, X %*%t(X)))
  constraints <- list(A >= 0, A <= C, sum(Y * A) == 0)
  prob <- Problem(objective, constraints)
  
  # Problem solution
  solution <- solve(prob)
  A_res <- solution$getValue(A)
  
  w_res <- colSums(cbind(Y * A_res, Y * A_res) * X)
  # cal b_res(according to support vector)
  
  ind <- which(A_res > 0.00001)
  b_res <- min(1 - (Y[ind, 1])*(X[ind, ]%*%w_res))
  
  # b_res <- Y[ind, 1] - sum(A_res * Y *  X %*% t(X[ind, , drop = FALSE]))
  return(list(w_res = w_res, b_res = b_res))
}

modelSoftMDual <- SVMSoftMarginDual(X, Y)
w_res <- modelSoftMDual$w_res
b_res <- modelSoftMDual$b_res

ggplot(data = data2.train, aes(x = x1, y = x2, colour = Y)) + 
  geom_point(size = 2.0, shape = 16) + 
  geom_abline(slope = -w_res[1]/w_res[2], intercept = -b_res/w_res[2], colour = "red") +
  labs(title="Soft Margin SVM (Dual)(Train)") + theme(plot.title = element_text(hjust = 0.5))

#------ 5.4 高斯核函数 软间隔--------

X <- as.matrix(data2.train[, 1:2])
Y <- as.matrix(as.numeric(as.character(data2.train[, 3])))

SVMGaussiankSoftMargin <- function(X, Y, Sigma = 1, C = 1){
  # 求解高斯核SVM对偶形式
  # Input：
      # X: 特征
      # Y: 正负1标签
      # Sigma: 高斯核带宽
      # C: 超参数
  # Output:
      # A_res: alpha的解
      # b_res: b的解
  n <- ncol(X)
  m <- nrow(Y)
  A <- Variable(m)
  
  KernelM <- matrix(0, m, m)
  for(i in 1:m){
    for(j in 1:m){
      KernelM[i, j] <- GaussianK(X[i, ], X[j, ], Sigma)
    }
  }
  # 
  objective <- Minimize(-sum(A) + 1/2 * quad_form(Y * A, KernelM))
  constraints <- list(A >= 0, A <= C, sum(Y * A) == 0)
  prob <- Problem(objective, constraints)
  
  # Problem solution
  solution <- solve(prob)
  A_res <- solution$getValue(A)
  
  # cal b_res(according to support vector)
  ind <- which(A_res > 0.00001)[1]
  b_res <- Y[ind, 1] - sum(A_res * Y * KernelM[, ind])
  return(list(A_res = A_res, b_res = b_res, X = X, Y = Y, Sigma = Sigma))
}

modelGauss <- SVMGaussiankSoftMargin(X, Y, 1, 10)

PreGauss.train <- PredictGauss(data2.train, modelGauss)
# The Train confusion matrix
PreGauss.train$tb

fx0 <- Wherefx0(modelGauss)

ggplot(data = data2.train, aes(x = x1, y = x2, colour = Y)) + 
  geom_point(size = 2.0, shape = 16) + 
  labs(title="Gaussian Kernel SVM (Soft Margin, C=10)") + 
  theme(plot.title = element_text(hjust = 0.5)) +
  stat_contour(data = fx0, aes(x = x1, y = x2, z = Y, colour = "bond"), breaks=c(0))

##### 6、支持向量多分类 #####

#-------6.1 Iris数据 --------
set.seed(1)
ind <- sample(1:150, 100)
iris.train <- iris[ind, c(3:5)]
iris.test <- iris[c(1:150)[-ind], c(3:5)]
ggplot(data = iris.train, aes(x = Petal.Length, y = Petal.Width, colour = Species)) + 
  geom_point(size = 2.0, shape = 16) + labs(title="Iris Train Data") + 
  theme(plot.title = element_text(hjust = 0.5))

#-------6.2 高斯核函数软间隔多分类 --------

MultiGuassSVM <- function(da, Sigma = 1, C = 100){
  #
  m <- nrow(da)
  n <- ncol(da)
  #
  label <- levels(da[, ncol(da)])
  N <- length(label)
  X <- as.matrix(da[, 1:(n-1)])
  #
  fxdata <- data.frame(matrix(0, ncol = N, nrow = m))
  modellist <- list()
  for(i in 1:N){
    # 
    labeli <- label[i]
    Yi <- ifelse(da[, ncol(da)] == labeli, 1, -1)
    Yi <- matrix(Yi, ncol = 1)
    dai <- da
    dai[, ncol(dai)] <- Yi
    
    modeli <- SVMGaussiankSoftMargin(X, Yi, Sigma, C)
    PreGaussi <- PredictGauss(dai, modeli)
    fxdata[, i] <- PreGaussi$preYvalue
    modellist[[i]] <- modeli
  }
  Presign <- apply(fxdata, 1, function(x){order(x, decreasing=T)[1]})
  Presign <- label[Presign]
  return(list(modellist = modellist, Presign = Presign, fxdata = fxdata, 
              label = label, tb = table(Presign, True = da[, ncol(da)])))
}
modelMultSVM <- MultiGuassSVM(iris.train, Sigma = 0.1, C = 100)
modelMultSVM$tb

# plot
fx01 <- Wherefx0(modelMultSVM$modellist[[1]])
fx02 <- Wherefx0(modelMultSVM$modellist[[2]])
fx03 <- Wherefx0(modelMultSVM$modellist[[3]])

ggplot(data = iris.train, aes(x = Petal.Length, y = Petal.Width, colour = Species)) + 
  geom_point(size = 2.0, shape = 16) + 
  labs(title="Multi Guass SVM(Soft Margin, C = 100,Sigma = 0.1)") + 
  theme(plot.title = element_text(hjust = 0.5)) +
  stat_contour(data = fx01, aes(x = x1, y = x2, z = Y, colour = "bond"), 
               breaks=c(0), size = 1) +
  stat_contour(data = fx02, aes(x = x1, y = x2, z = Y, colour = "bond"), 
               breaks=c(0), size = 1) +
  stat_contour(data = fx03, aes(x = x1, y = x2, z = Y, colour = "bond"), 
               breaks=c(0), size = 1)

#--------- 6.3 检验泛化能力 ---------

PredictMultiGuass <- function(da, modellist, label){
  # 
  m <- nrow(da)
  n <- ncol(da)
  #
  N <- length(modellist)
  X <- as.matrix(da[, 1:(n-1)])
  #
  fxdata <- data.frame(matrix(0, ncol = N, nrow = m))
  for(i in 1:N){
    # 
    modeli <- modellist[[i]]
    PreGaussi <- PredictGauss(da, modeli)
    fxdata[, i] <- PreGaussi$preYvalue
  }
  Presign <- apply(fxdata, 1, function(x){order(x, decreasing=T)[1]})
  Presign <- label[Presign]
  return(list(Presign = Presign, tb = table(Presign, True = da[, ncol(da)])))
}

PredMultiGUass <- PredictMultiGuass(iris.test, modelMultSVM$modellist, modelMultSVM$label)
PredMultiGUass$tb

# plot
ggplot(data = iris.test, aes(x = Petal.Length, y = Petal.Width, colour = Species)) + 
  geom_point(size = 2.0, shape = 16) + 
  labs(title="Multi Guass SVM (Test)") + 
  theme(plot.title = element_text(hjust = 0.5)) +
  stat_contour(data = fx01, aes(x = x1, y = x2, z = Y, colour = "bond"), 
               breaks=c(0), size = 1) +
  stat_contour(data = fx02, aes(x = x1, y = x2, z = Y, colour = "bond"), 
               breaks=c(0), size = 1) +
  stat_contour(data = fx03, aes(x = x1, y = x2, z = Y, colour = "bond"), 
               breaks=c(0), size = 1)

###### 7、支持向量机回归 #####
# -------7.1 数据集 ------------
X <- seq(1, 10, 0.1)
set.seed(2)
Y <- X + 1 + rnorm(length(X), 0, 0.3)
linerda <- data.frame(X = X, Y = Y, label = c(rep("Train", 61), rep("Test", 30)))
linerda.train <- linerda[1:61, 1:2]
linerda.test <- linerda[62:91, 1:2]

ggplot(data = linerda, aes(x = X, y = Y, colour = label)) + 
  geom_point(size = 2, shape = 16) + labs(title = "Linear Data") + 
  theme(plot.title = element_text(hjust = 0.5))

#------- 7.2 模型 --------------
SVRbase <- function(da, epsilon = 0.3, C = 1){
  #
  n <- ncol(da)
  m <- nrow(da)
  X <- as.matrix(da[, 1:(n-1)], nrow = m)
  Y <- as.matrix(da[, n], nrow = m)
  #
  w <- Variable(n-1)
  b <- Variable(1)
  kexi1 <- Variable(m)
  kexi2 <- Variable(m)
  #
  objective <- Minimize(0.5 * norm(w, "F")^2 + C * sum(kexi1 + kexi2))
  constraints <- list(X %*% w + b - Y <= epsilon + kexi1,
                      Y - (X %*% w + b) <= epsilon + kexi2,
                      kexi1 >= 0, kexi2 >= 0)
  prob <- Problem(objective, constraints)
  
  # Problem solution
  solution <- solve(prob)
  w_res <- solution$getValue(w)
  b_res <- solution$getValue(b)
  return(list(w_res = w_res, b_res = b_res))
}

modelSVRbase <- SVRbase(linerda.train)
w_res <- modelSVRbase$w_res
b_res <- modelSVRbase$b_res

ggplot(data = linerda, aes(x = X, y = Y, colour = label)) + 
  geom_point(size = 2, shape = 16) + labs(title = "Linear Data") + 
  theme(plot.title = element_text(hjust = 0.5)) + 
  geom_abline(slope = w_res, intercept = b_res, size = 0.6)

#---------- 7.3 检验 -------------
predictbaseSVR <- function(da, w_res, b_res){
  #
  n <- ncol(da)
  m <- nrow(da)
  X <- as.matrix(da[, 1:(n-1)], nrow = m)
  Y <- as.matrix(da[, n], nrow = m)
  #
  PreY <- X %*% w_res + b_res
  # sum of squared errors
  SSE <- sum((PreY - Y)^2)
  return(list(PreY = PreY, SSE = SSE))
}

Prelinertest <- predictbaseSVR(linerda.test, w_res, b_res)
print(paste("SSE = ", round(Prelinertest$SSE,3)))
