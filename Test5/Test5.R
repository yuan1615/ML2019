setwd("D:/2018--2020-研究生/2、上海工程技术大学/机器学习/Year2019/Test5")
##### 1、加载包 #####
options(stringsAsFactors = FALSE)
library(ggplot2)
library(easyGgplot2)
library(tidyverse)
library(caret)

##### 2、西瓜数据处理 #####
wmdaorign <- read.csv("西瓜数据3.0.csv")
wmdaorign <- wmdaorign[, -1]
dmy <- dummyVars(~., data = wmdaorign)
wmda <- data.frame(predict(dmy, newdata = wmdaorign))

##### 3、神经网络函数 #####
InitParam <- function(nx, nh, ny, seed = 2){
  # 初始化参数（针对两层神经网络）
  # Input
    # nx: 输入变量个数
    # nh: 隐层神经元个数
    # ny: 输出层神经元个数
    # seed: 设置随机数种子
  # Output
    # ParamList: 参数列表
  set.seed(seed)
  W1 <- matrix(rnorm(nx * nh), ncol = nx) * 0.01
  b1 <- matrix(rep(0, nh), ncol = 1)
  W2 <- matrix(rnorm(ny * nh), ncol = nh) * 0.01
  b2 <- matrix(rep(0, ny), ncol = 1)
  return(list(W1 = W1, b1 = b1, W2 = W2, b2 = b2))
}
InitParamDeep <- function(Ldims, seed = 2){
  # 初始化参数（针对N层神经网络）
  # Input
    # Ldims: 各层神经元个数
    # seed: 设置随机数种子
  # Output
    # ParamList: 参数列表
  set.seed(seed)
  Paramlist <- list()
  L = length(Ldims)
  for(i in 2:L){
    Wi <- matrix(rnorm(Ldims[i-1] * Ldims[i]), ncol = Ldims[i-1]) * 0.01
    bi <- matrix(rep(0, Ldims[i]), ncol = 1)
    Paramlist[[i-1]] <- list(Wi = Wi, bi = bi)
  }
  return(Paramlist)
}

LinearForward <- function(A, W, b){
  # 新型前向传播
  # Input :
    # A: 输入变量
    # W: 参数矩阵
    # b: 截距项
  # Output：
    # Z: 线性输出
  Z <- W %*% A + matrix(rep(b, ncol(A)), ncol = ncol(A))
  return(list(Z = Z, A = A, W = W, b = b))
}
sigmoid <- function(Z){
  # 激活函数
  # Input:
    # Z: 线性输出结果
  # Output
    # A: 激活输出
  A <- 1 / (1 + exp(-Z))
  return(A)
}
LinActForward <- function(Aprev, W, b){
  # 单层传播
  # Input:
    # Aprev: 前一个激活输出
    # W: 参数矩阵
    # b: 截距项
  # Output:
    # A: 本层激活输出
    # 需保留计算过程（后续反向传播会用到）
  templist <- LinearForward(Aprev, W, b)
  Z <- templist$Z
  A <- sigmoid(Z)
  return(list(Z = Z, A = A, W = W, b = b, Aprev = Aprev))
}

LModelForward <- function(X, Paramlist){
  # 前向传播
  # Input：
    # X: 指标
    # Paramlist: 各层参数矩阵
  # Output
    # AL: 最终激活输出
    # cacheslist: 每一层得保留过程（反向传播参数）
  cacheslist <- list()
  A <- X
  L <- length(Paramlist)
  
  for(i in 1:L){
    Aprev <- A
    W <- Paramlist[[i]]$W
    b <- Paramlist[[i]]$b
    templist <- LinActForward(Aprev, W, b)
    A <- templist$A
    cacheslist[[i]] <- templist
  }
  return(list(A = A, cacheslist = cacheslist))
}

ComputeCost <- function(AL, Y){
  # 计算损失(逻辑回归损失函数)
  # Input:
    # AL: 最终激活输出
    # Y： 实际标签
  # Output:
    # cost
  m <- length(Y)
  cost <- -1/m * sum(Y * log(AL) + (1-Y) * log(1-AL))
  return(cost)
}

LinActBackward <- function(dA, cache){
  # 单层反向传播
  # Input:
    # dA: A的偏导数
    # cache: 单层前向传播保留的运算过程
  # Output:
    # dAprev: 前一层A的偏导数
    # 参数W、b的梯度
  APrev <- cache$Aprev
  W <- cache$W
  b <- cache$b
  Z <- cache$Z
  m <- ncol(APrev)
  #
  s <- 1/(1+exp(-Z))
  dZ <- dA * (s * (1 - s))
  dAprev <- t(W) %*% dZ
  dW <- 1/m * dZ %*% t(APrev)
  db <- 1/m * rowSums(dZ)
  return(list(dAprev = dAprev, dW = dW, db = db))
}
LModelBackward <- function(AL, Y, cacheslist){
  # 反向传播
  # Input：
    # AL: 最终激活结果
    # Y: 实际标签
    # cacheslist: 全部前向传播保留过程
  # Output:
    # 各层参数的梯度（列表形式）
  grads <- list()
  L <- length(cacheslist)
  m <- length(AL)
  #
  dAL <- -Y/AL + (1-Y)/(1-AL)
  dA <- dAL
  
  for(i in L:1){
    # 
    grads[[i]] <- LinActBackward(dA, cacheslist[[i]])
    dA <- grads[[i]]$dAprev
    
  }
  return(grads)
}

UpdateParams <- function(Paramlist, grads, learnrate = 0.1){
  # 梯度下降法更新参数
  # Input:
    # Paramlist: 参数列表
    # grads: 梯度列表
    # learnrate: 学习速率
  # Output：
    # 更新后参数列表
  L <- length(Paramlist)
  for(i in 1:L){
    Paramlist[[i]]$Wi <- Paramlist[[i]]$Wi - learnrate * grads[[i]]$dW
    Paramlist[[i]]$bi <- Paramlist[[i]]$bi - learnrate * grads[[i]]$db    
  }
  return(Paramlist)
}


NNmodel <- function(X, Y, Ldims, seed = 2, learnrate = 0.1, numiter = 10000, printcost = FALSE){
  # 梯度下降法求解神经网络参数
  # Input:
    # X: 指标值（矩阵形式，列表示样本）
    # Y：实际标签（矩阵形式，列表示样本）
    # Ldims: 初始化参数矩阵
    # seed: 随机数种子（初始化参数矩阵）
    # learnrate: 学习速率
    # numiter: 梯度下降迭代次数
  # Output:
    # 跟新后参数与每次损失值
  costs <- vector(length = numiter)
  Paramlist <- InitParamDeep(Ldims, seed = seed)
  #
  for(i in 1:numiter){
    #
    Forward <- LModelForward(X, Paramlist)
    AL <- Forward$A
    cacheslist <- Forward$cacheslist
    #
    cost <- ComputeCost(AL, Y)
    #
    grads <- LModelBackward(AL, Y, cacheslist)
    #
    Paramlist <- UpdateParams(Paramlist, grads, learnrate)
    if(printcost == TRUE & i%%100==0){
      print(paste("Cost after iteration", i, ":", cost))
    }
    costs[i] <- cost
  }
  return(list(Paramlist = Paramlist, costs = costs))
}

NNmodelSGD <- function(X, Y, Ldims, seed = 2, learnrate = 0.1, numiter = 100, printcost = FALSE){
  # 随机梯度下降
  # Input:
    # X: 指标值（矩阵形式，列表示样本）
    # Y：实际标签（矩阵形式，列表示样本）
    # Ldims: 初始化参数矩阵
    # seed: 随机数种子（初始化参数矩阵）
    # learnrate: 学习速率
    # numiter: 梯度下降迭代次数
  # Output:
    # 跟新后参数与每次损失值
  m <- ncol(Y)
  costs <- vector(length = numiter*m)
  Paramlist <- InitParamDeep(Ldims, seed = seed)
  Xall <- X
  Yall <- Y
  for(i in 1:numiter){
    # 
    for(j in 1:m){
      # 随机梯度下降（对每个样本循环）
      X <- Xall[, j, drop = FALSE]
      Y <- Yall[, j, drop = FALSE]
      Forward <- LModelForward(X, Paramlist)
      AL <- Forward$A
      cacheslist <- Forward$cacheslist
      #
      cost <- ComputeCost(AL, Y)
      #
      grads <- LModelBackward(AL, Y, cacheslist)
      #
      Paramlist <- UpdateParams(Paramlist, grads, learnrate)
      if(printcost == TRUE & i%%100==0){
        print(paste("Cost after iteration", i, ":", cost))
      }
      costs[m*(i-1) + j] <- cost
    }
    
  }
  return(list(Paramlist = Paramlist, costs = costs))
}

NNpredict <- function(NewX, NewY, Model){
  # 神经网络预测
  # Input:
    # NewX: 新样本
    # NewY：
    # Model：训练好的模型
  # Output：
    # 预测值，混淆矩阵，精度
  PreY <- LModelForward(NewX, Model$Paramlist)$A
  PreY <- ifelse(PreY >= 0.5, 1, 0)
  tb <- table(PreY, NewY)
  accuracy <- sum(diag(tb)) / sum(tb)
  return(list(PreY = PreY, tb = tb, accuracy = accuracy))
}


##### 4、模型建立 #####

X <- t(as.matrix(wmda[, 1:19]))
Y <- t(as.matrix(wmda[, 21]))

set.seed(seed = 1)
ind <- sample(ncol(Y), 12)
X.train <- X[, ind]
Y.train <- Y[, ind, drop = FALSE]
X.test <- X[, -ind]
Y.test <- Y[, -ind, drop = FALSE]

#------------4.1 梯度下降-----------------

Model <- NNmodel(X.train, Y.train, Ldims = c(19, 10, 1), seed = 2, learnrate = 0.1, numiter = 2000)

# plot cost
costs <- data.frame(iter = 1:2000, cost = Model$costs)
ggplot(data = costs, aes(x = iter, y = cost)) + geom_line(color = "red")

# Predict
Predict.train <- NNpredict(X.train, Y.train, Model)
print(paste("The Train accuracy is : ", Predict.train$accuracy * 100, "%"))
print("The confusion matrix is : ")
Predict.train$tb

# Test
Predict.test <- NNpredict(X.test, Y.test, Model)
print(paste("The Test accuracy is : ", Predict.test$accuracy * 100, "%"))
print("The confusion matrix is : ")
Predict.test$tb

#------------4.2 随机梯度下降-----------------

ModelSGD <- NNmodelSGD(X.train, Y.train, Ldims = c(19, 10, 1), 
                       seed = 2, learnrate = 0.1, numiter = 100)

# plot cost
costs <- data.frame(iter = 1:(12 * 100), cost = ModelSGD$costs)
ggplot(data = costs, aes(x = iter, y = cost)) + geom_line(color = "blue", lwd = 1)

# Predict
Predict.train <- NNpredict(X.train, Y.train, ModelSGD)
print(paste("The Train accuracy is : ", Predict.train$accuracy * 100, "%"))
print("The confusion matrix is : ")
Predict.train$tb

# Test
Predict.test <- NNpredict(X.test, Y.test, ModelSGD)
print(paste("The Test accuracy is : ", Predict.test$accuracy * 100, "%"))
print("The confusion matrix is : ")
Predict.test$tb
