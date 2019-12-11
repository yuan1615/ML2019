options(stringsAsFactors = FALSE)
options(drop = FALSE)
# 加载包
library(ggplot2)
library(easyGgplot2)
library(MASS)
library(lattice)
library(mixtools)
setwd("D:/2018--2020-研究生/2、上海工程技术大学/机器学习/Year2019/Test6")

##### 朴素贝叶斯 #####

#------- 1、数据处理 ---------

da <- read.csv("data.csv")
da.train <- da[1:14, ]
da.test <- da[15, ]

knitr::kable(da)

#-------2、模型建立 ----------
naiveBayes <- function(da, Classn = 6, Factorn = c(2:5)){
  # 建立贝叶斯模型
  # da:数据框格式
  # Clsssn：分类类别
  # Factorn：属性字段
  
  Class <- levels(as.factor(da[, Classn]))
  Pc <- vector(length = length(Class))
  for(i in 1:length(Class)){
    Pc[i] <- (length(da[which(da[, Classn] == Class[i]), Classn]) + 1)/(nrow(da) + 2)
  }
  Pc <- data.frame(Class = Class, P = Pc)
  #
  Factorlist <- list()
  k <- 1
  for(i in Factorn){
    Classi <- levels(as.factor(da[, i]))
    temp <- as.data.frame(matrix(nrow = length(Classi), ncol = nrow(Pc) + 1))
    temp[, 1] <- Classi
    for(j in 1:length(Classi)){
      for(t in 1:length(Class)){
        temp[j, t + 1] <- (length(da[which(da[, i] == Classi[j] & da[, Classn] == Class[t]), i]) + 1)/(length(da[which(da[, Classn] == Class[t]), i]) + length(Classi))
      }
    }
    colnames(temp) <- c("Class", Pc$Class)
    Factorlist[[k]] <- temp
    k <- k+1
  }
  names(Factorlist) <- colnames(da[, Factorn])
  return(list(Pc = Pc, Factorlist = Factorlist))
  
}

mymodel <- naiveBayes(da.train, 6, c(2:5))

#--------------- 3、预测 ------------------------
predictnB <- function(da, mymodel, Factorn = c(2:5)){
  # 预测新样本所属类别
  # da:新样本数据
  # mymodel: 贝叶斯模型返回值
  # Factorn: 类别
  factors <- colnames(da[, Factorn])
  Classn <- nrow(mymodel$Pc)
  P <- data.frame(Classn = mymodel$Pc$Class, P = 1)
  for(i in 1:Classn){
    for(j in 1:length(factors)){
      temp <- mymodel$Factorlist[factors[j]][[1]]
      ind <- which(temp$Class == da[,factors[j]])
      P$P <- as.numeric(P$P * temp[ind, -1])
    }
    
  }
  
  return(P)
}

knitr::kable(predictnB(da.test, mymodel, c(2:5)))

##### EM算法求解二维高斯混合分布参数 ########

#--------- 创建实验数据 -----------
CreatData <- function(n1, n2, mu1, mu2, Sigma1, Sigma2, seed = 3){
  # 创建数据集
  # n1 数据个数
  # n2
  # mu1: 1类样本均值向量
  # mu2：2类样本均值向量
  # Sigma1:1类样本协方差矩阵
  # Sigma2:2类样本协方差矩阵
  # seed：随机数种子
  set.seed(seed)
  X1 <- mvrnorm(n1, mu1, Sigma1)
  set.seed(seed)
  X2 <- mvrnorm(n2, mu2, Sigma2)
  df <- data.frame(X = c(X1[, 1], X2[, 1]), Y = c(X1[, 2], X2[, 2]),
                   label = factor(c(rep(-1, n1), rep(1, n2))))
  return(df)
}

da <- CreatData(100, 100, c(3, -4), c(12, 8), 25*diag(2), 64*diag(2))

ggplot(data = da, aes(x = X, y = Y, colour = label)) + 
  geom_point(size = 2.0, shape = 16) + 
  geom_point(aes(x = 3, y = -4), color = "red", size = 3) +
  geom_point(aes(x = 12, y = 8), color = "blue", size = 3)

da <- da[, -3]
#--------- EM算法求解参数 -----------
GMM <- function(da){
  # 求解高斯混合分布参数
  # da: 数据集
  #
  K <- 2
  N <- nrow(da)
  D <- ncol(da)  #判断高斯分布维数
  Gamma <- matrix(0, N, K)
  Psi <- matrix(0, N, K)
  Mu <- matrix(0, K, D)
  LM <- matrix(0, K, D)
  Sigma <- array(0, dim = c(D,D,K))
  Pi <- matrix(0, 1, K)
  #
  # 选择随机的两个样本点作为期望迭代初值
  Mu[1, ] <- as.numeric(da[sample(1:(N/2), 1), ])
  Mu[2, ] <- as.numeric(da[sample((1+N/2):N, 1), ])
  # 所有数据的协方差作为协方差初值
  for(i in 1:K){
    Pi[i] <- 1/K
    Sigma[, , i] <- cov(da)
  }
  
  LMu <- Mu
  LSigma <- Sigma
  Lpi <- Pi
  # 循环求解
  while (1==1) {
    #
    for(k in 1:K){
      
    }
    
  }
  

}






