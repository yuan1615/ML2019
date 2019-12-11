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



