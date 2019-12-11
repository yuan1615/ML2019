library(ggplot2)
library(easyGgplot2)
library(ggisoband)
library(MASS)
setwd("D:\\2018--2020-研究生\\2、上海工程技术大学\\机器学习\\Year2019\\Test2")

wmda <- read.csv(file = "data/西瓜数据3.0a.csv")
wmda$label <- as.factor(wmda$label)

knitr::kable(head(wmda))
summary(wmda[, -1])


ggplot(data = wmda, aes(x = density, y = sugar, color = label)) + 
  geom_point(size = 2.0, shape = 16)


gz <- function(z){
  return(1/(1 + exp(-z)))
}
gzda <- data.frame()
for(i in seq(-10, 10, 0.1)){
  gzda <- rbind(gzda, data.frame(z = i, 
                                 gz = gz(i)))
}
ggplot(data = gzda, aes(x = z, y = gz)) + 
  geom_line() + geom_point(x = 0, y = 0.5, colour = "red")

cost <- function(x){
  return(c(-log(x), -log(1-x)))
}
costda <- data.frame()
for(i in seq(0, 1, 0.01)){
  costda <- rbind(costda, data.frame(hx = i, 
                                     cost1 = cost(i)[1], 
                                     cost0 = cost(i)[2]))
}
p1 <- ggplot(data = costda, aes(x = hx, y = cost1)) + geom_line() +
  geom_point(x = 1, y = 0, colour = "red") + 
  geom_text(x = 0.5, y = 2.5, label = "y = 1", size = 5)
p2 <- ggplot(data = costda, aes(x = hx, y = cost0)) + geom_line() +
  geom_point(x = 0, y = 0, colour = "red") +
  geom_text(x = 0.5, y = 2.5, label = "y = 0", size = 5)

ggplot2.multiplot(p1, p2, cols=2)

sigmoid <- function(z){
  return(1 / (1 + exp(-z)))
}
gradientDescent <- function(X, y, theta, alpha, num_iters){
  # Initialize some useful values
  m <- length(y) # number of training examples
  J.history <- rep(0, num_iters)
  for(i in 1:num_iters){
    theta <- theta - (1/m) * (t(X) %*% (sigmoid(X %*% theta) - y))
    #  Save the cost J in every iteration    
    J.history[i] <- (1/m) * 
      sum(-y * log(sigmoid(X %*% theta)) - 
            (1-y) * (log(1 - sigmoid(X %*% theta))))
  }
  return(list(theta = theta, J = J.history))
}

X <- as.matrix(wmda[, 2:3])
X <- cbind(1, X)
y <- as.matrix(as.numeric(as.character(wmda$label)))
initial_theta <- matrix(rep(0, 3), ncol = 1)
alpha <- 0.1
Ret <- gradientDescent(X, y, initial_theta, alpha, 
                       num_iters = 5000)

as.numeric(Ret$theta)
ggplot(data = data.frame(item = 1:length(Ret$J), J = Ret$J),
       aes(x = item, y = J)) + geom_line()

ggplot(data = wmda, aes(x = density, y = sugar, color = label)) + 
  geom_point(size = 2.0, shape = 16) + 
  geom_abline(slope = -Ret$theta[2] / Ret$theta[3], 
              intercept = -Ret$theta[1]/Ret$theta[3], 
              color = "red")
mapFeature <- function(x1, x2, degree){
  df <- matrix(1, nrow = length(x1))
  for(i in 1:degree){
    for(j in 0:i){
      x <- x1^(i - j) * x2^(j)
      df <- cbind(df, x)
    }
    
  }
  return(df)
}
x1 <- wmda$density
x2 <- wmda$sugar
X <- mapFeature(x1, x2, 6)

gradientDescent <- function(X, y, theta, alpha, num_iters, lambda){
  # Initialize some useful values
  m <- length(y) # number of training examples
  n <- ncol(X)
  J.history <- rep(0, num_iters)
  for(i in 1:num_iters){
    theta[1] <- theta[1] - 
      (1/m) * (t(X[, 1]) %*% (sigmoid(X %*% theta) - y))
    theta[2:n] <- theta[2:n] - 
      (1/m) * (t(X[, 2:n]) %*% (sigmoid(X %*% theta) - y)) + 
      lambda/m * theta[2:n]
    #  Save the cost J in every iteration    
    J.history[i] <- (1/m) * 
      sum(-y * log(sigmoid(X %*% theta)) - 
            (1-y) * (log(1 - sigmoid(X %*% theta)))) + 
      (lambda/2/m) * sum(theta[2:n] ^2)
  }
  return(list(theta = theta, J = J.history))
}

y <- as.matrix(as.numeric(as.character(wmda$label)))
initial_theta <- matrix(rep(0, 28), ncol = 1)
alpha <- 0.1
lambda <- 0
Ret <- gradientDescent(X, y, initial_theta, alpha, 
                       num_iters = 100000, lambda)

as.numeric(Ret$theta)
p <- sigmoid(X %*% Ret$theta)
pos <- which(p >= 0.5)
neg <- which(p < 0.5)
p[pos] <- 1
p[neg] <- 0
t <- table(p, wmda$label)
print(paste("prediction accuracy = ", sum(t) / sum(diag(t)) * 100,
            "%"))
ggplot(data = data.frame(item = 1:length(Ret$J), J = Ret$J),
       aes(x = item, y = J)) + geom_line()
x1 <- seq(0, 0.8, 0.01)
x2 <- seq(0, 0.5, 0.01)
x.grad <- data.frame()
for(i in x1){
  for(j in x2){
    x.grad <- rbind(x.grad, c(i, j))
  }
}
colnames(x.grad) <- c("x1", "x2")
X.grad <- mapFeature(x.grad[, 1], x.grad[, 2], 6) 
p <- sigmoid(X.grad %*% Ret$theta)
idx <- which(p < 0.5+0.01 & p > 0.5-0.01)

ggplot(data = wmda, aes(x = density, y = sugar, color = label)) + 
  geom_point(size = 2.0, shape = 16) + 
  geom_line(data = x.grad[idx, ], aes(x = x1, y = x2),  colour = "red")

HessianMatrix <- function(X, y, theta, num_iters){
  # Initialize some useful values
  m <- length(y) # number of training examples
  J.history <- rep(0, num_iters)
  
  for(i in 1:num_iters){
    partial1 <- (1/m) * (t(X) %*% (sigmoid(X %*% theta) - y))
    partial2 <- (1/m) * (t(X) %*% (X * as.numeric(
      (sigmoid(X %*% theta) * 
         (1 - sigmoid(X %*% theta))))) )
    theta <- theta - solve(partial2) %*% partial1
    #  Save the cost J in every iteration    
    J.history[i] <- (1/m) * 
      sum(-y * log(sigmoid(X %*% theta)) - 
            (1-y) * (log(1 - sigmoid(X %*% theta))))
  }
  return(list(theta = theta, J = J.history))
}
X <- as.matrix(wmda[, 2:3])
X <- cbind(1, X)
initial_theta <- matrix(rep(0, 3), ncol = 1)
Ret <- HessianMatrix(X, y, initial_theta, num_iters = 10)

as.numeric(Ret$theta)
p1 <- ggplot(data = data.frame(item = 1:length(Ret$J), J = Ret$J),
             aes(x = item, y = J)) + geom_line()
p2 <- ggplot(data = wmda, aes(x = density, y = sugar, color = label)) + 
  geom_point(size = 2.0, shape = 16) + 
  geom_abline(slope = -Ret$theta[2] / Ret$theta[3], 
              intercept = -Ret$theta[1]/Ret$theta[3], 
              color = "red")

ggplot2.multiplot(p1, p2, cols=2)


HessianMatrix2 <- function(X, y, theta, num_iters, lambda){
  # Initialize some useful values
  m <- length(y) # number of training examples
  n <- ncol(X)
  J.history <- rep(0, num_iters)
  
  for(i in 1:num_iters){
    partial1 <- matrix(rep(0, n))
    partial1[1] <- (1/m) * (t(X[, 1]) %*% (sigmoid(X %*% theta) - y))
    partial1[2:n] <- (1/m) * (t(X[, 2:n]) %*% (sigmoid(X %*% theta) - y)) + 
      lambda/m * theta[2:n]
    
    partial2 <- (1/m) * (t(X) %*% (X * as.numeric(
      (sigmoid(X %*% theta) * 
         (1 - sigmoid(X %*% theta))))) )
    theta <- theta - ginv(partial2) %*% partial1
    #  Save the cost J in every iteration    
    J.history[i] <- (1/m) * 
      sum(-y * log(sigmoid(X %*% theta)) - 
            (1-y) * (log(1 - sigmoid(X %*% theta)))) + 
      (lambda/2/m) * sum(theta[2:n] ^2)
  }
  return(list(theta = theta, J = J.history))
}
x1 <- wmda$density
x2 <- wmda$sugar
X <- mapFeature(x1, x2, 6)
y <- as.matrix(as.numeric(as.character(wmda$label)))
initial_theta <- matrix(rep(0, 28), ncol = 1)
lambda <- 0
Ret <- HessianMatrix2(X, y, initial_theta,
                      num_iters = 10, lambda)

as.numeric(Ret$theta)
p <- sigmoid(X %*% Ret$theta)
pos <- which(p >= 0.5)
neg <- which(p < 0.5)
p[pos] <- 1
p[neg] <- 0
t <- table(p, wmda$label)
print(paste("prediction accuracy = ", round(sum(diag(t)/sum(t)), 4) * 100,
            "%"))
ggplot(data = data.frame(item = 1:length(Ret$J), J = Ret$J), 
       aes(x = item, y = J)) + geom_line()

LDA <- function(X, y){
  pos <- which(y == 1)
  neg <- which(y == 0)
  u1 <- as.matrix(colMeans(X[pos, ]))
  u0 <- as.matrix(colMeans(X[neg, ]))
  Sw <- (t(X[pos, ]) - as.numeric(u1)) %*% t(t(X[pos, ]) - as.numeric(u1)) + 
    (t(X[neg, ]) - as.numeric(u0)) %*% t(t(X[neg, ]) - as.numeric(u0))
  theta <- ginv(Sw) %*% (u0 - u1)
  return(list(u1=u1, u0=u0, theta = theta))
}

X <- as.matrix(wmda[, 2:3])
y <- as.matrix(as.numeric(as.character(wmda$label)))
Ret <- LDA(X, y)
theta <- Ret$theta
print(theta)


p <- ggplot(data = wmda, aes(x = density, y = sugar, colour = label)) + 
  geom_point(size = 2.0, shape = 16) + xlim(0, 0.8) + ylim(0, 0.8) +
  geom_abline(slope = (theta[2]/theta[1]), 
              intercept = 0, color = "red")

for(i in 1:nrow(X)){
  k <- theta[2]/theta[1]
  x1 <- (X[i, 2] + X[i, 1]/k) / (k + 1/k)
  x2 <- k * x1
  da <- data.frame(x = c(X[i, 1], x1), y = c(X[i, 2], x2))
  if(y[i] == 1){
    p <- p + geom_point(x = x1, y = x2, colour = "blue", size = 2) + 
      geom_line(data = da, aes(x = x, y = y, colour = "1"))
  }else{
    p <- p + geom_point(x = x1, y = x2, colour = "red", size = 2) + 
      geom_line(data = da, aes(x = x, y = y, colour = "0"))
  }
}

p + coord_equal(ratio=1)

u1 <- Ret$u1
u0 <- Ret$u0
k <- theta[2]/theta[1]
x11 <- (u1[2, 1] + u1[1, 1]/k) / (k + 1/k)
x21 <- k * x11
x10 <- (u0[2, 1] + u0[1, 1]/k) / (k + 1/k)
x20 <- k * x10
#
x1.u <- (x11 + x10)/2
x2.u <- (x21 + x20)/2


ggplot(data = wmda, aes(x = density, y = sugar, colour = label)) + 
  geom_point(size = 2.0, shape = 16) + xlim(0, 0.8) + ylim(0, 0.8) +
  geom_abline(slope = (theta[2]/theta[1]), intercept = 0, color = "red") +
  geom_abline(slope = -1/k, intercept = (1/k)*x1.u + x2.u, colour = "yellow", size = 1.08) + 
  coord_equal(ratio=1) + theme(legend.title=element_blank()) + 
  geom_point(aes(x = x11, y = x21, colour = "1"), size = 2.0, shape = 16) + 
  geom_point(aes(x = x10, y = x20, colour = "0"), size = 2.0, shape = 16)

x <- runif(100, -8, 8)
y1 <- x + rnorm(length(x)) + 4
y2 <- x + rnorm(length(x)) - 4

ggplot() + geom_point(aes(x = x, y = y1), colour = "red") +
  geom_point(aes(x = x, y = y2), colour = "blue") + 
  geom_abline(aes(slope = 1, intercept = 0, colour = "PCA"), size = 1.1)+ 
  geom_abline(aes(slope = -1, intercept = 0, colour = "LDA"),  size = 1.1)+
  xlim(-14, 14) + ylim(-14, 14) +
  coord_equal(ratio=1) + theme(legend.title=element_blank())
