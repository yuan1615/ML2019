# Variables minimized over
x <- Variable(1)
y <- Variable(1)

# Problem definition
objective <- Minimize(x^2 + y^2)
constraints <- list(x >= 0, 2*x + y == 1)
prob2.1 <- Problem(objective, constraints)

# Problem solution
solution2.1 <- solve(prob2.1)
solution2.1$status

####
suppressMessages(suppressWarnings(library(CVXR)))
n <- 20
m <- 1000
offset <- 0
sigma <- 45
DENSITY <- 0.2

set.seed(183991)
beta_true <- stats::rnorm(n)
idxs <- sample(n, size = floor((1-DENSITY)*n), replace = FALSE)
beta_true[idxs] <- 0
X <- matrix(stats::rnorm(m*n, 0, 5), nrow = m, ncol = n)
y <- sign(X %*% beta_true + offset + stats::rnorm(m, 0, sigma))

beta <- Variable(n)
obj <- -sum(logistic(-X[y <= 0, ] %*% beta)) - sum(logistic(X[y == 1, ] %*% beta))
prob <- Problem(Maximize(obj))
result <- solve(prob)

log_odds <- result$getValue(X %*% beta)
beta_res <- result$getValue(beta)
y_probs <- 1/(1 + exp(-X %*% beta_res))



val <- cbind(c(1,2), c(3,4))
value(cumsum(Constant(val)))
value(cumsum_axis(Constant(val)))
x <- Variable(2,2)
prob <- Problem(Minimize(cumsum(x)[4]), list(x == val))
result <- solve(prob)
result$value
result$getValue(cumsum(x))



x <- Variable(2)
val <- matrix(c(-3,3))
prob <- Problem(Minimize(neg(x)[1]), list(x == val))
result <- solve(prob)
result$value





