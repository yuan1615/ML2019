setwd("D:\\2018--2020-研究生\\2、上海工程技术大学\\机器学习\\Year2019\\Test3")

options(stringsAsFactors = FALSE)
library(ggplot2)
library(easyGgplot2)


wmda <- read.csv(file = "西瓜数据2.0.csv")
wmda[, 8] <- as.factor(wmda[, 8])

knitr::kable(wmda)


TreeGenerate <- function(da, method = "ID3"){
  
  # Input
  #    da : 1(ID), 1:n-1(attribute), n(label)
  # Output
  #    tree
  
  CalEnt <- function(pk){
    Entd <- -(pk * log2(pk))
    Entd[is.na(Entd)] <- 0
    Entd <- sum(Entd)
    return(Entd)
  }
  
  tree <- list()
  # compute boot node Entd
  pvec <- as.numeric(table(da[, ncol(da)])) / nrow(da)
  Entd <- CalEnt(pvec)
  tree[[1]] <- list(a = "boot", aname = "boot", boot = da[, 1], Entd = Entd)
  a <- 2 # count tree
  dalist <- list(da = list(da = da), count = 1)
  treelast <- list()
  treenew <- list()
  # Generate tree
  while(length(dalist$da) > 0){
    Retlist <- list()
    k <- 1  # count dalistnew
    dalistnew <- list()
    for(t in 1:length(dalist$da)){
      # 
      count <- dalist$count
      dat <- dalist$da[[t]]
      # compute boot node Entd
      pvec <- as.numeric(table(dat[, ncol(dat)])) / nrow(dat)
      Entd <- CalEnt(pvec)
      # choose attribute
      AEnt <- rep(0, length = ncol(dat) - 2)
      IV <- AEnt
      Gini <- rep(0, length = ncol(dat) - 2)
      ARetlist <- list()
      for(i in 2:(ncol(dat)-1)){
        avec <- levels(as.factor(dat[, i]))
        aretlist <- list()
        for(j in 1:length(avec)){
          aj.boot <- dat[which(dat[, i] == avec[j]), 1]
          pvec <- table(dat[which(dat[, 1] %in% aj.boot), ncol(dat)])
          pvec <- as.numeric(pvec) / length(aj.boot)
          AEnt[i-1] <- AEnt[i-1] + CalEnt(pvec) * length(aj.boot) / nrow(dat)
          IV[i-1] <- IV[i-1] - length(aj.boot) / nrow(dat) * 
            log2(length(aj.boot) / nrow(dat))
          Gini[i-1] <- Gini[i-1] + (1-sum(pvec^2)) * length(aj.boot) / nrow(dat)
          aretlist[[j]] <- list(a = colnames(dat)[i], aname = avec[j], 
                                boot = aj.boot, Entd = CalEnt(pvec))
        }
        ARetlist[[i-1]] <- aretlist
      }
      # good attribute
      ########## ID 3 #############
      if(method == "ID3"){
        ret <- ARetlist[[which(AEnt == min(AEnt))[1]]]
      }
      ######### C4.5 ############
      if(method == "C4.5"){
        ind <- which(AEnt <= mean(AEnt))
        #
        Gr <- (1 - AEnt[ind]) / IV[ind]
        ret <- ARetlist[[ind[which(Gr == max(Gr))[1]]]]
      }
      ######## Gini ############
      if(method == "Gini"){
        ret <- ARetlist[[which(Gini == min(Gini))[1]]]
      }
      
      #
      retlist <- list()
      
      #           # print Tree last
      if(count == 1){
        print(paste("||", "Boot", " {" , paste(da[, 1], collapse = ",") ,"} ",
                    "Ent = ", round(tree[[1]]$Entd, 3), sep = ""))
      }else{
        print(paste(paste(rep("..", count-1), collapse = "."), count-1, ")",
                    tail(treelast[[t]]$a, 1), "=", tail(treelast[[t]]$aname, 1),
                    " {" , paste(treelast[[t]]$boot, collapse = ",") ,"} ",
                    sep = ""))
      }
      
      for(i in 1 : length(ret)){
        # Get the previous level of classification
        # use treelast
        if(count == 1){
          treelast <- tree
        }
        classe <- as.character(da[which(da[, 1] %in% ret[[i]]$boot), ncol(da)])
        classe <- table(as.numeric(classe))
        
        if(ret[[i]]$Entd == 0 | length(ret[[i]]$boot) == 0){
          # print Tree
          print(paste(paste(rep("..", count), collapse = "."), count, ")", 
                      ret[[i]]$a, "=", ret[[i]]$aname,
                      " {", paste(ret[[i]]$boot, collapse = ","), "} ",
                      ifelse(names(which(classe == max(classe)))[1] == 1,
                             "Good", "Bad"), sep = ""))
          #
          retlist[[i]] <- list(a = c(treelast[[t]]$a, ret[[i]]$a), 
                               aname = c(treelast[[t]]$aname, ret[[i]]$aname), 
                               boot = ret[[i]]$boot, Entd = ret[[i]]$Entd, 
                               class = names(which(classe == max(classe)))[1])
        }
        
        if(ret[[i]]$Entd != 0 & length(ret[[i]]$boot) != 0){
          dalistnew[[k]] <- dat[which(dat[, 1] %in% ret[[i]]$boot), 
                                -which(colnames(dat) == ret[[i]]$a)]
          # truenew
          treenew[[k]] <- list(a = c(treelast[[t]]$a, ret[[i]]$a), 
                               aname = c(treelast[[t]]$aname, ret[[i]]$aname), 
                               boot = ret[[i]]$boot, Entd = ret[[i]]$Entd, 
                               class = names(which(classe == max(classe)))[1])
          k <- k + 1
        }
      }
      Retlist[[t]] <- retlist
    } # end for (dalist)
    count <- count + 1
    tree[[count]] <- Retlist
    dalistnew <- list(da = dalistnew, count = count)
    dalist <- dalistnew
    treelast <- treenew
  } # end while
  return(tree)
}


Tree <- TreeGenerate(wmda, method = "ID3")



Tree <- TreeGenerate(wmda, method = "C4.5")


Tree <- TreeGenerate(wmda, method = "Gini")


wmda.train <- wmda[c(1,2,3,6,7,10,14:17), c(1,6,2,3,4,5,7,8)]
wmda.test <- wmda[c(4,5,8,9,11:13),  c(1,6,2,3,4,5,7,8)]


Tree.train <- TreeGenerate(wmda.train, method = "ID3")


TreeGenerate2 <- function(da.train, da.test, method = "ID3"){
  
  # Input
  #    da : 1(ID), 1:n-1(attribute), n(label)
  # Output
  #    tree
  da <- da.train
  
  CalEnt <- function(pk){
    Entd <- -(pk * log2(pk))
    Entd[is.na(Entd)] <- 0
    Entd <- sum(Entd)
    return(Entd)
  }
  
  tree <- list()
  # compute boot node Entd
  pvec <- as.numeric(table(da[, ncol(da)])) / nrow(da)
  Entd <- CalEnt(pvec)
  tree[[1]] <- list(a = "boot", aname = "boot", boot = da[, 1], Entd = Entd)
  a <- 2 # count tree
  dalist <- list(da = list(da = da), count = 1)
  dalist.test <- list(da.test = list(da.test=da.test), count = 1)
  treelast <- list()
  treenew <- list()
  treeall <- list()
  # Generate tree
  while(length(dalist$da) > 0){
    Retlist <- list()
    k <- 1  # count dalistnew
    dalistnew <- list()
    datestlistnew <- list()
    for(t in 1:length(dalist$da)){
      # 
      count <- dalist$count
      dat <- dalist$da[[t]]
      dat.test <- dalist.test$da.test[[t]]
      # compute boot node Entd
      pvec <- as.numeric(table(dat[, ncol(dat)])) / nrow(dat)
      Entd <- CalEnt(pvec)
      # choose attribute
      AEnt <- rep(0, length = ncol(dat) - 2)
      IV <- AEnt
      Gini <- rep(0, length = ncol(dat) - 2)
      ARetlist <- list()
      for(i in 2:(ncol(dat)-1)){
        avec <- levels(as.factor(dat[, i]))
        aretlist <- list()
        for(j in 1:length(avec)){
          aj.boot <- dat[which(dat[, i] == avec[j]), 1]
          pvec <- table(dat[which(dat[, 1] %in% aj.boot), ncol(dat)])
          pvec <- as.numeric(pvec) / length(aj.boot)
          AEnt[i-1] <- AEnt[i-1] + CalEnt(pvec) * length(aj.boot) / nrow(dat)
          IV[i-1] <- IV[i-1] - length(aj.boot) / nrow(dat) * 
            log2(length(aj.boot) / nrow(dat))
          Gini[i-1] <- Gini[i-1] + (1-sum(pvec^2)) * length(aj.boot) / nrow(dat)
          aretlist[[j]] <- list(a = colnames(dat)[i], aname = avec[j], 
                                boot = aj.boot, Entd = CalEnt(pvec))
        }
        ARetlist[[i-1]] <- aretlist
      }
      # good attribute
      ########## ID 3 #############
      if(method == "ID3"){
        ret <- ARetlist[[which(AEnt == min(AEnt))[1]]]
      }
      ######### C4.5 ############
      if(method == "C4.5"){
        ind <- which(AEnt <= mean(AEnt))
        #
        Gr <- (1 - AEnt[ind]) / IV[ind]
        ret <- ARetlist[[ind[which(Gr == max(Gr))[1]]]]
      }
      ######## Gini ############
      if(method == "Gini"){
        ret <- ARetlist[[which(Gini == min(Gini))[1]]]
      }
      
      #
      retlist <- list()
      
      #           # print Tree last
      if(count == 1){
        print(paste("||", "Boot", " {" , paste(da[, 1], collapse = ",") ,"} ",
                    "Ent = ", round(tree[[1]]$Entd, 3), sep = ""))
      }else{
        print(paste(paste(rep("..", count-1), collapse = "."), count-1, ")",
                    tail(treelast[[t]]$a, 1), "=", tail(treelast[[t]]$aname, 1),
                    " {" , paste(treelast[[t]]$boot, collapse = ",") ,"} ",
                    sep = ""))
      }
      ###### prepruning ######
      # 
      T.rat.old <- max(table(dat.test$label))/nrow(dat.test)
      #
      Tcount <- 0
      for(i in 1:length(ret)){
        reti <- ret[[i]]
        if(length(reti)>0){
          fl <- table(dat[which(dat[, 1] %in% reti$boot), ncol(dat)])
          fl <- names(fl)[which(fl == max(fl))[1]]
          ind1 <- which(colnames(dat.test) == reti$a)
          ind2 <- which(dat.test[, ind1] == reti$aname)
          Tcount <- Tcount + sum(as.numeric(as.character(dat.test[ind2, ncol(dat.test)]) == fl))
        }else{
          Tcount <- 0
        }
      }
      if(Tcount/nrow(dat.test) <= T.rat.old){
        cl <- table(dat.test$label)
        print(paste("Class:", ifelse(names(cl)[which(cl == max(cl))[1]] == "1",
                                     "Good", "Bad"), sep = ""))
        print("No increase in accuracy, terminate classification")
        
        next()
      }
      ###
      for(i in 1 : length(ret)){
        # Get the previous level of classification
        # use treelast
        if(count == 1){
          treelast <- tree
        }
        classe <- as.character(da[which(da[, 1] %in% ret[[i]]$boot), ncol(da)])
        classe <- table(as.numeric(classe))
        
        if(ret[[i]]$Entd == 0 | length(ret[[i]]$boot) == 0){
          # print Tree
          print(paste(paste(rep("..", count), collapse = "."), count, ")", 
                      ret[[i]]$a, "=", ret[[i]]$aname,
                      " {", paste(ret[[i]]$boot, collapse = ","), "} ",
                      ifelse(names(which(classe == max(classe)))[1] == 1,
                             "Good", "Bad"), sep = ""))
          #
          retlist[[i]] <- list(a = c(treelast[[t]]$a, ret[[i]]$a), 
                               aname = c(treelast[[t]]$aname, ret[[i]]$aname), 
                               boot = ret[[i]]$boot, Entd = ret[[i]]$Entd, 
                               class = names(which(classe == max(classe)))[1])
        }
        
        if(ret[[i]]$Entd != 0 & length(ret[[i]]$boot) != 0){
          dalistnew[[k]] <- dat[which(dat[, 1] %in% ret[[i]]$boot), 
                                -which(colnames(dat) == ret[[i]]$a)]
          ### dat
          reti <- ret[[i]]
          ind1 <- which(colnames(dat.test) == reti$a)
          ind2 <- which(dat.test[, ind1] == reti$aname)
          datestlistnew[[k]] <- dat.test[ind2, -which(colnames(dat.test) == reti$a)]
          # truenew
          treenew[[k]] <- list(a = c(treelast[[t]]$a, ret[[i]]$a), 
                               aname = c(treelast[[t]]$aname, ret[[i]]$aname), 
                               boot = ret[[i]]$boot, Entd = ret[[i]]$Entd, 
                               class = names(which(classe == max(classe)))[1])
          k <- k + 1
        }
      }
      Retlist[[t]] <- retlist
    } # end for (dalist)
    if(length(Retlist) > 0){
      count <- count + 1
      tree[[count]] <- Retlist
      dalistnew <- list(da = dalistnew, count = count)
      datestlistnew <- list(da.test = datestlistnew, count = count)
      dalist.test <- datestlistnew
      dalist <- dalistnew
      treelast <- treenew
    }else{
      dalistnew <- list(da = dalistnew, count = count)
      datestlistnew <- list(da.test = datestlistnew, count = count)
      dalist.test <- datestlistnew
      dalist <- dalistnew
    }
  } # end while
  return(tree)
}

Tree <- TreeGenerate2(wmda.train, wmda.test, method = "ID3")

TreeGenerate3 <- function(da, method = "ID3", countinue.ind = c(8, 9)){
  
  # Input
  #    da : 1(ID), 1:n-1(attribute), n(label)
  # Output
  #    tree
  method = "ID3"
  indnames <- colnames(da)[countinue.ind]
  CalEnt <- function(pk){
    Entd <- -(pk * log2(pk))
    Entd[is.na(Entd)] <- 0
    Entd <- sum(Entd)
    return(Entd)
  }
  
  tree <- list()
  # compute boot node Entd
  pvec <- as.numeric(table(da[, ncol(da)])) / nrow(da)
  Entd <- CalEnt(pvec)
  tree[[1]] <- list(a = "boot", aname = "boot", boot = da[, 1], Entd = Entd)
  a <- 2 # count tree
  dalist <- list(da = list(da = da), count = 1)
  treelast <- list()
  treenew <- list()
  # Generate tree
  while(length(dalist$da) > 0){
    Retlist <- list()
    k <- 1  # count dalistnew
    dalistnew <- list()
    for(t in 1:length(dalist$da)){
      # 
      count <- dalist$count
      dat <- dalist$da[[t]]
      dat2 <- dat
      # compute boot node Entd
      pvec <- as.numeric(table(dat[, ncol(dat)])) / nrow(dat)
      Entd <- CalEnt(pvec)
      # choose attribute
      AEnt <- rep(0, length = ncol(dat) - 2)
      IV <- AEnt
      Gini <- rep(0, length = ncol(dat) - 2)
      ARetlist <- list()
      for(i in 2:(ncol(dat)-1)){
        ##### countinue ######
        if(colnames(dat)[i] %in% indnames){
          dat.order <- dat[order(dat[, i]),]
          Ta1 <- dat.order[-nrow(dat.order), i]
          Ta2 <- dat.order[-1, i]
          Ta <- (Ta2 + Ta1) / 2
          AEnt.temp <- rep(0, length(Ta))
          IV.temp <- rep(0, length(Ta))
          for(ii in 1:length(Ta)){
            dat.temp <- dat
            Taii <- Ta[ii]
            ind1 <- which(dat.temp[, i] <= Taii)
            ind2 <- which(dat.temp[, i] > Taii)
            dat.temp[ind1, i] <- paste("<=", Taii)
            dat.temp[ind2, i] <- paste(">", Taii)
            ###
            avec <- levels(as.factor(dat.temp[, i]))
            aretlist <- list()
            for(j in 1:length(avec)){
              aj.boot <- dat.temp[which(dat.temp[, i] == avec[j]), 1]
              pvec <- table(dat.temp[which(dat.temp[, 1] %in% aj.boot), ncol(dat.temp)])
              pvec <- as.numeric(pvec) / length(aj.boot)
              AEnt.temp[ii] <- AEnt.temp[ii] + CalEnt(pvec) * length(aj.boot) / nrow(dat.temp)
              IV.temp[ii] <- IV.temp[ii] - length(aj.boot) / nrow(dat.temp) * 
                log2(length(aj.boot) / nrow(dat.temp))
              Gini[i-1] <- Gini[i-1] + (1-sum(pvec^2)) * length(aj.boot) / nrow(dat.temp)
              aretlist[[j]] <- list(a = colnames(dat.temp)[i], aname = avec[j], 
                                    boot = aj.boot, Entd = CalEnt(pvec))
            }
            
          }
          Tagood <- Ta[which(AEnt.temp == min(AEnt.temp))[1]]
          ### ID3   ####
          ind1 <- which(dat2[, i] <= Tagood)
          ind2 <- which(dat2[, i] > Tagood)
          dat2[ind1, i] <- paste("<=", Tagood)
          dat2[ind2, i] <- paste(">", Tagood)
          ### c4.5 ###
        }
        
        avec <- levels(as.factor(dat2[, i]))
        aretlist <- list()
        for(j in 1:length(avec)){
          aj.boot <- dat2[which(dat2[, i] == avec[j]), 1]
          pvec <- table(dat2[which(dat[, 1] %in% aj.boot), ncol(dat2)])
          pvec <- as.numeric(pvec) / length(aj.boot)
          AEnt[i-1] <- AEnt[i-1] + CalEnt(pvec) * length(aj.boot) / nrow(dat2)
          IV[i-1] <- IV[i-1] - length(aj.boot) / nrow(dat2) * 
            log2(length(aj.boot) / nrow(dat2))
          Gini[i-1] <- Gini[i-1] + (1-sum(pvec^2)) * length(aj.boot) / nrow(dat2)
          aretlist[[j]] <- list(a = colnames(dat2)[i], aname = avec[j], 
                                boot = aj.boot, Entd = CalEnt(pvec))
        }
        ARetlist[[i-1]] <- aretlist
        
        #####
      }
      # good attribute
      ########## ID 3 #############
      if(method == "ID3"){
        ret <- ARetlist[[which(AEnt == min(AEnt))[1]]]
      }
      ######### C4.5 ############
      if(method == "C4.5"){
        ind <- which(AEnt <= mean(AEnt))
        #
        Gr <- (1 - AEnt[ind]) / IV[ind]
        ret <- ARetlist[[ind[which(Gr == max(Gr))[1]]]]
      }
      ######## Gini ############
      if(method == "Gini"){
        ret <- ARetlist[[which(Gini == min(Gini))[1]]]
      }
      
      #
      retlist <- list()
      
      #           # print Tree last
      if(count == 1){
        print(paste("||", "Boot", " {" , paste(da[, 1], collapse = ",") ,"} ",
                    "Ent = ", round(tree[[1]]$Entd, 3), sep = ""))
      }else{
        print(paste(paste(rep("..", count-1), collapse = "."), count-1, ")",
                    tail(treelast[[t]]$a, 1), "=", tail(treelast[[t]]$aname, 1),
                    " {" , paste(treelast[[t]]$boot, collapse = ",") ,"} ",
                    sep = ""))
      }
      
      for(i in 1 : length(ret)){
        # Get the previous level of classification
        # use treelast
        if(count == 1){
          treelast <- tree
        }
        classe <- as.character(da[which(da[, 1] %in% ret[[i]]$boot), ncol(da)])
        classe <- table(as.numeric(classe))
        
        if(ret[[i]]$Entd == 0 | length(ret[[i]]$boot) == 0){
          # print Tree
          print(paste(paste(rep("..", count), collapse = "."), count, ")", 
                      ret[[i]]$a, "=", ret[[i]]$aname,
                      " {", paste(ret[[i]]$boot, collapse = ","), "} ",
                      ifelse(names(which(classe == max(classe)))[1] == 1,
                             "Good", "Bad"), sep = ""))
          #
          retlist[[i]] <- list(a = c(treelast[[t]]$a, ret[[i]]$a), 
                               aname = c(treelast[[t]]$aname, ret[[i]]$aname), 
                               boot = ret[[i]]$boot, Entd = ret[[i]]$Entd, 
                               class = names(which(classe == max(classe)))[1])
        }
        
        if(ret[[i]]$Entd != 0 & length(ret[[i]]$boot) != 0){
          dalistnew[[k]] <- dat[which(dat[, 1] %in% ret[[i]]$boot), 
                                -which(colnames(dat) == ret[[i]]$a)]
          # truenew
          treenew[[k]] <- list(a = c(treelast[[t]]$a, ret[[i]]$a), 
                               aname = c(treelast[[t]]$aname, ret[[i]]$aname), 
                               boot = ret[[i]]$boot, Entd = ret[[i]]$Entd, 
                               class = names(which(classe == max(classe)))[1])
          k <- k + 1
        }
      }
      Retlist[[t]] <- retlist
    } # end for (dalist)
    count <- count + 1
    tree[[count]] <- Retlist
    dalistnew <- list(da = dalistnew, count = count)
    dalist <- dalistnew
    treelast <- treenew
  } # end while
  return(tree)
}


wmda <- read.csv(file = "西瓜数据3.0.csv")
wmda[, 10] <- as.factor(wmda[, 10])
knitr::kable(head(wmda, 2))
Tree <- TreeGenerate3(wmda)











