

##############
# INTRODUÇÃO #
##############


# Neste laboratório, nós vamos aplicar métodos de aprendizado de máquina supervisionado
#para o problema de detecção de anomalias.

# Vamos utilizar o seguinte dataset:
setwd('C:/Users/lukas/OneDrive/Área de Trabalho/Lukas Lopes/ITA/4o Semestre - CIVIL 21/TRA-48/Lab4') 
#data <- read.csv(file='TRA48_Lab6_data.csv', header=TRUE, colClasses='factor')
load('TRA48_Lab6_data.Rda')
summary(data)

# O dataset apresenta um histórico de voos entre as regiões de Washington (aeroportos BWI, DCA, IAD) e 
#New York (aeroportos EWR, JFK, LGA), contendo as seguintes variáveis:
 # flightnumber - número do voo
 # carrier - companhia aérea
 # deptime - horário de saída do aeroporto de origem
 # dayofweek - dia da semana
 # orig - aeroporto de origem
 # dest - aeroporto de destino
 # weather - variável binária que indica se houve algum impacto meteorológico no horário de saída do voo (weather=1)
 # delayed - variável binária que indica se o voo sofreu atraso na saída (delayed=1), isto é, se a diferença entre
#o horário real de saída e o horário planejado de saída foi maior que 15 min

# O nosso objetivo é utilizar um método de aprendizado de máquina supervisionado para detectar se um voo
# deve sofrer um atraso.


######################################################
# ETAPA 1: PREPARAÇÃO DAS BASES DE TREINAMENTO/TESTE #
######################################################


# Primeiramente, vamos particionar a base de dados em treinamento e teste, utilizando a proporção 75%/25%.
nobs <- length(data[,1])
set.seed(1)
idxtrain <- sample(c(1:nobs),0.75*nobs)
data_train <- data[idxtrain,]
data_test <- data[-idxtrain,]

# Como em um típico problema de detecção de anomalias, percebemos que o número de voos atrasados (delayed=1)
#é muito menor (~ 20% das observações). Assim, devemos adotar uma estratégia para lidar com este problema
#de desbalanceamento das classes e viabilizar o correto aprendizado do modelo.
# Conforme discutido na vídeo-aula, uma forma de mitigar este problema é dar pesos diferentes para as 
#observações e utilizá-los durante o treinamento do modelo.

# Vamos definir o peso das observações a partir das proporções das classes. Quanto maior a proporção, menor o peso.
counts <- summary(data_train$delayed)
prop_class0 <- counts[1]/sum(counts) 
prop_class1 <- counts[2]/sum(counts) 
weights <- numeric(length(data_train$delayed))
weights[which(data_train$delayed==0)] <- 1/prop_class0
weights[which(data_train$delayed==1)] <- 1/prop_class1
data_train$weights <- weights


########################################################################################
# ETAPA 2: TREINAMENTO DO MODELO DE CLASSIFICAÇÃO USANDO O MÉTODO DE ÁRVORE DE DECISÃO #
########################################################################################


# Vamos utilizar o método de árvore de decisão para criar o modelo de classificação para previsão de atraso de voo.

library(tree)

# Gerar árvore de decisão completa usando a base de treinamento "data_train", onde "delayed" é o output 
#e (dayofweek,carrier,orig,dest,deptime,weather) são input.
tree_model <- tree(delayed ~ dayofweek + carrier + orig + dest + deptime + weather, data=data_train, weights=weights, control=tree.control(nobs=length(data_train$delayed),mindev=0,minsize=1))
# Usamos o argumento "weights" para dar pesos diferentes às observações e mitigar o problema de desbalanceamento das classes.

# Visualizar árvore completa
plot(tree_model)
text(tree_model)

# Realizar um processo de validação cruzada 5-fold para obter árvores podadas de diferentes tamanhos
#a partir da árvore completa gerada anteriormente 'tree_model'
set.seed(1)
cv_results <- cv.tree(tree_model, FUN=prune.tree, K=5)

# Plotar os erros de classificação em função do tamanho da árvore (número de nós terminais)
plot(cv_results$size, cv_results$dev, type="b")

# Índice da árvore de menor erro
min_idx <- which.min(cv_results$dev)

# Número de nós terminais da árvore de menor erro
optimal_tree_size <- cv_results$size[min_idx]

# Vamos podar a árvore usando a função "prune.tree", indicando o tamanho ótimo obtido a partir do processo de validação cruzada através do argumento "best"
pruned_tree_model = prune.tree(tree_model, best=optimal_tree_size)
plot(pruned_tree_model)
text(pruned_tree_model)

# Finalmente, vamos prever y na base de teste usando a árvore podada ótima e avaliar a acurácia da previsão
yhat <- predict(pruned_tree_model,data_test,type='class')
library(caret)
confusion <- confusionMatrix(yhat,data_test$delayed, positive='1')
print(confusion$table)
print(confusion$byClass)  # SPECIALIZED PERFORMANCE METRICS
accuracy <- sum(diag(confusion$table))/sum(confusion$table)
print(accuracy)


#######################################################################
# ETAPA 3: TREINAMENTO DO MODELO DE CLASSIFICAÇÃO USANDO O MÉTODO SVM #
#######################################################################


# Alternativamente, vamos utilizar o método SVM para criar o modelo de classificação para previsão de atraso de voo.

library(e1071)

# Na funções "svm" e "tune" da biblioteca "e1071", o argumento "class.weights" permite indicar um vetor
#com diferentes pesos para cada classe.
weights <- table(data_train$delayed)  
prop_class0 <- weights[1]/sum(weights) 
prop_class1 <- weights[2]/sum(weights) 
weights[1] <- 1
weights[2] <- 1

# Realizar um processo de validação cruzada para determinar o modelo SVM ótimo.
set.seed(1)
cv_results=tune(svm, delayed ~ dayofweek + carrier + orig + dest + deptime + weather, data=data_train, class.weights=weights, ranges=list(cost=c(100), kernel=c('radial')))
summary(cv_results)
# Usamos o argumento "class.weights" para dar pesos diferentes às observações e mitigar o problema de desbalanceamento das classes.

# Modelo SVM de menor erro
best_svm_model <- cv_results$best.model
summary(best_svm_model)

#best_svm_model = svm(delayed ~ dayofweek + carrier + orig + dest + deptime + weather, class.weights=weights, data=data_train, kernel="radial", cost=100)

# Finalmente, vamos prever y na base de teste usando e avaliar a acurácia da previsão
yhat <- predict(best_svm_model,data_test,type='class')
confusion <- confusionMatrix(yhat,data_test$delayed, positive='1')
print(confusion$table)
print(confusion$byClass)  # SPECIALIZED PERFORMANCE METRICS
accuracy <- sum(diag(confusion$table))/sum(confusion$table)
print(accuracy)





