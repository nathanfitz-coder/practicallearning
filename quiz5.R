#R version 3.4.4 
  
print("Hello, world!")
library(MASS)

attach(Boston)

head(Boston)

fit = lm(nox ~ poly(dis, 3), data = Boston)
#summary(fit)
print('#8')
sum(resid(fit)^2)


#print(poly(dis, 3))
print('#9')
predict(fit,newdata=list(dis=6))


print('#10')
summary(fit)$coef


print('#11')
fit = lm(nox ~ poly(dis, 4), data = Boston)
sum(resid(fit)^2)


print('#12')
predict(fit,newdata=list(dis=6))

print('#13')
summary(fit)$coef


