# RVM

------

- [x] SVM局限性
- [x] RVM回归
- [x] RVM分类

------

## 1. SVM局限性

- 输出是分类结果不是后验概率
- 推广到多分类有很多问题
- 超参（soft margin:$C,v$,回归：$\epsilon$）要通过交叉验证确定
- 核函数的要求：正定的

------

## 2. RVM-回归
相关向量机RVM
假设模型的残差（观察值与估计值）服从精度$\beta$的高斯分布
$$P(t|x,w,\beta)=N(t|w\phi(x),\beta^{-1})$$
RVM:
$$y(x) = w\phi(x) = \sum_n w k(x,x_n)+b$$
- w的先验概率分布，均值0，每个$w_i$的精度$\alpha_i$
$$p(w|\alpha) = \prod_i N(w_i|0,\alpha^{-1})$$
可求得w的后验概率分布仍是高斯分布
$$p(w|t,X,\alpha,\beta) = N(w|m,\Sigma)$$

- 给定新样本x,对权重向量积分得到输出的概率分布：
$$P(t|x,\alpha,\beta,X,T)=\int p(t|w,\beta)p(w|\alpha,X,T) dw$$

- 求解超参$\alpha \beta$
通过最大化边际似然函数$p(t_true|x,\alpha,\beta,X,T)$得到超参
方法一是令偏导数为0，方法二是EM算法。交替重估计直到收敛得到超参的解。

- 讨论
1. 稀疏性来自于，有些alpha趋于无穷，即w趋于0,对应的基函数对于模型的预测没有作用，被剪枝。w不趋于0的数据叫做相关向量。
2. 超参一次训练得到，不需要交叉验证
3. 比SVM更稀疏，但没有减小泛化误差
4. 可以给出概率形式的预测


------

## 3. RVM-分类
- 二分类
logistic sigmoid变换$y(x,w) = \sigma(w \phi(x))$
拉普拉斯近似，对边缘似然函数的近似求$\alpha$，迭代加权最小平方法不断重新估计直至收敛。
- 多分类
softmax $y=exp(wx)/\sum exp(wx)$



## reference
https://www.zhihu.com/question/27705322
《PRML》