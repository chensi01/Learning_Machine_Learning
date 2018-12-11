# 逻辑回归

------
原作者给出的文章链接 (https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc) 图片加载不出来，所以看了李航《统计学习方法》中对应的逻辑回归部分

- [x] logistics 分布
- [x] 二项逻辑回归模型和模型参数估计方法（极大似然法）
- [x]  多元逻辑回归模型


------

## logistics 分布
连续性随机变量 分布函数 概率密度函数
位置参数$\mu$ 形状参数$\gamma$
分布函数时logis函数，s形曲线，中心对称
$\gamma$越小，s越窄
## 二项逻辑回归模型和模型参数估计方法（极大似然法）

分类模型，条件概率，监督学习，$Y=0/1$
$$P(Y=1|X) = \frac{e^{wx+b}}{1+e^{wx+b}}$$
$$P(Y=0|X) = \frac{1}{1+e^{wx+b}}$$

*对数线性模型*
几率（odds）：事件发生与不发生的概率的比值
$Y=1$发生的对数几率为$x$的线性函数：
$$log\frac{P(Y=1|X)}{P(Y=0|X)} = log(e^{wx+b})=wx+b$$

极大似然估计：转化为以对数似然函数为目标函数的优化问题，用梯度下降/拟牛顿法求解

## 多元逻辑回归模型
多分类
$$P(Y=k|X) = \frac{e^{w_k x+b_k }}{1+\sum_{k}e^{w_k x+b_k }} k=1,2,3,K-1$$
$$P(Y=0|X) = \frac{1}{1+\sum_k e^{wx+b}} k=K$$


