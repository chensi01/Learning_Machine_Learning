# SVM

------

- [x] hard margin
- [x] soft margin
- [x] SVM多分类

------

## 1. hard margin
- 完全线性可分：y=wx+b(1正类/-1负类)

- 点(x,y)到平面)的距离
$d = \frac{yf(x)}{||w||^2}$
最小化最大的距离$arg\ min_{w,b} \max\ d$

- trick:缩放w,b的值，使得对于max{d}的点，$yf(x)=1$,那么其他点$yf(x)>1$
最小化最大的距离转化为：
$arg\ min_{w,b} \max\ \frac{1}{||w||^2}$
$subj. to\ y_nf(x_n)>=1$
- 拉格朗日
$ \min L(w,b) = \frac{1}{2} ||w||^2 - \sum_n a_n (y_nf(x_n)-1)$
$\frac{\triangledown L(w,b)}{\triangledown w} = w-\sum a_n y_n x_n = 0 $ （w只由支持向量决定）
$\frac{\triangledown L(w,b)}{\triangledown b} = \sum a_n y_n = 0 $

- 带回L(w,b)得到对偶形式
$max_a \sum a_n -\frac{1}{2}\sum_{i,j}y_i y_j a_i a_j x_i^T x_j$
$subj. to\ a_n \ge 0\ \ \ \sum a_n y_n=0$

- 其中，$a_n=0$，即$y_nf(x_n)>1$,这些点是完全分对的点，对$w$没有贡献（$w= - \sum a_n y_n x_n$）
$a_n>0$，即$y_nf(x_n)=1$,这些点是支持向量，对$w$的影响（w= \sum a_n y_n x_n），其中正样本贡献为正，负样本贡献为负，正负样本来回拉扯。

------

## 2.soft margin
- 线性不可分：$y=wx+b$(1正类/-1负类)
- 松弛变量$\xi$ 原约束$t(wx+b) \ge 1$ 现在$t_n (w_n x_n+b_n) \ge 1-\xi_n$

- 最小化最大的松弛后的距离：
$\arg\ \min_{w,b} C\sum_n\xi_n+\frac{1}{2}||w||^2$
$subj. to\ y_nf(x_n) \ge 1-\xi_n$ 
$\xi_n \ge 0$
- 拉格朗日
$ \min L(w,b) = \frac{1}{2} ||w||^2+C\sum_n\xi_n - \sum_n a_n (y_nf(x_n)-1+\xi_n) - \sum_n \mu_n\xi_n$
$\frac{\triangledown L(w,b)}{\triangledown w} = w-\sum a_n y_n x_n = 0 $ $\frac{\triangledown L(w,b)}{\triangledown b} = \sum a_n y_n = 0 $
$\frac{\triangledown L(w,b)}{\triangledown \xi_n} = C-a_n-\mu_n = 0 $

- 盒限制：$0<=a_n<=C$
1. $a_n=0$,$y_nf(x_n)-1+\xi_n>0$，完全分对，对平面无贡献
2. $a_n=C$，$y_nf(x_n)-1+\xi_n=0$,$\mu_n=0,\xi_n>0$,在边缘内部，$\xi_n<=1$分对，$\xi_n>1$分错
3. $0<a_n<C$，$y_nf(x_n)-1+\xi_n=0$,$\mu_n>0,\xi_n=0$,$y_nf(x_n)=1$，支持向量，在边界上。

------

## 3.SVM多分类
没有标准处理答案，但 一对多 用的多
1. one-versus-the-rest
- K分类则构造K个分类器，选y_k最大的作为分类结果
- 问题：1.不同分类器给出的y_k不可比  2.训练集不平衡，正负例样本个数不对称(改善方法：正例label=1负例label=$-\frac{1}{K-1}$
2. one-versus-one
- 每对类别之间训练一个分类器（$\frac{K(K-1)}{2}$个）
- 出现有奇异性的区域（区域的点不属于任何类别）