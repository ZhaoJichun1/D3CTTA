import torch
import math
import numpy as np


class OnlineMeanVariance:
    def __init__(self, dim):
        self.n = 0
        self.mean = torch.zeros(dim)  # 均值初始化为零
        self.M2 = torch.zeros(dim)    # 累积平方差初始化为零

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_mean(self):
        return self.mean

    def get_variance(self):
        if self.n > 1:
            return self.M2 / self.n
        else:
            return torch.full_like(self.mean, float('nan'))  # 如果只有1个样本，方差为 NaN

    def gaussian_probability(self, x):
        mean = self.get_mean()
        variance = self.get_variance()
        index = variance > 1e-10
        mean = mean[index]
        variance = variance[index]
        x = x[index]
        # print(mean)
        # print(variance)
        
        # 如果方差过小，可能导致数值问题，因此可以加一个较小的正则化值
        epsilon = 1e-6
        variance += epsilon
        
        # 计算对角协方差矩阵的行列式和逆（对角矩阵的行列式是方差的乘积，逆是倒数）
        cov_diag = torch.diag(variance)
        cov_inv = torch.diag(1.0 / variance)
        cov_det = torch.sum(torch.log(variance))
        # print(torch.log(variance))
        # print(torch.sum(torch.log(variance)))
        # print(cov_det)
        

        d = x.size(0)  # 样本的维度
        # print()
        # print(math.exp(math.log((2 * math.pi) ** d) + cov_det))
        # print(d)
        # print((math.sqrt(math.exp(math.log((2 * math.pi) ** d) + cov_det))))
        norm_factor = 1.0 / (math.sqrt(math.exp(math.log((2 * math.pi) ** d) + cov_det)))
        diff = x - mean
        exponent = -0.5 * (diff.T @ cov_inv @ diff)
        
        # print(exponent)
        # print(norm_factor)
        # print(math.exp(exponent))
        # input()

        return norm_factor * math.exp(exponent)


class OnlineMeanCovariance:
    def __init__(self, dim):
        self.n = 0
        self.mean = torch.zeros(dim)  # 均值初始化为零
        self.cov_matrix = torch.zeros((dim, dim))  # 协方差矩阵初始化为零

    def update(self, x):
        print(x)
        input()
        self.n += 1
        if self.n == 1:
            self.mean = x
        else:
            delta = x - self.mean
            self.mean += delta / self.n
            self.cov_matrix += (self.n - 1) / self.n * (torch.outer(delta, delta) - self.cov_matrix)

    def get_mean(self):
        return self.mean

    def get_covariance(self):
        if self.n > 1:
            return self.cov_matrix
        else:
            return torch.full_like(self.cov_matrix, float('nan'))  # 如果样本数少于2，协方差为 NaN

    def gaussian_probability(self, x):
        mean = self.get_mean()
        covariance = self.get_covariance()
        # print(mean)
        # print(covariance)

        # 如果协方差矩阵不可逆，可能导致数值问题，因此可以加一个较小的正则化值
        epsilon = 1e-6
        covariance += torch.eye(covariance.size(0)) * epsilon
        
        # 计算协方差矩阵的行列式和逆
        cov_inv = torch.inverse(covariance)
        cov_det = torch.det(covariance)

        d = x.size(0)  # 样本的维度
        norm_factor = 1.0 / torch.sqrt((2 * math.pi) ** d * cov_det)
        diff = x - mean
        exponent = -0.5 * (diff.T @ cov_inv @ diff)
        # print(exponent)
        # print(norm_factor)
        # input()
        
        return norm_factor * torch.exp(exponent)


if __name__ == '__main__':
    a = torch.rand(8)
    b = torch.rand(8)
    c = torch.rand(8)
    gua = OnlineMeanVariance(8)
    gua.update(a)
    gua.update(b)
    input()
    print(gua.mean)
    print(gua.M2)
    print(gua.gaussian_probability(b))

    