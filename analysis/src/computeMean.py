
import numpy as np
from scipy.stats import ttest_ind


KR = np.loadtxt("timeKernelThenRecv.txt")
RK = np.loadtxt("timeRecvThenKernel.txt")

meanKR = np.mean(KR)
meanRK = np.mean(RK)

stdKR = np.std(KR)
stdRK = np.std(RK)

print("KR = %0.3g pm %0.1g" % (meanKR, stdKR))
print("RK = %0.3g pm %0.1g" % (meanRK, stdRK))

[t, prob] = ttest_ind(KR, RK)

print(t, prob)
