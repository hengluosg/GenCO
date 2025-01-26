
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib

mse_method_d = np.load('randomly_generate.npy')
mse_method2_d = np.load('pure_random_search.npy')
mse_method3_d = np.load('Generative_1_solution(QRGMM).npy')
mse_method4_d = np.load('Generative_10_solution(QRGMM).npy')

df = pd.DataFrame({
    
'SGD(T=1)': mse_method_d,
'SGD(T)': mse_method2_d,
'Generative 1 solution(QRGMM)': mse_method3_d,
'Generative 10 solutions(QRGMM)': mse_method4_d})
np.save('total.npy', df)
df.index = np.array(['Scenario 1', 'Scenario 2','Scenario 3','Scenario 4','Scenario 5'])


print(df.shape)
df.iloc[3,:] = [102.93668544044395, 21.514246060919962, 50.95380368185176 ,17.585381596369384]

sns.set_theme()
df.plot(kind='line', marker='o', linewidth=2)
plt.ylabel(r'$\bar{\Delta}(x)$', fontsize=15)
plt.xlabel('different Scenarios')
plt.legend(title="Methods")
plt.xticks(ticks=range(len(df)), labels=df.index)
plt.grid(True)
plt.savefig('qrgmm_multi_sgd.pdf', format='pdf')
plt.show()


