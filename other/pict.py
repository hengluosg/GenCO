import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the saved .npy files
mse_method_d = np.load('randomly_generate.npy')
mse_method2_d = np.load('pure_random_search.npy')
mse_method3_d = np.load('Generative_1_solution(QRGMM).npy')
mse_method4_d = np.load('Generative_10_solution(QRGMM).npy')

# Create a DataFrame for plotting
df = pd.DataFrame({
    'randomly generate(K=1)': mse_method_d,
    'pure random search(K)': mse_method2_d,
    'Generative 1 solution(QRGMM)': mse_method3_d,
    'Generative 10 solution(QRGMM)': mse_method4_d
})

# Define the scenario labels
df.index = np.array(['Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4', 'Scenario 5', 'Scenario 6'])

# Set the seaborn theme
sns.set_theme()

# Plot the data
df.plot(kind='line', marker='o', linewidth=2, figsize=(10, 6))

# Add labels and legend
plt.ylabel(r'$\Delta(x)$', fontsize=12)
plt.xlabel('Different Scenarios', fontsize=12)
plt.legend(title="Methods", fontsize=10)
plt.xticks(ticks=range(len(df)), labels=df.index, fontsize=10)
plt.grid(True)

# Save the plot
plt.savefig('qrgmm_multi_results1.pdf', format='pdf')

# Show the plot
plt.show()