import numpy as np
import matplotlib.pyplot as plt

fpr_values = np.load('fpr_values_vaw.npy')
tpr_values = np.load('tpr_values_vaw.npy')

print(fpr_values)
print(tpr_values)

plt.plot(fpr_values, tpr_values)
plt.axis([0, 1, 0, 1])
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')

plt.grid(True)
plt.plot([0, 1], [0, 1], 'r--')

plt.show()





