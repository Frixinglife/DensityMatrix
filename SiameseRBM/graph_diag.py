import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np

N_d = 0

data_diag_d = []
with open("matrix_diag_double.txt") as f:
	k = 0
	for line in f:
		if k == 0:
			N_d = int(line)
			k += 1
		else:
			data_diag_d.append(float(line))

N_f = 0

data_diag_f = []
with open("matrix_diag_float.txt") as f:
	k = 0
	for line in f:
		if k == 0:
			N_f = int(line)
			k += 1
		else:
			data_diag_f.append(float(line))
         
plt.title("Диагональ матрицы плотности ρ")
plt.xlabel("Номер элемента, i")
plt.ylabel("ρ(i,i)")

ax = plt.gca()

plt.plot(range(N_d), data_diag_d, label = 'Двойная точность', color = 'g')
plt.plot(range(N_f), data_diag_f, label = 'Одинарная точность', color = 'b')

plt.legend()
plt.grid(True, linestyle='-', color='0.75')

plt.savefig('graph_diag.png')
plt.show()