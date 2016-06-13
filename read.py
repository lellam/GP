import numpy as np
import matplotlib.pyplot as plt
""
def main():
	
	A = np.loadtxt("output/out.dat")
	plt.plot(np.exp(A))
	plt.show()
	print np.exp(A).mean()
	
	A = np.loadtxt("output/data.dat")
	A = np.transpose(A)
	plt.plot(A[0],A[1])
	plt.show()
	return

if __name__ == "__main__":
	main()