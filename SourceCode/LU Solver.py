from numpy import exp, linspace, zeros, ones, array, double
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve

def solution(x): #Exact Solution
	return 1.-(1.-exp(-10.))*x - exp(-10.*x)

def f(x): #Right Hand Side
	return 100.*exp(-10.*x)


N = raw_input('What value of N do you want? Type "exit" to close. \n ')

while N != 'exit':
	RT = [] #Values for runtimes tridiagonal solver
	RTLU = [] #Values for runtimes LU Solver
	N = int(N)
	for r in ones(10): #Runs program 10 times, to calculate average runtimes
		
		#Boundary Conditions, Array Preparation		
		X = linspace(0,1,N+1)
		h = (X[-1]-X[0])/N

		F = f(X)*(h**2); F_ = zeros(N+1)
		
		'''
		# Tridiagonal solver #	
		'''		

		V = zeros(N+1)
		d = 2.*ones(N+1); d_ = zeros(N+1)	
		d_[1] = d[1]; F_[1] = F[1]

		#Precalculations
		I = array(range(1,N))
		d_ = 1.+1.)/I
	
		#Calculation Loop
		start = timer()
		#Forward Substitution
		for i in range(2,N):
			F_[i] = F[i] + F_[i-1]/d_[i-2]
	
		#Backward Substitution
		V[N-1] = F_[N-1]/d_[N-2]
		for i in range(N-2,0,-1):
			V[i] = (F_[i]+V[i+1])/d_[i-1]

		end = timer()
		RT.append(end-start)
	
		'''
		# LU solver #
		'''		
		
		#Setting up	matrix A	
		A = zeros(shape=(N-1,N-1))
		for i in range(N-1):
			A[i][i] = 2.
			if (i != 0) and (i != N-2):
				A[i][i+1] = -1.	
				A[i][i-1] = -1.
			elif i == 0:
				A[i][i+1] = -1.
			elif i == N-2:
				A[i][i-1] = -1.
		
		#LU Solver		
		startLU = timer()
		LU, P = lu_factor(A) #LU Decomposition
		V_lu = lu_solve((LU,P),F[1:-1]) #LU Solver
		endLU = timer()
		RTLU.append(endLU-startLU)
		V_lu = array([0] + list(V_lu) + [0]) #Adding boundary terms

		#Plotting, if desired
		'''
		plt.figure()	
		plt.plot(X,solution(X),'g.',X,V,'b',X,V_lu,'r--')
		plt.title('Numerical vs exact solution, N=%d' % N)
		plt.legend(['Exact','Tridiagonal','LU solver'])
		plt.xlabel('x');plt.ylabel('v(x)')
		#plt.savefig('Task1e_%d.png' % N)
		plt.show()
		plt.close()		
		'''

	#Print average runtimes
	print "Average runtime tridiagonal solver: %f" % (float(sum(RT))/len(RT))
	print "Average runtime LU solver: %f" % (float(sum(RTLU))/len(RTLU))		
	N = raw_input('What value of N do you want? Type "exit" to close. \n ')

print "Thank you for running this program"







