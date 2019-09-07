from numpy import exp, linspace, zeros, ones, array, log10
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve

def solution(x): #Exact Solution
	return 1.-(1.-exp(-10.))*x - exp(-10.*x)

def f(x): #Right Hand Side
	return 100.*exp(-10.*x)


def PoissonSolver(N,f): #Tridiagonal Solver

	#Boundary Conditions, Array preparation
	N = int(N)
	X = linspace(0,1,N+1)
	h = (X[-1]-X[0])/N

	F = f(X)*(h**2); F_ = zeros(N+1)
	V = zeros(N+1)

	d = 2.*ones(N+1); d_ = zeros(N+1)
	d_[1] = d[1]; F_[1] = F[1]
	
	#Precalculations
	I = array(range(1,N))
	d_ = 1.+1.)/I

	#Tridiagonal Solver
	start = timer()
	#Forward Substitution
	for i in range(2,N):
		F_[i] = F[i] + F_[i-1]/d_[i-2]

	#Backward Substitution
	V[N-1] = F_[N-1]/d_[N-2]
	for i in range(N-2,0,-1):
		V[i] = (F_[i]+V[i+1])/d_[i-1]

	end = timer()
	print end-start
	return X,V,h
	####
	
def LU_solver(N,f):
	#Boundary Conditions, Array Setup	
	X = linspace(0,1,N+1)
	h = (X[-1]-X[0])/N
	F = f(X)*(h**2)
	
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
		
	#LU Solver Algorithm	
	startLU = timer()
	LU, P = lu_factor(A)
	V_lu = lu_solve((LU,P),F[1:-1])
	endLU = timer()
	V_lu = array([0] + list(V_lu) + [0])
	return V_lu

H = [] #Values for h, stepsize
R = [] #Values for relative error, tridiagonal solver
RLU = [] #Values for relative error, LU solver

#Calculate errors for different Solvers
for i in list(raw_input('What powers of 10 do you want for N? Enter as list, eg 1234 for 1, 2, 3 and 4.\n')):

	N = 10**int(i)
	x,v,h = PoissonSolver(N,f)
	u = solution(x)
	v_lu = LU_solver(N,f)

	relerr = abs((v[1:-2] - u[1:-2])/u[1:-2])
	relerrLU = abs((v_lu[1:-2] - u[1:-2])/u[1:-2])
	H.append(log10(h))
	R.append(log10(max(relerr)))
	RLU.append(log10(max(relerrLU)))		

#Print relative errors and their difference
print R
print RLU
print array(R)-array(RLU) 

#Plotting, if desired
'''
plt.plot(H,array(R)-array(RLU))
plt.title('Relative error as function of step size')
plt.xlabel('$log_{10}(h)$')
plt.ylabel('$log_{10}(\epsilon)$')
plt.show()
'''
