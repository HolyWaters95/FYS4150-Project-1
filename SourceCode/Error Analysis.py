from numpy import exp, linspace, zeros, ones, array, log10
from timeit import default_timer as timer
import matplotlib.pyplot as plt

def solution(x): #Exact Solution
	return 1.-(1.-exp(-10.))*x - exp(-10.*x)

def f(x): #Right Hand Side
	return 100.*exp(-10.*x)


def PoissonSolver(N,f):

	#Boundary Conditions, Array Preparation
	N = int(N)
	X = linspace(0,1,N+1)
	h = (X[-1]-X[0])/N

	F = f(X)*(h**2); F_ = zeros(N+1)
	V = zeros(N+1)

	d = 2.*ones(N+1); d_ = zeros(N+1)
	d_[1] = d[1]; F_[1] = F[1]
	
	#Precalculations
	I = array(range(1,N))
	d_ = 2.-(I-1.)/I

	#Special Case Solver
	start = timer()
	#Forward Substitution
	for i in range(2,N):
		F_[i] = F[i] + F_[i-1]/d_[i-2]

	#Backwatd Substitution
	V[N-1] = F_[N-1]/d_[N-2]
	for i in range(N-2,0,-1):
		V[i] = (F_[i]+V[i+1])/d_[i-1]

	end = timer()
	print end-start
	return X,V,h



#Error Calculations for various N

H = [] #Values of h, stepsize
R = [] #Values for maximum relative error

for i in list(raw_input('What powers of 10 do you want for N? Enter as list, e.g. 1234 for 1,2,3,4.\n')):

	N = 10**int(i)
	x,v,h = PoissonSolver(N,f)
	u = solution(x)

	relerr = abs((v[1:-2] - u[1:-2])/u[1:-2])
	H.append(log10(h))
	R.append(log10(max(relerr)))
		
#Plotting
plt.plot(H,R,'b',H,R,'bo')
plt.title('Relative error as function of step size')
plt.xlabel('$log_{10}(h)$')
plt.ylabel('$log_{10}(\epsilon)$')
plt.show()



