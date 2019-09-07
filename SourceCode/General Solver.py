from numpy import exp, linspace, zeros, ones
from timeit import default_timer as timer
import matplotlib.pyplot as plt

def solution(x): #Exact solution
	return 1.-(1.-exp(-10.))*x - exp(-10.*x)

def f(x): #Right hand side
	return 100.*exp(-10.*x)

Save = raw_input('Do you want to save plots? Type y or n. \n')
N = raw_input('What value of N do you want? Type "exit" to close. \n ')

while N != 'exit':

	#Boundary Conditions, Array preparation	
	N = int(N)
	X = linspace(0,1,N+1)
	h = (X[-1]-X[0])/N

	F = f(X)*(h**2); F_ = zeros(N+1)
	V = zeros(N+1)

	d = 2.*ones(N+1); d_ = zeros(N+1)
	e = -1.*ones(N);  c = -1.*ones(N); 
	d_[1] = d[1]; F_[1] = F[1]

	#Tridiagonal Solver
	start = timer()
	
	#Forward Substitution
	for i in range(2,N): 
		d_[i] = d[i] - e[i-1]*(c[i-1]/d_[i-1])
		F_[i] = F[i] - F_[i-1]*(c[i-1]/d_[i-1])

	#Backward Substitution	
	V[N-1] = F_[N-1]/d_[N-1]
	for i in range(N-2,0,-1):
		V[i] = (F_[i]-e[i]*V[i+1])/d_[i]

	end = timer()
	print end-start #Print algorithm Runtime

	#Plotting	
	plt.figure()	
	plt.plot(X,solution(X),'ro',X,V,'b')
	plt.title('Numerical vs exact solution, N=%d' % N)
	plt.legend(['Exact','Numerical'])
	plt.xlabel('x');plt.ylabel('v(x)')
	if Save == "y":
		plt.savefig('Task1b_%d.png' % N)
	plt.show()

	N = raw_input('What value of N do you want? Type "exit" to close. \n ')

print "Thank you for running this program"
