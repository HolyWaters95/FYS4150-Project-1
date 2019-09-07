from numpy import exp, linspace, zeros, ones, array
from timeit import default_timer as timer
import matplotlib.pyplot as plt

def solution(x): #Exact Solution
	return 1.-(1.-exp(-10.))*x - exp(-10.*x)

def f(x): #Right Hand Side
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
 
	d_ = zeros(N+1)
	d_[1] = 2; F_[1] = F[1]

	#Precalculations
	I = array(range(1,N))
	d_ = 1.+1./I

	#Special Case Solver
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

	#Plotting
	plt.figure()	
	plt.plot(X,solution(X),'ro',X,V,'b')
	plt.title('Numerical vs exact solution, N=%d' % N)
	plt.legend(['Exact','Numerical'])
	plt.xlabel('x');plt.ylabel('v(x)')
	if Save == "y":	
		plt.savefig('Task1c_%d.png' % N)
	plt.show()

	N = raw_input('What value of N do you want? Type "exit" to close. \n ')

print "Thank you for running this program"
