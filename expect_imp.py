import numpy as np
import itertools
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm

random.seed(101)

def target_f(x,y):

	f = -x*(x-1)*(x-3)*(x-4) + x + y*y

	return f

# Gaussian Process estimate of the mean and variance of the target function
def get_param(new_pt,data,target,xi):


	rbf = RBF(length_scale=2.0)
	gp = GaussianProcessRegressor(kernel=rbf).fit(data,target)
	new_pt = np.asarray(new_pt).reshape(1,2)
	pt_mean, pt_std = gp.predict(new_pt,return_std=True)

	Z = 0
	if pt_std != 0:
		Z = get_Z(xi,pt_mean,pt_std)

	return [pt_mean, pt_std, Z]

def get_Z(xi,gp_mean,gp_std):

	Z = 0
	if gp_std != 0:
		best_val = target_f(*best_pt)
		Z = (gp_mean - best_val -xi) / gp_std

	return Z			

# Compute Expectation improvement
def expect_imp(best_val,points,test_pts,test_val,xi):

	best_ei = None
	next_pt = None
	for pt in points:
		pt_mean, pt_std, Z = get_param(pt,test_pts,test_val,xi)

		pt_ei = 0
		if pt_std != 0:
			st_norm = norm()
			cdf_Z = st_norm.cdf(Z)
			pdf_Z = st_norm.pdf(Z)
			pt_ei = (pt_mean - best_val - xi) * cdf_Z + (pt_std * pdf_Z)

		if best_ei == None:
			best_ei = pt_ei
			best_pt = pt	
		elif pt_ei > best_ei:
			best_ei = pt_ei
			best_pt = pt
				
	return best_pt	

# Compare to random sample
def plot_target_f(x,y, test_pts, rand_pts):

	X, Y = np.meshgrid(x, y)
	Z = target_f(X, Y)
	plt.figure(figsize=(10,10))
	ax = plt.axes(projection='3d')
	ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none',alpha=0.5)
	ax.scatter(test_pts[:,0],test_pts[:,1],target_f(test_pts[:,0],test_pts[:,1])+0.1,marker="o",label="Expected Improvement",color='r',alpha=1,s=50)
	ax.scatter(rand_pts[:,0],rand_pts[:,1],target_f(rand_pts[:,0],rand_pts[:,1])+0.1,marker="o",label="Random Points",color='b',alpha=1,s=50)

	for x in range(len(test_pts)):
		ax.text(test_pts[x,0],test_pts[x,1],target_f(test_pts[x,0],test_pts[x,1])+0.1,s=str(x),alpha=1,size=15)
	plt.legend()
	plt.xlabel("x")
	plt.ylabel("y")
	plt.show()


##############################################
# Main
##############################################

# Setup data
x_pts = np.linspace(0, 4, 30)
y_pts = np.linspace(-2, 2, 30)

# data points
data_pts = list(itertools.product(x_pts,y_pts))

# VERY sensitive to initial point selection
next_pt =  (0.4137931,0)
#next_pt =  (2.0,0)
next_val = target_f(*next_pt)

test_pts = np.asarray([[*next_pt]])
test_vals = [next_val]
print()
print("Initial Point: ",test_pts)
print("Initial Value: ",test_vals)
print()

best_pt = (-1,-1)
best_val = -1
num = 5
xi = 0.01	# exploration parameter
for x in range(num):

	print()
	print("########################################################")
	print("iteration: ", x)
	print("next_pt: ",next_pt)
	print("next_val: ", target_f(*next_pt))

	# get max expected improvement given known points
	next_pt = expect_imp(next_val,data_pts,test_pts,test_vals,xi)

	test_pts = np.append(test_pts,[list(next_pt)],axis=0)
	next_val = target_f(*next_pt)
	test_vals.append(next_val)
	if next_val > best_val:
		best_val = next_val
		best_pt = next_pt
	print()
	print("test_pts")
	print(test_pts)
	print()
	print("test_vals")
	print(test_vals)
	print()

print("best_pt: ",best_pt)
print("best_val: ",best_val)
print()

# select num random points for comparison
rand_pts = random.choices(data_pts,k=num+1)
rand_pts = np.asarray(list(map(list,rand_pts)))

print("Random point_values")
print(target_f(rand_pts[:,0],rand_pts[:,1]))
print()

plot_target_f(x_pts, y_pts, test_pts, rand_pts)
