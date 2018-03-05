import control
import numpy as np
import scipy.integrate as integrate
from matplotlib.pyplot import *
import plot_helper

# parameters
k = 0.00    # friction
g = 9.81    # gravity

def nonlinear_pendulum(x, t, u):
    return [x[1], -k*x[1]-g*np.sin(x[0])+u*np.cos(x[0]), x[3], u]

def nonlinear_pendulum_feedback(x,t,K):
    u = np.dot(K,x)
    return nonlinear_pendulum(x,t,u)

# state: x = (ang, ang_vel, pos, vel)
# A =   [  0  1 0 0 ]
#       [ -g -k 0 0 ]
#       [  0  0 0 1 ]
#       [  0  0 0 0 ]
# B  = [ 0 0 0 1 ]

A = np.array([[0, 1, 0, 0], [-g, -k, 0, 0], [0, 0, 0, 1], [0,0,0,0]])
B = np.array([[0],[1],[0],[1]])
C = np.identity(4)
D = np.zeros((4, 1))

def linear_pendulum(x, t, u):
    return np.dot(A,x) + np.dot(B,u)

def linear_pendulum_feedback(x,t,K):
    u = np.dot(K,x)
    return linear_pendulum(x,t,u)

p = np.linalg.eig(A)
Kp = -np.array(control.place(A, B, [-1, -1, -1, -1]))
p_ = np.linalg.eig(A+np.dot(B,Kp))

Q = np.eye(4)
Q[0,0] = 1.0
R = 1.0e-3 * np.eye(1)
K, _, _ = control.lqr(A, B, Q, R)
K = -K

# simulation of the system without control
x0 = np.array([2.0, 0.0, 1.0, 0.0])
T = np.arange(0.0, 10.0, 0.05)
u = np.array([0.0])

xout = integrate.odeint(nonlinear_pendulum, x0, T, args=(u,))
#xout = integrate.odeint(nonlinear_pendulum_feedback, x0, T, args=(K,))

plot_helper.animate_pendulum(T, xout)
plot_helper.plot_state_trajectories(T, xout)

pass
