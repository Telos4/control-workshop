import control
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import plot_helper

# parameters
m = 0.3    # friction
M = 1
r = 15
g = 9.81    # gravity

# state: x = (ang, ang_vel, pos, vel)
def nonlinear_pendulum(x, t, u):
    # (phi, dphi, x, dx) = (x[0], x[1], x[2], x[3])
    return [x[1],
            (g * (m+M)*np.sin(x[0])+np.cos(x[0])*(u+x[1]**2*m*r*np.sin(x[0])))/(r*(m+M+np.cos(x[0])**2)),
            x[3],
            (u+m*(x[1]**2*r-g*np.cos(x[0]))*np.sin(x[0]))/(m+M+m*np.cos(x[0])**2)]

def nonlinear_pendulum_feedback(x,t,K):
    u = np.dot(K,x)
    return nonlinear_pendulum(x,t,u)

A = np.array([[0, 1, 0, 0], [g*(M+m)/(2*m+M)*r, 0, 0, 0], [0, 0,  0, 1], [-g*m/(2*m+M),0,0,0]])
B = np.array([[0],[1/(2*m+M)*r],[0],[1/(2*m+M)]])

C = control.ctrb(A,B)

Q = np.eye(4)
Q[2,2] = 1.0
Q[3,3] = 1.0
R = 1.0 * np.eye(1)
K, _, _ = control.lqr(A, B, Q, R)
K = -K

#F = -np.array(control.place(A, B, [-1, -1, -1, -1]))

#M = A + np.dot(B, F)

# simulation of the system without control
x0 = np.array([0.1, 0.0, 0.0, 0.0])
T = np.arange(0.0, 15.0, 0.05)
u = np.array([0.0])

xout = integrate.odeint(nonlinear_pendulum, x0, T, args=(u,))
#xout_l = integrate.odeint(linear_pendulum, x0, T, args=(u,))
#xout = integrate.odeint(nonlinear_pendulum_feedback, x0, T, args=(K,))

plot_helper.animate_pendulum(T, xout)
plot_helper.plot_state_trajectories(T, xout)
plt.show()

pass


