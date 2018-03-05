import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def plot_state_trajectories(T, xs):
    n = len(xs[0,:])
    nx = int(np.ceil(np.sqrt(n)))

    fig, axs = plt.subplots(nx, nx)

    stop = False
    for i in range(0, nx):
        for j in range(0, nx):
            axs[i,j].plot(T, xs[:,i*nx+j])
            axs[i,j].set_xlim([T[0], T[-1]])
            axs[i,j].set(xlabel='t')
            axs[i,j].set_title('x['+str(i*nx+j+1)+']')
            j += 1

            if i*nx+j >= n:
                stop = True
                break
        i += 1
        if stop:
            break

    #plt.show()

def animate_pendulum(T, xs):

    l = 1

    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-2, 2), ylim=(-1.2*l, 1.1*l))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    x1 = xs[:,2]
    y1 = np.zeros(xs[:,2].shape)

    x2 = l * np.sin(xs[:,0]) + x1
    y2 = l * np.cos(xs[:,0])

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        thisx = [x1[i], x2[i]]
        thisy = [y1[i], y2[i]]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (T[i]))
        return line, time_text

    t = (T[-1] - T[0])/len(T)   # time interval length
    ani = animation.FuncAnimation(fig, animate, frames=len(x1), interval=t*1000, blit=True, init_func=init, repeat=True)

    plt.show()

