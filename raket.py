import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

g = np.array([0, -9.82]) # m/s^2
c = 0.05 # kg/m
k = 700 # m/s

# tillståndsvektor [x, vx, y, vy]
# position = (x, y)
# hastighet = (vx, vy)


def m(t):
    if t <= 10:
        return 8 - 0.4*t
    else:
        return 4

def mprim(t):
    if t <= 10:
        return -0.4
    else:
        return 0

def F(t, v):
    return m(t)*g - (c * np.sqrt(v[0]**2 + v[1]**2) * v)

def θ(tillstånd):
    if tillstånd[2] < 20: # om positionen i y led inte överstiger 20, fortsätt rakt uppåt
        return -(np.pi/2)
    else:
        deltay = 60 - tillstånd[2]
        deltax = 80 - tillstånd[0]
        θ = np.arctan2(deltay, deltax) + np.pi
        return θ


    # elif (tillstånd[0]<80 and tillstånd[2]<60):
    #     deltay = 60 - tillstånd[2]
    #     deltax = 80 - tillstånd[0]
    #     θ = np.arctan((deltay/deltax)) + np.pi
    #     #print(θ,'1')
    #     return θ
    # elif (tillstånd[0]>80 and tillstånd[2]<60):
    #     deltay = 60 - tillstånd[2]
    #     deltax = 80 - tillstånd[0]
    #     θ = (np.arctan(-(deltay/deltax)) + np.pi)
    #     #print(θ,'2')
    #     return θ
    # elif (tillstånd[0]>80 and tillstånd[2]>60):
    #     deltay = 60 - tillstånd[2]
    #     deltax = 80 - tillstånd[0]
    #     θ = np.arctan((deltay/deltax))
    #     #print(θ,'3')
    #     return θ
    # elif (tillstånd[0]<80 and tillstånd[2]>60):
    #     deltay = 60 - tillstånd[2]
    #     deltax = 80 - tillstånd[0]
    #     θ =-np.arctan((deltay/deltax))
    #     #print(θ,'4')
    #     return θ

    


def u(tillstånd):
    uvec = np.zeros(2)
    uvec[0]= k*np.cos(θ(tillstånd))
    uvec[1] = k*np.sin(θ(tillstånd))

    return uvec

# tillståndsvektor [x, vx, y, vy]
def rocket(t, tillstånd):
    vxy = np.array([tillstånd[1], tillstånd[3]]) # hastighetsvektor
    a = np.array(F(t, vxy)/m(t) + (mprim(t)/m(t)) * u(tillstånd)) # accelerationsvektor, (ax, ay)

    print(a, tillstånd[3])

    der = np.zeros(4)
    der[0] = tillstånd[1]
    der[1] = a[0]
    der[2] = tillstånd[3]
    der[3] = a[1]

    return der # der = [vx, ax, vy, ay], vx och vy hastighet i x- och y-led. ax och ay är accelerationen

def RK(f, tspan, bv, h):
    interval = round((tspan[1]-tspan[0])/h)
    time = np.linspace(tspan[0], tspan[1], interval + 1)

    res = np.zeros((len(time), len(bv)))

    state = bv
    res[0,:] = state

    for i in range(len(time)-1):
        k1 = f(time[i], state)
        k2 = f(time[i]+(h/2), state+((h/2)*k1))
        k3 = f(time[i]+(h/2), state+((h/2)*k2))
        k4 = f(time[i+1], state+h*k3)
        newstate = state + (h/6)*(k1+2*k2+2*k3+k4)

        res[i+1,:] = newstate
        state = newstate

        #print(state)

        if 79.5<state[0]<80.5:
            if 59.5<state[2]<60.5:
                break

    return res


tspan = [0,15]
tillstånd = [0, 0, 0, 0]

sol = solve_ivp(rocket, tspan, tillstånd, atol=1e-12, rtol=1e-12)
xs  = sol.y[0] 
ys  = sol.y[2]

res = RK(rocket, tspan, tillstånd, 0.1)
x = res[:,0]
y = res[:,2]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(xs, ys, label='solve_ivp', color='blue')
ax1.scatter([80], [60], color='green', label='mål')  # markera målet
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Bana med solve_ivp')
ax1.axis('equal')
ax1.grid(True)
ax1.legend()

ax2.plot(x, y, label='RK', color='red')
ax2.scatter([80], [60], color='green', label='mål')  # markera målet
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Bana med Runge Kutta')
ax2.axis('equal')  
ax2.grid()
ax2.legend()

plt.tight_layout()
plt.show()
