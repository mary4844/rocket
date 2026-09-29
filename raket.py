# I funktionen för att räkna ut vinkeln, θ(t) så används en koefficient, 1/3, för att anpassa vinkeln för det specifika målet (80,60)
# eftersom det inte räcker att endast rikta raketen mot målet eftersom den då kommer att missa. Koefficientens värde har testats fram.
# Dessutom adderar man π till vinkeln för att för att vrida den ett halvt varv. Det gör man eftersom vinkeln är vad motorn ska peka åt 
# och inte riktningen för raketen, och motorn ska peka åt motsatt håll från raketen för att ge kraft åt raketen.

# I RK-4-lösaren så finns det ett villkor för att avbryta då 79.5< x-position <80 och 59.5< y-position <60.5 eftersom raketen då befinner 
# i princip rätt punkt. Anledningen till att villkoret inte skrivs med exakta värden är att lösaren använder sig av tiddsteg för att stega
# framåt i banan och man då hoppar över vissa punkter. Villkoret är inte med i solve_ivp. 

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


def m(t): # massan som en funktion av tiden
    if t <= 10:
        return 8 - 0.4*t
    else:
        return 4

def mprim(t): # derivatan av m(t)
    if t <= 10:
        return -0.4
    else:
        return 0

def F(t, v): # funktion för externa krafter som verkar på raketen
    return m(t)*g - (c * np.sqrt(v[0]**2 + v[1]**2) * v)

def θ(tillstånd): # funktion för att avgöra vilken vinkel motorn ska på åt
    if tillstånd[2] <= 20: # om positionen i y led inte överstiger 20, fortsätt rakt uppåt
        return -(np.pi/2)
    else:
        deltay = 60 - tillstånd[2]
        deltax = 80 - tillstånd[0]
        θ = (1/3)*np.arctan2(deltay, deltax) + np.pi # räknar ut vinkel baserat på raketens position i förhållande till målet
        return θ
    
def u(tillstånd):
    uvec = np.zeros(2)
    uvec[0]= k*np.cos(θ(tillstånd))
    uvec[1] = k*np.sin(θ(tillstånd))

    return uvec

# tillståndsvektor [x, vx, y, vy]
def rocket(t, tillstånd): # funktion för att räkna ut derivata
    vxy = np.array([tillstånd[1], tillstånd[3]]) # hastighetsvektor
    a = np.array(F(t, vxy)/m(t) + (mprim(t)/m(t)) * u(tillstånd)) # accelerationsvektor, (ax, ay)

    der = np.zeros(4)
    der[0] = tillstånd[1]
    der[1] = a[0]
    der[2] = tillstånd[3]
    der[3] = a[1]

    return der # der = [vx, ax, vy, ay], vx och vy hastighet i x- och y-led. ax och ay är accelerationen

def RK(f, tspan, bv, h): # RK-4 lösare
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

        if 79.5<state[0]<80.5:
            if 59.5<state[2]<60.5:
                break

    return res

tspan = [0,12] # tidspann
tillstånd = [0, 0, 0, 0] # begynnelsevillkor, [x-position, x-hastighet, y-position, y-hastigeht]

sol = solve_ivp(rocket, tspan, tillstånd, atol=1e-12, rtol=1e-12)
xs  = sol.y[0] 
ys  = sol.y[2]

h = 0.01 # tidssteg

res = RK(rocket, tspan, tillstånd, h)
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
