import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy as sp
norm = np.linalg.norm

class Cercle():
    def __init__(self, x, y, r, ):
        self.x = x
        self.y = y
        self.r = r
        self.n_tranches = 2 * m.ceil(self.r / e_tranches)
    
    def init_real(self):
        self.xs = x_s(self)
        self.alphas = alpha_s(self, self.xs)
        self.Ns, self.Ts, self.Us, self.Ws = NTUW_s(self, self.xs)
        self.ls = l_s(self, self.xs)
    
    def h(self, x):
        if (not isinstance(x, np.float64)) and (not isinstance(x, float)):
            x = x[0]
        if (x < self.x - self.r): return self.y + 2 * (self.x - self.r - x)
        if (x > self.x + self.r): return self.y + 2 * (x - self.x - self.r)
        return self.y - m.sqrt(self.r**2 - (x - self.x)**2)
        
### Calculs géométrie
class Sol():
    def __init__(self):
        self.H = 12
        self.pente_left = 1/2
        self.pente_right = -12/50
        self.crest = 6
        self.p1 = np.array((0, 0, 0))
        self.p2 = np.array((24, 12, 0))
        self.p3 = np.array((30, 12, 0))
        self.p4 = np.array((80, 0, 0))

    def h_sol(self, x):
        if (x <= 0): return 0
        if (x <= 24): return self.pente_left * x
        if (x <= 30): return 12
        if (x <= 80): return 12 + self.pente_right * (x - 30)
        return 0
    
    def h_water(self, x):
        if (x <= 22): return 11
        if (x >= 80): return 0
        return 11 - (11 / 58) * (x - 22)
    
    def distance(self, x, y):
        d_sol = y
        d_crest = y - self.H if y > self.H else np.inf
        d_left = norm(np.cross(self.p2 - self.p1, self.p1 - np.array((x, y, 0)))) / norm(self.p2 - self.p1)
        d_right = norm(np.cross(self.p4 - self.p3, self.p3 - np.array((x, y, 0)))) / norm(self.p4 - self.p3)
        if (x < 24): return min(d_sol, d_left)
        if (x < 30): return d_crest
        return min(d_sol, d_right)

def calc_alpha(x_l, x_r, cercle):
    if (abs(x_l - x_r) < 1e-3): return 0
    h_c_l, h_c_r = cercle.h(x_l), cercle.h(x_r)
    return m.atan((h_c_r - h_c_l) / (x_r - x_l))

### Calcul des poids
def hauteur_seche(x, cercle):
    h_s = sol.h_sol(x)
    h_c = cercle.h(x)
    h_w = sol.h_water(x)
    if (h_s < h_c): return 0
    if (h_s < h_w): return 0
    return h_s - max(h_c, h_w)

def hauteur_eau(x, cercle):
    h_s = sol.h_sol(x)
    h_c = cercle.h(x)
    h_w = sol.h_water(x)
    if (h_s < h_c): return 0
    if (h_w < h_c): return 0
    return h_w - h_c

def hauteur_sat(x, cercle):
    h_s = sol.h_sol(x)
    h_c = cercle.h(x)
    h_w = sol.h_water(x)
    if (h_s < h_c): return 0
    if (h_w < h_c): return 0
    return min(h_s, h_w) - h_c

def calc_weight_ww(x_l, x_r, cercle):
    w_dry = gamma_dry * sp.integrate.quad(hauteur_seche, x_l, x_r, args=(cercle, ))[0]
    w_water = gamma_w * sp.integrate.quad(hauteur_eau, x_l, x_r, args=(cercle, ))[0]
    w_sat = (gamma_sat - gamma_w) * sp.integrate.quad(hauteur_sat, x_l, x_r, args=(cercle, ))[0]
    return w_dry + w_water + w_sat, w_water

def calc_NTUW(x_l, x_r, cercle):
    w, w_water = calc_weight_ww(x_l, x_r, cercle)
    alpha = calc_alpha(x_l, x_r, cercle)
    return w * m.cos(alpha), w * m.sin(alpha), w_water / m.cos(alpha), w

### Création tranches
def x_s(cercle):
    xs_ = []
    for i in range(cercle.n_tranches + 1):
        x = cercle.x - (e_tranches * cercle.n_tranches / 2) + i * e_tranches
        xs_.append(max(min(x, cercle.x + cercle.r), cercle.x - cercle.r))
        if (cercle.h(x) > sol.h_sol(x)):
            xs_[i] = sp.optimize.root(lambda x_: cercle.h(x_) - sol.h_sol(x_), (x)).x[0]
    return np.array(xs_)

def alpha_s(cercle, xs_):
    alphas = []
    for i in range(cercle.n_tranches):
        alphas.append(calc_alpha(xs_[i], xs_[i+1], cercle))
    return np.array(alphas)

def NTUW_s(cercle, xs_):
    N, T, U, W = [], [], [], []
    for i in range(cercle.n_tranches):
        n, t, u, w = calc_NTUW(xs_[i], xs_[i+1], cercle)
        if (xs_[i+1] - xs_[i] < 0.01): n, t, u, w = 0, 0, 0, 0
        N.append(n)
        T.append(t)
        U.append(u)
        W.append(w)
    return np.array(N), np.array(T), np.array(U), np.array(W)

def l_s(cercle, xs_):
    ls = []
    for i in range(cercle.n_tranches):
        l = m.sqrt((xs_[i+1] - xs_[i])**2 + (cercle.h(xs_[i+1]) - cercle.h(xs_[i]))**2)
        if (xs_[i+1] - xs_[i] < 0.01): l = 0
        ls.append(l)
    return np.array(ls)

### Calcul F.S.
def fellenius(cercle):
    numerator = 0
    denominator = 0
    Ns, Ts, Us = cercle.Ns, cercle.Ts, cercle.Us
    if (cercle.x > 27): Ts = -Ts
    ls = cercle.ls
    for tranche in range(cercle.n_tranches):
        numerator += (c * ls[tranche]) + ((Ns[tranche] - Us[tranche]) * m.tan(phi))
        if (Ts[tranche] < 0):
            numerator += abs(Ts[tranche])
        else:
            denominator += Ts[tranche]
    return numerator / denominator

def bishop(cercle, FS_0):
    numerator = 0
    denominator = 0
    Ts, Us, Ws = cercle.Ts, cercle.Us, cercle.Ws
    if (cercle.x > 27): Ts = -Ts
    ls = cercle.ls
    alphas = cercle.alphas
    for tranche in range(cercle.n_tranches):
        numnum = (c * ls[tranche]) + (m.tan(phi) / m.cos(alphas[tranche])) * (Ws[tranche] - Us[tranche] * m.cos(alphas[tranche]))
        dennum = 1 + (m.tan(phi) * m.tan(abs(alphas[tranche])) / FS_0)
        numerator += numnum / dennum
        if (Ts[tranche] < 0):
            numerator += abs(Ts[tranche])
        else:
            denominator += Ts[tranche]
    return numerator / denominator

### Dommées géométriques
sol = Sol()

### Données du sol
c = 10
phi = m.radians(25)
gamma_dry = 18
gamma_sat = 20
gamma_w = 10

### Données de calcul
e_tranches = 1/2

def fs(x, y, r):
    cercle = Cercle(x, y, r)
    cercle.init_real()
    FS0 = fellenius(cercle)
    print(FS0)
    FS_old = FS0
    FS = bishop(cercle, FS0)
    while (abs(FS - FS_old) > 1e-6):
        print(FS)
        FS_old = FS
        FS = bishop(cercle, FS)
    return FS

def find_worst_circle():
    xs = np.arange(-3, 27, 1)
    ys = np.arange(0, 100, 10)
    rs = np.ones((len(xs), len(ys))) * (-1)
    fss = np.full((len(xs), len(ys)), np.inf)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if (y < sol.h_sol(x)): continue
            print(f"Calcul de fs pour x={x} et y={y}")
            r_min = max(1, int(sol.distance(x, y)) + 3)
            r_max = 30 - x if x < 27 else x - 24
            for r in np.arange(r_min, r_max, 1):
                cur_fs = fs(x, y, r)
                if (cur_fs < fss[i, j]):
                    fss[i, j] = cur_fs
                    rs[i, j] = r
            print(rs[i, j])
            print(fss[i, j])
    return xs, ys, rs, fss


"""
xs, ys, rs, fss = find_worst_circle()
worst_fs = np.min(fss)
coord = np.unravel_index(np.argmin(fss), fss.shape)
x = xs[coord[0]]
y = ys[coord[1]]
r = rs[coord]
print(f"Le pire facteur de sécurité est {worst_fs} pour un cercle de centre ({x}, {y}) et de rayon {r}")
"""
print(fs(62, 60, 60))