import os
import sys
import timeit
import numpy as np
import pickle as pkl
import scipy.stats as ss
from scipy import integrate
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# python SpkAMP.py 1.0 2000 3.0 10 sigscale 1.0 Rad

rho = 0.5
N = 2000
lmda = 3.0
C = 5
f = "sigscale"
mu = 1.00
t = 100

nbins = 500   # Number of bins

d = "Rad"
dist = {
  "Ber": 0,
  "Gau": 1,
  "Poi": 2,
  "Rad": 3,
  "Uni": 4,
}

d_f = {
  "x": [lambda x: lmda/(x + 1), "$\lambda/x$"],
  "exp": [lambda x: lmda/np.exp(x), "$\lambda/e^x$"],
  "sigshift": [lambda x: lmda*(0.5 + 1/(1 + np.exp(x))),
                          "$\lambda(1/2 + 1/(1 + e^x)$"],
  "sigscale": [lambda x: 2*lmda/(1 + np.exp(x)), "$2\lambda/(1 + e^x)$"]
}

if len(sys.argv) > 1:
  rho, N, lmda, C, f, mu, d = sys.argv[1:]
  rho = float(rho)
  N = int(N)
  lmda = float(lmda)
  C = int(C)
  mu = float(mu)

r = (-3*lmda, 3*lmda)
logbins = np.logspace(np.log10(10**-6), np.log10(10**10), 32)
uni_normal = lambda x, c, mu: c*np.exp(-x**2/(2*mu))
# Rademacher
tri_normal = lambda x, c, mu: c*(rho*(np.exp(-(x - mu)**2/(2*mu)) + 
                                      np.exp(-(x + mu)**2/(2*mu))) + 
                                 (1 - rho)*np.exp(-x**2/(2*mu)))

def rejection_mthd_Sstc():
  xi = 27/80
  a2 = 2/3
  i = 0
  samples = np.zeros(N)
  while i < N:
    x = 2*(np.random.random()*2 - 1)
    if np.random.random()*0.39 <= \
        xi*(6*a2**2 + 2*a2*x**2 + x**4)*np.sqrt(4*a2 - x**2)/(2*np.pi):
      samples[i] = x
      i += 1
  return samples

def rejection_mthd_Qrtc():
  gma = (8 - 9*mu + np.sqrt(64 - 144*mu + 108*mu**2 - 27*mu**3))/27
  a2 = (np.sqrt(mu**2 + 12*gma) - mu)/(6*gma)
  i = 0
  samples = np.zeros(N)
  while i < N:
    x = 2*(2*np.random.random() - 1)*np.sqrt(a2)
    if np.random.random()*0.39 <= \
        (mu + 2*a2*gma + gma*x**2)*np.sqrt(4*a2 - x**2)/(2*np.pi):
      samples[i] = x
      i += 1
  return samples

def Spike(diagZ = True):
  # Message
  if dist[d] == 0:
    # Bernoulli
    x = np.random.binomial(1, rho, N)
  elif dist[d] == 1:
    # Gaussian
    x = np.random.normal(rho, 1, N)
  elif dist[d] == 2:
    # Poisson
    pass
  elif dist[d] == 3:
    # Rademacher
    x = np.power(-1, np.random.binomial(1, 0.5, N))
    x[np.random.uniform(0, 1, N) < 1 - rho] = 0
  elif dist[d] == 4:
    # Uniform
    pass
  x = np.outer(x, x)

  if mu != 1.:
    # Structured Quartic Noise
    W = ss.ortho_group.rvs(N)
    if mu > 1.:
      L = np.diag(rejection_mthd_Sstc())
    else:
      L = np.diag(rejection_mthd_Qrtc())
    W = W@L@W.T
    # Noisy Data Matrix
    Y = (lmda/N)*x + W
  else:
    # Wigner Matrix
    W = np.random.randn(N, N)
    W = (W + W.T)/np.sqrt(2)
    # Noisy Data Matrix
    Y = np.sqrt(lmda/N)*x + W

  if diagZ:
    # Zero Diagonal
    np.fill_diagonal(Y, 0)
  return Y, x

mean = lambda x: lmda*np.sqrt(1 - x)*rho
d_eta = None
if dist[d] == 0:
  # Bernoulli
  eta = lambda x, y: (1 + (1 - rho)*np.exp(y/2 - x)/rho)**(-1)
  d_eta = lambda x, y: (rho - 1)*np.exp(y/2 - x)/rho
elif dist[d] == 1:
  # Gaussian
  eta = lambda x, y: (x + rho)/(y + 1)
  # d_eta = lambda x, y: 1/(x + rho)
  d_eta = lambda x, y: (y + 1)/(x + rho)**2
elif dist[d] == 2:
  # Poisson
  pass
elif dist[d] == 3:
  # Rademacher
  eta = lambda x, y: np.sinh(x)/(np.cosh(x) + np.exp(y/2)*(1 - rho)/rho)
  d_eta = lambda x, y: (1 + (1 - rho)*np.cosh(x)*np.exp(y/2)/rho)/np.sinh(x)**2
elif dist[d] == 4:
  # Uniform
  pass

def SE(min_eps = 1e-15):
  eps = 1e-7
  q = [eps - 1, eps]
  while np.abs(q[-1] - q[-2]) > min_eps:
    r = lmda*q[-1]
    q.append(
      integrate.quad(
        lambda Z: np.exp(-(Z**2)/2)*eta(r + np.sqrt(r)*Z, r)/np.sqrt(2*np.pi),
        -20, 20, epsrel=1e-7, epsabs=0)[0])
  return 1 - np.array(q[1:])**2

def cavities(b, v, Y):
  Y2 = Y**2
  c = 1
  if mu == 1.:
    c = lmda/N
  O = c*Y2.dot(v)
  return (np.sqrt(c)*Y.dot(b[1]) - O*b[0], c*(v + norm(b[1], 2)**2) - O, O)

def marginals(b, v, f = None):
  p = eta(b, v)
  if f is None:
    f = d_eta(b, v)
  return (p, p**2*f, f)

def AMP(Y, x, B, MSE, Mrg, Ons, min_eps = 1e-12):
  # Initialization
  k = t
  b = (1e-3)*np.random.randn(2, N)
  v = (1e-3)*np.random.randn(N)
  # AMP algorithm
  while k and np.mean((b[1] - b[0])**2) > min_eps:
    cB, cV, O = cavities(b, v, Y)
    # Onsager Analysis
    O = O.mean()
    mse = (norm(np.outer(b[1], b[1]) - x)/(N*rho))**2
    MSE[t - k] += mse
    Ons[t - k] += O
    hist = np.histogram(cB, bins=nbins, range=r)[0]
    B[t - k] += hist/N
    b[0], b[1], v, M = [b[1], *marginals(cB, cV)]
    # Marginal Analysis
    M = M.mean()
    Mrg[t - k] += M
    k -= 1
  a = t - k + 1
  while k:
    Mrg[t - k] += M
    MSE[t - k] += mse
    Ons[t - k] += O
    B[t - k] += hist/N
    k -= 1
  return MSE, Mrg, B, Ons, a

def LAMP(Y, x, B, MSE, MRG, min_eps = 1e-12):
  # Initialization
  k = t
  b = (1e-3)*np.random.randn(2, N)
  v = (1e-3)*np.random.randn(N)
  while k and np.mean((b[1] - b[0])**2) > min_eps:
    cB, cV, _ = cavities(b, v, Y)
    mse = (norm(np.outer(b[1], b[1]) - x)/(N*rho))**2
    MSE[t - k] += mse
    hist = np.histogram(cB, bins=nbins, range=r, density=True)[0]
    B[t - k] += hist/N
    b[0], b[1], v, _ = [b[1], *marginals(cB, cV, MRG[t - k]*np.ones(N))]
    k -= 1
  while k:
    MSE[t - k] += mse
    B[t - k] += hist/N
    k -= 1
  return MSE, B

def BAMP(Y, x, B, MSE, MRG, min_eps = 1e-12):
  # Initialization
  k = t
  b = np.zeros((t + 1, N), dtype=float)
  b[0] = (1e-3)*np.random.randn(N)
  if mu > 1.:
    # Sestic
    xi = 27/80
    while k and np.mean((b[1] - b[0])**2) > min_eps:
      cB = xi*(lmda*np.linalg.matrix_power(Y, 5) -
               lmda**2*(np.linalg.matrix_power(Y, 4) +
                        np.linalg.matrix_power(Y, 2))).dot(b[t - k]) \
           - MRG[t - k]*b[t - k]
           #- np.sum([MRG[i]*b[i] for i in range(1, t - k)])
      mse = (norm(np.outer(b[1], b[1]) - x)/(N*rho))**2
      MSE[t - k] += mse
      hist = np.histogram(cB, bins=nbins, range=r, density=True)[0]
      B[t - k] += hist/N
      b[t - k + 1] = eta(cB, 0)
      k -= 1
  else:
    # Quartic
    gma = (8 - 9*mu + np.sqrt(64 - 144*mu + 108*mu**2 - 27*mu**3))/27
    while k and np.mean((b[1] - b[0])**2) > min_eps:
      cB = lmda*(mu*Y - gma*(np.linalg.matrix_power(Y, 2)*lmda -
                             np.linalg.matrix_power(Y, 3))).dot(b[t - k]) - \
          np.sum([MRG[i]*b[i] for i in range(1, t - k)])
      mse = (norm(np.outer(b[1], b[1]) - x)/(N*rho))**2
      MSE[t - k] += mse
      hist = np.histogram(cB, bins=nbins, range=r, density=True)[0]
      B[t - k] += hist/N
      b[t - k + 1] = eta(cB, 0)
      k -= 1
  while k:
    MSE[t - k] += mse
    B[t - k] += hist/N
    k -= 1
  return MSE, B

def main():
  force = True
  s = f"_{mu:05.2f}_{d}_{rho:04.2f}_{N:05d}_{lmda:05.2f}_{C:04d}"
  print(s)
  global t
  # State Evolution
  se =  SE()
  if len(se) > t:
    MSE_SE = se
    t = len(se)
  else:
    MSE_SE = np.zeros(t)
    MSE_SE[:len(se)] = se
    MSE_SE[len(se):] = se[-1]
  ts = timeit.default_timer()
  B = np.zeros((t, nbins), dtype=float)
  MSE = np.zeros(t, dtype=float)
  Ons = np.zeros(t, dtype=float)
  MRG = np.zeros(t, dtype=float)
  B_L = np.zeros((t, nbins), dtype=float)
  MSE_L = np.zeros(t, dtype=float)
  B_F = np.zeros((t, nbins), dtype=float)
  MSE_F = np.zeros(t, dtype=float)
  MRG_F = d_f[f][0](np.arange(t))
  B_B = np.zeros((t, nbins), dtype=float)
  MSE_B = np.zeros(t, dtype=float)
  bins = np.linspace(*r, num=nbins + 1, endpoint=True, dtype=float)[:-1]
  p = f"../data/docs/Spike{'_'.join(s.split('_')[:-1])}" + "_{}.pkl"
  for i in range(C):
    if not os.path.isfile(p.format(f"{i:04d}")) or force:
      with open(p.format(f"{i:04d}"), 'wb') as fp:
        pkl.dump(Spike(), fp)
  for i in range(C):
    with open(p.format(f"{i:04d}"), 'rb') as fp:
      MSE, MRG, B, Ons, _ = AMP(*pkl.load(fp), B, MSE, MRG, Ons)
  B /= C
  MSE /= C
  MRG /= C
  Ons /= C
  for i in range(C):
    with open(p.format(f"{i:04d}"), 'rb') as fp:
      spike = pkl.load(fp)
    MSE_L, B_L = LAMP(*spike, B_L, MSE_L, Ons)
    MSE_F, B_F = LAMP(*spike, B_F, MSE_F, MRG_F)
  if mu != 1.:
    for i in range(C):
      with open(p.format(f"{i:04d}"), 'rb') as fp:
        MSE_B, B_B = BAMP(*pkl.load(fp), B_B, MSE_B, MRG_F)
  MSE_L /= C
  B_L /= C
  MSE_F /= C
  B_F /= C
  MSE_B /= C
  B_B /= C
  plt.figure(figsize=(5, 4))
  plt.xlim(0, t)
  plt.ylim(0, 1.5)
  plt.xlabel('Iteration')
  plt.ylabel('MSE')
  if mu == 1.:
    plt.plot(MSE_SE, '-', label="SE", c='r')
    plt.plot(MSE, 'o', label="AMP", c='b', markersize=2)
  plt.plot(MSE_L, 'o', label="$\mathbb{O}$", c='g', markersize=2)
  plt.plot(MSE_F, 'o', label=d_f[f][1], c='k', markersize=2)
  if mu != 1.:
    plt.plot(MSE_B, '^', label=d_f[f][1], markersize=2)
  plt.legend()
  plt.yscale("log")
  plt.savefig(f"../img/SE{s}.pdf", format="pdf")
  plt.clf()

  plt.figure(figsize=(5, 4))
  plt.bar(bins, B[-1], width=np.diff(bins)[0], color="b", alpha=0.3)
  m_mse = mean(MSE[-1])
  m_se = mean(MSE_SE[-1])
  m_l = mean(MSE_L[-1])
  m_f = mean(MSE_F[-1])
  m_b = mean(MSE_B[-1])
  if dist[d] == 0:
    # Bernoulli
    c = np.sqrt(lmda/N)*np.ones(t)
    plt.plot(bins,
             ss.skewnorm.pdf(bins, -m_se*rho, m_se/rho, m_se/rho)*c[0],
             c='r', label="SE $\mathcal{N}($" + \
                          f"{m_se:.2e}, {np.sqrt(m_se):.2e})")
    plt.plot(bins, ss.skewnorm.pdf(bins, -m_mse, m_mse/rho, m_mse/rho)*c[0],
             c='b', label="MSE $\mathcal{N}($" + \
                          f"{m_mse:.2e}, {np.sqrt(m_mse):.2e})")
    plt.plot(bins, ss.skewnorm.pdf(bins, -m_l, m_l/rho, m_l/rho)*c[0], c='g',
             label="$\mathbb{O}$ $\mathcal{N}($" +
                   f"{m_l:.2e}, {np.sqrt(m_l):.2e})")
    plt.plot(bins, ss.skewnorm.pdf(bins, -m_f, m_f/rho, m_f/rho)*c[0], c='k',
             label=d_f[f][1] + " $\mathcal{N}($" +
                   f"{m_f:.2e}, {np.sqrt(m_f):.2e})")
  elif dist[d] == 1:
    # Gaussian
    c = [curve_fit(uni_normal, bins, B[i], p0=(0.5, mean(MSE[i])))[0]
         for i in range(t)]
    c = np.sqrt(lmda/N)*np.ones(t)
    plt.plot(bins, uni_normal(bins, c[0], m_se), c='r',
             label="SE $\mathcal{N}(0, $" + f"{np.sqrt(m_se):.2e})")
    plt.plot(bins, uni_normal(bins, c[0], m_mse), c='b',
             label="MSE $\mathcal{N}(0, $" + f"{np.sqrt(m_mse):.2e})")
    plt.plot(bins, uni_normal(bins, c[0], m_l), c='g',
             label="$\mathbb{O}$ $\mathcal{N}(0, $" + f"{np.sqrt(m_l):.2e})")
    plt.plot(bins, uni_normal(bins, c[0], m_f), c='k',
             label=d_f[f][1] + " $\mathcal{N}(0, $" + f"{np.sqrt(m_f):.2e})")
  elif dist[d] == 2:
    # Poisson
    pass
  elif dist[d] == 3:
    # Rademacher
    c = [curve_fit(tri_normal, bins, B[i], p0=(0.5, mean(MSE[i])))[0]
         for i in range(t)]
    if mu == 1.:
      plt.plot(bins, tri_normal(bins, c[-1][0], m_se), c='r',
               label="SE $\mathcal{N}_2($" + 
                     f"{m_se:.2e}, {np.sqrt(m_se):.2e})")
      plt.plot(bins, tri_normal(bins, c[-1][0], m_mse), c='b',
               label="MSE $\mathcal{N}_2($" +
                     f"{m_mse:.2e}, {np.sqrt(m_mse):.2e})")
    plt.plot(bins, tri_normal(bins, c[-1][0], m_l), c='g',
             label="$\mathbb{O}$ $\mathcal{N}_2($" +
                   f"{m_l:.2e}, {np.sqrt(m_l):.2e})")
    plt.plot(bins, tri_normal(bins, c[-1][0], m_f), c='k',
             label=d_f[f][1] + " $\mathcal{N}_2($" +
                   f"{m_f:.2e}, {np.sqrt(m_f):.2e})")
    if mu != 1.:
      plt.plot(bins, tri_normal(bins, c[-1][0], m_b), label=d_f[f][1] + \
               " $\mathcal{N}_2($" + f"{m_b:.2e}, {np.sqrt(m_b):.2e})")
  plt.ylabel("Cavity Mean (AMP) Dist.")
  plt.legend()
  plt.savefig(f"../img/AMP{s}.pdf", format="pdf")

  header = f"# Parameters: N = {N}, Dist = Rad({rho}), SNR = {lmda}\n" \
           f"# Func: {d_f[f][1]}\n" \
           f"# Exec Time: {(timeit.default_timer() - ts)/60:.2e} min.\n"
  # Save Data
  MSE = np.c_[MSE_SE, MSE, MSE_L, MSE_F, MSE_B, MRG, Ons, MRG_F, c].tolist()
  with open(f"../data/SE{s}.dat", "w") as fp:
    fp.write(header)
    fp.write("# SE\tMSE\tMSE_L\tMSE_F\tMSE_B\tMRG\tOns\tMRG_F\tc\tmu\tsigma\n")
    fp.write("\n".join(["\t".join([str(b) for b in MSE[i]])
                        for i in range(t)]))

  for cB, c in zip([B, B_L, B_F, B_B], ["", "L", "F", "B"]):
    cB = np.c_[bins, cB.T].tolist()
    with open(f"../data/{c}AMP{s}.dat", "w") as fp:
      fp.write(header)
      fp.write("\n".join(["\t".join([str(b) for b in cB[i]])
                          for i in range(nbins)]))

  #if force:
    #os.system(f"rm {os.path.splitext(p.format('*'))[0]}")
  return

if __name__ == "__main__": main()
