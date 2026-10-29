import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
def S_aux(t, a):
    z = t / (np.sqrt(2.0) * a)
    return 1.0 - erf(z) + np.sqrt(2.0 / np.pi) * (t / a) * np.exp(-t**2 / (2.0 * a**2))
def f_aux(t, a):
    return np.sqrt(2.0 / np.pi) * (t**2 / a**3) * np.exp(-t**2 / (2.0 * a**2))
def S_MOMB(t, a, p):
    b = p / (2.0 * (p - 1.0))
    s0 = S_aux(t, a)
    return b * s0 / (1.0 - (1.0 - b) * s0)
def f_MOMB(t, a, p):
    b = p / (2.0 * (p - 1.0))
    s0 = S_aux(t, a)
    f0 = f_aux(t, a)
    return b * f0 / ((1.0 - (1.0 - b) * s0)**2)
def h_MOMB(t, a, p):
    return f_MOMB(t, a, p) / S_MOMB(t, a, p)
t = np.linspace(0.001, 6, 500)
a = -1.0
p_vals = [0.2, 0.4, 0.6]
colors = ['#1f77b4', '#d62728', '#2ca02c']
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
# Painel 1: Densidade
for p, color in zip(p_vals, colors):
    axes[0].plot(t, f_MOMB(t, a, p), label=f'p={p}', color=color, linewidth=2.2)
axes[0].set_title('MOMB density (a=-1)', fontsize=12)
axes[0].set_xlabel('t')
axes[0].set_ylabel('f_MOMB', rotation=0, labelpad=15)
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend(frameon=False)
# Painel 2: Sobrevivência
for p, color in zip(p_vals, colors):
    axes[1].plot(t, S_MOMB(t, a, p), label=f'p={p}', color=color, linewidth=2.2)
axes[1].set_title('Defective MOMB survival (a=-1)', fontsize=12)
axes[1].set_xlabel('t')
axes[1].set_ylabel('S_MOMB', rotation=0, labelpad=15)
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend(frameon=False)
# Painel 3: Risco / Hazard
for p, color in zip(p_vals, colors):
    axes[2].plot(t, h_MOMB(t, a, p), label=f'p={p}', color=color, linewidth=2.2)
axes[2].set_title('MOMB hazard function (a=-1)', fontsize=12)
axes[2].set_xlabel('t')
axes[2].set_ylabel('h_MOMB', rotation=0, labelpad=15)
axes[2].grid(True, linestyle=':', alpha=0.6)
axes[2].legend(frameon=False)
plt.tight_layout()
plt.savefig('momb_density_survival_hazard.pdf', format='pdf', bbox_inches='tight')
plt.savefig('momb_density_survival_hazard.png', dpi=300)
plt.show()