5 Hz derivation for the **Dynamixel XL-330-M288-T**.

---

### 1 Rotor inertia from the PDF

* XL330-M288 $I_{zz}=2.2712629\times10^{3}\;{\rm g·mm^{2}}$&#x20;
* Convert $1\;{\rm g·mm^{2}} = 10^{-9}\;{\rm kg·m^{2}}$

$$
J_{m}=2.2712629\times10^{3}\times10^{-9}
       =2.2713\times10^{-6}\;{\rm kg·m^{2}}
$$

---

### 2 Reflect inertia through the gearbox

Gear ratio $N=288$

$$
J_{\mathrm{ref}}
  = N^{2} J_{m}
  = (288)^{2}\times 2.2713\times10^{-6}
  \approx 0.1884\,\mathrm{kg\cdot m^{2}}
$$


---

### 3 Estimate joint-side viscous friction

Using datasheet values (5 V): $T_{\text{stall}}=0.52\;{\rm N·m}$, no-load speed $n_{\text{nl}}=103\;{\rm rpm}=10.79\;{\rm rad/s}$

$$
b_{\text{ref}}\approx\frac{T_{\text{stall}}}{\omega_{\text{nl}}}
               =\frac{0.52}{10.79}
               \approx0.048\;{\rm N·m·s/rad}
$$

---

### 4 Choose the closed-loop bandwidth

Target $f=5\;\text{Hz}$ ⇒ $\omega_{n}=2\pi f\approx31.42\;\text{rad/s}$

---

### 5 Critical-damping PD gains

$$
\begin{aligned}
k_{p} &= J_{\text{ref}}\;\omega_{n}^{2}
      =0.1884\times31.42^{2}
      \approx \boxed{1.86\times10^{2}\;{\rm N·m/rad}} \\[6pt]
k_{v,\text{total}} &= 2J_{\text{ref}}\;\omega_{n}
                   =2\times0.1884\times31.42
                   =11.84\;{\rm N·m·s/rad} \\[4pt]
k_{v} &= k_{v,\text{total}}-b_{\text{ref}}
      =11.84-0.048
      \approx \boxed{1.18\times10^{1}\;{\rm N·m·s/rad}}
\end{aligned}
$$

*(Subtracting $b_{\text{ref}}$ ensures the **total** damping (model + real friction) is critical.)*

---
