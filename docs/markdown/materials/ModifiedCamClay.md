# MCC

## Questions

- sign convention for hardening rule, Pc

## Material parameters

| Symbol   | Name                             |
| -------- | -------------------------------- |
| $\beta$  | Outer radius of yield surface    |
| $p_t$    | tensile yield hydrostatic stress |
| $p_{c0}$ | initial preconsolidation stress  |
| $K$      | Bulk modulus (linear)            |
| $G$      | Shear Modulus                    |
| $M$      | friction coefficient             |

## Procedures

### Calculate preconsolidation pressure

Preconsolidation pressure
$$p_c (\alpha)  =   e^{\frac{v}{\lambda - \kappa}\alpha} p_{c0} \tag{A}$$

Radius of ellipse
$$a (\alpha , p_c)=\frac{p_c (\alpha) + p_t}{1+\beta} \tag{B}$$

Constant in yield surface, related to inner radius of ellipse

$$
b(p,a) =
\begin{cases}
 1 &&  p \geq p_t -a \\
 \beta && p < p_t - a
\end{cases}
\tag{C}
$$

Yield surface
$$\Phi (p,q, a, b) = \frac{1}{b^2} [ p - p_t + a]^2 + [\frac{q}{M} ]^2 - a^2 \tag{D}$$

Reduced return mapping equatiions
$$p_{n+1} (\alpha_{n+1},\alpha_{n},p^{tr}) = p^{tr} + K(\alpha_{n+1} - \alpha_{n}) \tag{E}$$

$$
\bm s_{n+1}(\Delta\gamma,q^{tr}) = \frac{M^2}{M^2 +6G\Delta\gamma}\bm s^{tr} \tag{G}
$$

$$
q_{n+1}(\Delta\gamma,q^{tr}) = \frac{M^2}{M^2 +6G\Delta\gamma}q^{tr} \tag{F}
$$

$$
\alpha_{n+1} (\alpha_n, \Delta\gamma, p_{n+1}, a_{n+1}) = \alpha_n - \Delta\gamma \frac{2}{b^2}[p_{n+1} - p_t + a_{n+1}] \tag{H}
$$

## Algorithm summary

### Trail elastic step

Trail elastic strain
$$\bm \varepsilon^{e \ tr}_{n+1} = \bm \varepsilon^{e}_n  + \Delta \bm{\varepsilon} \tag{1}$$

Trail bolumetric (elastic) strain
$$\varepsilon_{v \ n+1}^{e \ tr} = \text{trace} (\bm \varepsilon^{e \ tr}_{n+1}) \tag{2}$$

Trail pressure
$$p^{tr}_{n+1} = K \varepsilon^{e \ tr}_{v \ n+1} \tag{3}$$

Trail deviatoric (elastic) strain tensor
$$\bm{\varepsilon}_{d \ n+1}^{e \ tr}= \bm \varepsilon^{e \ tr}_{n+1}  - \frac{1}{3} \varepsilon_{v \ n+1}^{e \ tr} \bm{I} \tag{4}$$

Trail deviatoric stress tensor
$$\bm s^{tr}_{n+1} = 2G\bm \varepsilon^{e \ tr}_{d \ n+1} \tag{5}$$

Trail von Mises effective stress
$$q^{tr}_{n+1} = \sqrt{\frac{3}{2} \bm s^{tr}_{n+1}: \bm s^{tr \ T}_{n+1} } \tag{6}$$

Specific volume
(initial volume divided by solid volume)
$$v=V/V_s \tag{7}$$

Trail plastic volumetric strain (internal variable)
positive compressive volumetric strain
$$\alpha^{tr}_{n +1} = \alpha_{n} = - \varepsilon^{p}_{v \ n} $$

Trail preconsolidation pressure
(See procedure A)
$$p_{c \ n+1}^{tr} = p_c(\alpha^{tr}_{n +1}) \tag{9}$$

(See procedure B)
$$a^{tr}_{n+1}=a(p_{c \ n+1}^{tr}) \tag{10}$$

### Check if stress is within elastic domain

(See procedure C)
$$b^{tr}_{n+1} = b(p^{tr}_{n+1},a^{tr}_{n+1}) \tag{11}$$

See if stress is in elastic domain
(See procedure D)
$$\Phi^{tr} =\Phi (p^{tr}_{n+1},q^{tr}_{n+1}, a^{tr}_{n+1}, b^{tr}_{n+1}) \tag{12}$$

if $\Phi^{tr} \leq 0$ then

### Accept trail solution

$$ \bm \sigma*{n+1} = \bm s^{tr}*{n+1} + p^{tr}\_{n+1} \bm I \tag{13}$$

$$\bm \varepsilon^e_{n+1} = \bm \varepsilon^{e \ tr}_{n+1} \tag{14}$$

$$\alpha_{n +1} = \alpha^{tr}_{n +1} \tag{15}$$

### else return mapping procedure

Solve reduced system of equations to find

$(\Delta\gamma_{n+1},\alpha_{n+1})$
among all
$(\Delta\gamma^{k}_{n+1},\alpha^{k}_{n+1})$
pairs such that $$\bm R = \bm 0 \tag{16}$$

While $|\bm R| > tol$

(procedures E,F,G)

$$p^k_{n+1} =p_{n+1} (\alpha^k_{ n +1},\alpha^{tr}_{n +1},p^{tr}_{n+1}) \tag{17}$$

$$\bm s^k_{n+1} = \bm s_{n+1}(\Delta\gamma^{k}_{n+1},\bm s^{tr}_{n+1}) \tag{18}$$

$$q^k_{n+1} = q_{n+1}(\Delta\gamma^{k}_{n+1},q^{tr}_{n+1}) \tag{19}$$

Procedure A, B, C
$$p_{c \ n+1}^{k} = p_c(\alpha^k_{ n +1}) \tag{20}$$

$$a^{k}_{n+1}=a(p_{c \ n+1}^{k}) \tag{21}$$

$$b^{k}_{n+1} = b(p^{k}_{n+1},a^{k}_{n+1}) \tag{22}$$

Procedure H

(updated hardening variable)
$$\overline\alpha_{n+1} = \overline\alpha_{n+1} (\alpha^{tr}_{n +1}, \Delta\gamma^{k}_{n+1}, p^k_{n+1}, a^k_{n+1}) \tag{23}$$

do (until)

if not perform return mapping procedure by solving

$$

\begin{bmatrix}
R_1 \\R_2
\end{bmatrix}

=
\begin{bmatrix}

\Phi (p^{k}_{n+1},q^{k}_{n+1}, a^{k}_{n+1}, b^{k}_{n+1})

\\
\alpha^{k }_{n+1} - \overline\alpha_{n+1}
\end{bmatrix} =

\begin{bmatrix}
0 \\ 0
\end{bmatrix}
$$

Find new approximations for $(\Delta\gamma^{k \ *}_{n+1},\alpha^{k \ *}_{n+1})$

The residual derivative matrix

$ \bm d := \begin{bmatrix}
\frac{\partial R_1 } { \partial \Delta \gamma} && \frac{\partial R_1}{ \partial \varepsilon_v^p} \\
\frac{\partial R_2 } { \partial \Delta \gamma} && \frac{\partial R_2}{ \partial \varepsilon_v^p} \\
\end{bmatrix} \\ $

New guess for $\Delta \gamma^k_{n+1} $ and $\varepsilon^{p \ k}_{v \ n+1}$

$\begin{bmatrix}
\Delta \gamma^{k+1}_{n+1} \\
\varepsilon_{v \ n+1}^{p \ k+1}
\end{bmatrix} =
\begin{bmatrix}
\Delta \gamma^{k}_{n+1}  \\
\varepsilon_{v \ n+1}^{p \ k}
\end{bmatrix} - \bm d^{-1}
\begin{bmatrix}
R_1 \\
R_2
\end{bmatrix}
$

We have

$\bm d = \begin{bmatrix}
\frac{-12 G}{M^2+6G\Delta \gamma} (\frac{q}{M})^2 && \frac{2[p (\alpha) -p_t +a (\alpha) ] }{b^2} (K + H ) -2a(\alpha) H \\
\frac{2[p (\alpha) -p_t +a (\alpha) ] }{b^2} &&
1 + \frac{2 \Delta \gamma}{b^2}(K +H)
\end{bmatrix}$

where

$H = \frac{d a (\alpha)}{d \alpha} $

$\frac{d a (\alpha)}{d \alpha} = \frac{1}{1+\beta}\frac{d p_c (\alpha)}{d \alpha}$

$\frac{d p_c (\alpha)}{d \alpha} = \frac{v_0}{\lambda - \kappa} p_c(\alpha) $

---

Hardening rule

$\alpha = -\varepsilon_v^p$

$ln\frac{ p_c } {p_{c0}} = \frac{v_0}{\lambda - \kappa} \Delta\alpha$

so

$\frac{ p_c } {p_{c0}} = e^{\frac{v_0}{\lambda - \kappa} \Delta\alpha}$

or

$p_c  =   e^{\frac{v}{\lambda - \kappa} \Delta\alpha} p_{c0}$

$p_c (\alpha) =  (1 +\beta) a(\alpha) - p_t$

$a (\alpha)=\frac{p_c + p_t}{1+\beta}$

derivative

$\frac{d p_c }{ d \alpha } = \frac{v_0}{\lambda - \kappa} p_c$

## Out dated

Consider compression positive strain and pressure

$\dot p_c = \dot \varepsilon^p_v (\frac{v}{\lambda - \kappa }) p_{c}$

or

$p_{c \ n+1} = p_{c \ n}  + (\varepsilon^p_{v \ n+1} -\varepsilon^p_{v \ n})  (\frac{v}{\lambda - \kappa }) p_{c \ n}$

finally

$p_{c \ n+1} = p_{c \ n}[1  +\varepsilon^p_{v \ n+1}(\frac{v}{\lambda - \kappa }) -\varepsilon^p_{v \ n}(\frac{v}{\lambda - \kappa }) ]$

where $v=1+e$ is the updated specific volume

since

<!-- $\alpha = -\varepsilon^p_v $ -->

we have

$$p_{c \ n+1} = p_{c \ n}[ 1 - \alpha_{n}(\frac{v}{\lambda - \kappa }) +\alpha_{n +1}(\frac{v}{\lambda - \kappa })]$$

Consider compression positive strain and pressure

$\dot p_c = \dot \varepsilon^p_v (\frac{v}{\lambda - \kappa }) p_{c}$

or

$p_{c \ n+1} = p_{c \ n}  + (\varepsilon^p_{v \ n+1} -\varepsilon^p_{v \ n})  (\frac{v}{\lambda - \kappa }) p_{c \ n}$

finally

$p_{c \ n+1} = p_{c \ n}[1  +\varepsilon^p_{v \ n+1}(\frac{v}{\lambda - \kappa }) -\varepsilon^p_{v \ n}(\frac{v}{\lambda - \kappa }) ]$

where $v=1+e$ is the updated specific volume

since

<!-- $\alpha = -\varepsilon^p_v $ -->

we have

$$p_{c \ n+1} = p_{c \ n}[ 1 - \alpha_{n}(\frac{v}{\lambda - \kappa }) +\alpha_{n +1}(\frac{v}{\lambda - \kappa })]$$

$$\frac{\partial p_{c \ n +1}}{\partial \alpha_{n+1}} = p_{c \ n}(\frac{v}{\lambda - \kappa })$$

for $a(\alpha)$ we have

$$a (\alpha)=\frac{p_{c \ n +1} + p_t}{1+\beta}$$

$$H=\frac{d a_{n +1} }{d \alpha_{n+1}} = \frac{1}{1+\beta}\frac{d p_{c \ n+1}}{d \alpha_{n +1}}$$

$$\frac{\partial p_{c \ n +1}}{\partial \alpha_{n+1}} = p_{c \ n}(\frac{v}{\lambda - \kappa })$$

for $a(\alpha)$ we have

$$a (\alpha)=\frac{p_{c \ n +1} + p_t}{1+\beta}$$

$$H=\frac{d a_{n +1} }{d \alpha_{n+1}} = \frac{1}{1+\beta}\frac{d p_{c \ n+1}}{d \alpha_{n +1}}$$
