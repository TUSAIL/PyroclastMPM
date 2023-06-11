# von Mises

This model is the (associative) von Mises constitutive model with isotropic linear strain hardening. The return mapping algorithm used in this framework is mainly presented in de Souza, et al 20211 [1]. A short overview within the formalisim of theromodynamics with internal variables are given below.

Infinitesimal strain theory is adopted in this model. The total strain is split into an elastic and plastic parts

$$\boldsymbol\varepsilon = \boldsymbol\varepsilon^e + \boldsymbol\varepsilon^p \tag{1}$$

The free energy has the form

$$\psi = \psi(\boldsymbol\varepsilon^e, \overline \varepsilon^p) \ \ , \tag{2}$$

where $\overline \varepsilon^p$ is an internal variable, accumulated plastic strain. Then we assuming an additive split of energy into elastic and plastic parts

$$\psi(\boldsymbol\varepsilon^e, \overline \varepsilon^p) = \psi^e(\boldsymbol\varepsilon^e) +\psi^p(\overline \varepsilon^p) \tag{3} \ \ $$

The elastic part of the free energy is taken as linear isotropic

$$\overline \rho   \psi^e(\boldsymbol\varepsilon^e)  = G \boldsymbol\varepsilon^e_d:\boldsymbol\varepsilon^e_d + \frac{1}{2} K (\varepsilon^e_v)^2 \ , \tag{4}$$

where $G$ and $K$ is the shear are the shear bulk modulus respectively. The plastic part of the free energy is not revealed directly. Instead, by means of the Clausis Duhem inequality

$$(\boldsymbol \sigma - \overline\rho \frac{\partial \psi^e}{\partial \boldsymbol{\varepsilon}^e}):\boldsymbol{\dot \varepsilon}^e + \boldsymbol \sigma : \boldsymbol{\dot\varepsilon}^p - \kappa \dot{\overline \varepsilon^p} \geq0 \tag{5}$$

A plastic potential $\Psi$ is chosen such that the dissipative part of Equation (5), e.g, $\Upsilon= \boldsymbol \sigma : \boldsymbol{\dot\varepsilon}^p - \kappa \dot{\overline \varepsilon^p}$ is zero-valued at origin and convex [1].

The state variables of the stress constitutive model is

$$\boldsymbol{\sigma} = \overline\rho \frac{\partial \psi}{\partial \boldsymbol{\varepsilon}^e}, \ \ \ \ \ \  \kappa = \overline\rho \frac{\partial \psi}{\partial \overline\varepsilon} \tag{6}$$

The von-Mises yield function can be presented in invariant form

$$\Phi= q - \sigma_y(\overline\varepsilon)\tag{7}$$

where the von Mises effective stress $q$ is written in terms of the second invariant of the stress deviator, $q = \sqrt{3 J_2 }$, and $\sigma_y$ is the yield limit.

In associacitive laws the yield function is taken as the plastic potential $\Psi=\Phi$. The flow rule is given by
$$\dot{\boldsymbol\varepsilon}^p = \dot\gamma \bm N \tag{8}$$
with the flow vector as
$$\boldsymbol N =\frac{\partial \Phi}{ \partial \boldsymbol\sigma} = \sqrt{\frac{3}{2}} \frac{\boldsymbol s}{|\boldsymbol s|} \tag{9}$$

where $\dot \gamma \geq0$ is the plastic multiplier.

The von Mises flow vector is parallel to the deviatoric stress (in the principal plane). This is mainly due to its pressure insensitivty.

We use linear isotropic strain hardening with a yield limit of the form

$$
\sigma_y(\overline\varepsilon) = \sigma_{y0} + \kappa(\overline\varepsilon^p) \\
 = \sigma_{y0} +  H \overline\varepsilon^p \tag{10}
$$

where $H$ a constant linear isotropic hardening modulus.

The von Mises hardening rule defined as

$$\dot{\overline\varepsilon} = -\dot\gamma \boldsymbol H \tag{11}$$

The generalized hardening modulus is given as

$$\boldsymbol H =\frac{\partial \Phi}{ \partial \kappa} = 1 \tag{12} $$
.

[1] de Souza Neto, Eduardo A., Djordje Peric, and David RJ Owen. Computational
methods for plasticity: theory and applications. John Wiley & Sons, 2011.
