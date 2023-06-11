# Associative Drucker-Prager

This implementation is based on **Computational methods for plasticity** [[1]](#1).

## Yield function

$$\Phi(\bm\sigma,c)= \sqrt{J_2(s(\bm\sigma))} + \eta \ p(\bm \sigma) + \xi \ c , \tag{6.119}$$

where $c$ is the cohesion, $\eta$ and $\xi$ are parameters related to the Mohr-coulomb criteria.

## Flow rule

Associativity assumes that the yield function is the same as the plastic potential.

$$ \Psi = \Phi $$

$$\bm{\dot \varepsilon = \dot\gamma} \bm N \tag{6.157}$$

$$\bm N=\frac{1}{2 \sqrt{J_2 (\bm s)}}\bm s + \frac{\eta}{3} \bm I \tag{6.156}$$

## Hardening rule

Accumulated plastic strain.

$$\dot{\overline \varepsilon} = -\dot\gamma \frac{\partial \Phi }{\partial \kappa} \tag{6.205}= \gamma \ \dot \xi$$

Linear hardening model

$$c(\dot{\overline \varepsilon}) = c_0  + \kappa \ (\dot{\overline \varepsilon} )$$

## References

<a id="1">[1]</a>
de Souza Neto, Eduardo A., Djordje Peric, and David RJ Owen. Computational methods for plasticity: theory and applications. John Wiley & Sons, 2011.
