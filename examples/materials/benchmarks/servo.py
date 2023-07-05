# BSD 3-Clause License
# Copyright (c) 2023, Retief Lubbe
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
# this
#  list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import scipy


def mixed_control(
    particles,
    material,
    time,
    dt,
    mask,
    target_strain=None,
    target_stress=None,
    callback=None,
    callback_step=1,
    is_finite_strain=False,
    cycle=1,
    tolerance=1e-6,
):
    """Perform mixed control boundary conditions

    Parameters
    ----------
    particles : ParticlesContainer
        Single material point particle
    material : Material
        Material model
    time : float
        Total time to run
    dt : float
        Time step
    mask : np.array
        (3,3) np array of bools to indicate if components are stress or strian control
        (0 = stress, 1 = strain)
    target_strain : np.array, optional
        (3,3) np array for target strain control, by default None
    target_stress : np.array, optional
        (3,3) np array for the target stress, by default None
    callback : func, optional
        Callback function between loading steps, by default None
    callback_step : int, optional
        step when output should be performed, by default 1
    is_finite_strain : bool, optional
        Flag if deformation is driven by finite strain or small strain
        , by default False
    cycle : int, optional
        Number of targets to reach, by default 1
    tolerance : float, optional
        Tolerance of the optimization algorithm, by default 1e-6

    Returns
    -------
    tuple
        particles and material
    """
    if target_strain is None:
        if is_finite_strain:
            target_strain = np.identity(3)
        else:
            target_strain = np.zeros((3, 3))

    if target_stress is None:
        target_stress = np.zeros((3, 3))

    if cycle == 0:
        if is_finite_strain:
            strain_prev = np.identity(3)
        else:
            strain_prev = np.zeros((3, 3))
    else:
        strain_prev = particles.F[0]

    stress_prev = particles.stresses[0]

    N_segments = int(time / dt)  # NOSONAR

    strain_inc = (target_strain - strain_prev) / N_segments
    stress_inc = (target_stress - stress_prev) / N_segments

    # loop through through loading steps (or time)
    for n in range(N_segments):
        target_stress_N = stress_prev + stress_inc  # NOSONAR
        target_strain_N = strain_prev + strain_inc  # NOSONAR

        # initial guess is the previous strain
        x0 = strain_prev[mask]

        # if it is a stress control, the unknown strain increment is found using
        # an optimization algorithm (Krylov) by "dry running" the loading path
        # We turn off the history update so we don't modify the yield surface while
        # probing. This step is not needed for strain control.
        if len(x0) != 0:
            res = scipy.optimize.root(
                loading_path,
                x0,
                args=(
                    mask,
                    target_stress_N,
                    target_strain_N,
                    strain_prev,
                    dt,
                    particles,
                    material,
                    False,
                    is_finite_strain,
                ),
                method="krylov",
                tol=tolerance,
            )
            root_res = res.x
        else:
            root_res = []

        # Do an actual update with the found strain increment
        particles, material = loading_path(
            root_res,
            mask,
            target_stress_N,
            target_strain_N,
            strain_prev,
            dt,
            particles,
            material,
            do_update_history=True,
            is_finite_strain=is_finite_strain,
        )

        # update the previous strain and stress
        strain_prev = particles.F[0]
        strain_prev[mask] = root_res
        stress_prev = particles.stresses[0]

        # save the results
        if (callback is not None) & (n % callback_step == 0):
            callback(particles, material, n)

    return particles, material


def loading_path(
    strain_unknown,
    mask,
    target_stress,
    target_strain,
    strain_prev,
    dt,
    particles,
    material,
    do_update_history=False,
    is_finite_strain=False,
):
    """A multivariable function used to do a trail run to find the given stress targets

    OR

    Just do a update single step if the strain increment is known.

    The main input is the strain at the current trail unknown strain.

    Note
    ----

    If `is_finite_strain=False`, then the deformation gradient array F is used for the
    total strain and the velocity gradient is used for the strain increment.

    Parameters
    ----------
    strain_unknown : list
        flattened list of unknown strain components. If empty, its full strain
        controlled. We find the strain increment using an optimization algorithm.
    mask : np.array
        (3,3) np array of bools to indicate if components are stress or strian control
    target_stress : np.array
        (3,3) np array of target strain
    target_strain : np.array
        (3,3) np array of target strain
    strain_prev : np.array
        (3,3) np array of previous strain state (last loading step)
    dt : float
        timestep
    particles : ParticlesContainer
        Single material point particle
    material : Material
        Material model
    do_update_history : bool, optional
        Flag if we should update the history (and yield surface), by default False
    is_finite_strain : bool, optional
        Flag to indicate if deformation is driven by finite strain or small strain,
        by default False

    Returns
    -------
    np.array
        stress tensor
    """
    strain_trail = target_strain

    strain_trail[mask] = strain_unknown

    # use the strain increment instead of velocty gradient
    if is_finite_strain:
        dFdt = (strain_trail - strain_prev) / dt
        velgrad = dFdt @ np.linalg.inv(strain_trail)
    else:
        velgrad = strain_trail - strain_prev

    particles.velocity_gradient = [velgrad]

    particles.F = [strain_trail]

    # update yield surface and history variables
    material.do_update_history = do_update_history

    material.is_velgrad_strain_increment = ~is_finite_strain

    particles, _ = material.stress_update(particles, 0)

    if do_update_history:
        return particles, material

    stress = np.array(particles.stresses[0])

    stress = np.where(mask, target_stress, 0) - np.where(mask, stress, 0)

    return stress[mask]
