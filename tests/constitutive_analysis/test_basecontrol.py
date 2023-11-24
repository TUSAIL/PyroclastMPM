
#%%
from constitutive_analysis import basecontrol
import pyroclastmpm.MPM3D as pm
import numpy as np

def create_basecontrol():

    material = pm.Material() # placeholder class

    particles = pm.ParticlesContainer([[0.0, 0.0, 0.0]])

    basecontrol_cls = basecontrol.BaseControl(particles,material,verbose=False,tolerance=10,output_step=1000)
    return basecontrol_cls

# %%
def test_basecontrol():
    """
    Test if basecontrol class can be initialized
    """
    _ = create_basecontrol()
    assert True

def test_sign_flip():
    """
    Test if sign is flipped / tension compression
    """
    
    basecontrol_cls = create_basecontrol()

    basecontrol_cls.set_sign_flip(-1.0)

    assert basecontrol_cls.sign_flip ==-1.0, "sign flip not correct"


def test_store_results():

    basecontrol_cls = create_basecontrol()

    stress_tensor = np.eye(3)
    F_tensor = np.eye(3)*-100

    step = 2

    basecontrol_cls.particles.stresses = [stress_tensor]

    basecontrol_cls.particles.F = [F_tensor]

    basecontrol_cls.store_results(step)

    np.testing.assert_array_equal(np.array([stress_tensor]),basecontrol_cls.stress_list,"basecontrol store results for stress not equal to truth value" )

    # note for finite strain F is used as a placeholder
    np.testing.assert_array_equal(np.array([F_tensor]),basecontrol_cls.strain_list,"basecontrol store results for stress not equal to truth value" )

    assert basecontrol_cls.step_list == [step]

def test_post_processes():

    basecontrol_cls = create_basecontrol()

    pressure = 100

    dev_stress_tensor = np.ones((3,3))*20 - np.eye(3)*20

    stress_tensor = -np.eye(3)*pressure + dev_stress_tensor

    q_vm = np.sqrt(3 * 0.5* np.trace(dev_stress_tensor @ dev_stress_tensor.T))

    tau = 0.5*np.trace(dev_stress_tensor @ dev_stress_tensor.T)


    
    volumetric_strain = 1.2

    dev_strain_tensor = np.ones((3,3))*0.2 - np.eye(3)*0.2

    strain_tensor = -np.eye(3)*volumetric_strain*(1./3) + dev_strain_tensor


    gamma = 0.5*np.trace(dev_strain_tensor @ dev_strain_tensor.T)

    basecontrol_cls.stress_list = [stress_tensor, stress_tensor]

    basecontrol_cls.strain_list = [strain_tensor, strain_tensor]

    basecontrol_cls.post_process()

    # test pressure
    np.testing.assert_array_equal(basecontrol_cls.pressure_list,np.array([pressure,pressure]),"basecontroller pressure not calculated correctly")
    
    # test deviatoric stress tensor
    np.testing.assert_array_equal(basecontrol_cls.dev_stress_list,np.array([dev_stress_tensor,dev_stress_tensor]),"basecontroller deviatoric stress tensor not calculated correctly")
    
    # test q_vm
    np.testing.assert_array_equal(basecontrol_cls.q_vm_list,np.array([q_vm,q_vm]),"basecontroller deviatoric vom misses effective stress tensor not calculated correctly")
    
    # test tau
    np.testing.assert_array_equal(basecontrol_cls.tau_list,np.array([tau,tau]),"basecontroller deviatoric shear stress tensor tensor not calculated correctly")

    # test volumetric strain
    np.testing.assert_array_equal(basecontrol_cls.volumetric_strain_list,np.array([volumetric_strain,volumetric_strain]),"basecontroller volumetric strain not calculated correctly")

    # test deviatoric strain tensor
    np.testing.assert_array_equal(basecontrol_cls.dev_strain_list,np.array([dev_strain_tensor,dev_strain_tensor]),"basecontroller deviatoric strain tensor not calculated correctly")

    # test gamma
    np.testing.assert_array_equal(basecontrol_cls.gamma_list,np.array([gamma,gamma]),"basecontroller deviatoric shear strain tensor tensor not calculated correctly")


test_basecontrol()

test_sign_flip()

test_store_results()

test_post_processes()


# %%
