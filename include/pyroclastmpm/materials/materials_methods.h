namespace pyroclastmpm {

__host__ __device__ inline Real calc_pressure_sigma(const Matrix3r stress) {
  return -(Real)(1.0 / 3.0) * stress.trace();
}

__host__ __device__ inline Real calc_pressure_K(const Real bulk_modulus,
                                                const Real eps_e_v) {
  return bulk_modulus * eps_e_v;
}

__host__ __device__ inline Real
calc_pressure_nonlinK(const Real specific_volume_original, const Real kappa,
                      const Real pressure_prev, const Real eps_e_v,
                      const Real eps_e_v_prev) {

  const Real deps_e_v = eps_e_v - eps_e_v_prev;
  const Real pressure =
      pressure_prev * exp((-specific_volume_original / kappa) * deps_e_v);

  return pressure;
}

__host__ __device__ inline Matrix3r calc_devstress_sigmap(const Matrix3r stress,
                                                          const Real pressure) {
  return stress + pressure * Matrix3r::Identity();
}

__host__ __device__ inline Real calc_volstrain_eps(const Matrix3r strain) {
  return -strain.trace();
}

__host__ __device__ Real calc_bulk_modulus_p_eps_v(const Real pressure,
                                                   const Real eps_v) {
  return pressure / eps_v;
}
__host__ __device__ Real calc_shear_modulus_K_pois(const Real bulk_modulus,
                                                   const Real pois) {
  return (3.0 * (1.0 - 2.0 * pois)) / (2.0 * (1.0 + pois)) * bulk_modulus;
}

} // namespace pyroclastmpm