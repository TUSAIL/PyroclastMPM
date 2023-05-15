#include "pyroclastmpm/materials/nonlocalngf/nonlocalngfmat_kernels.cuh"

#include <eigen3/Eigen/Eigenvalues>  // header file

namespace pyroclastmpm {

extern __constant__ Real dt_gpu;

/**
 * @brief Stress update for nonlocal rheology (NGF). See
 * https://www.sciencedirect.com/science/article/pii/S0045782522001876
 *
 * @param particles_stresses_gpu
 * @param particles_FP_gpu
 * @param particles_ddg_gpu
 * @param particles_g_gpu
 * @param particles_pressure_gpu
 * @param particles_mu_gpu
 * @param particles_velocity_gradients_gpu
 * @param particles_F_gpu
 * @param num_particles
 * @return __global__
 */
__global__ void KERNEL__STRESS_UPDATE_NONLOCALRHEO(
    Matrix3r* particles_stresses_gpu,
    Matrix3r* particles_FP_gpu,
    Real* particles_ddg_gpu,
    Real* particles_g_gpu,
    Real* particles_pressure_gpu,
    Real* particles_mu_gpu,
    const Matrix3r* particles_velocity_gradients_gpu,
    const Matrix3r* particles_F_gpu,
    const Real shear_modulus,
    const Real bulk_modulus,
    const Real mu_s,
    const Real mu_2,
    const Real original_density,
    const int num_particles

) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;

  // if (tid >= num_particles) {
  //   return;
  // }  // block access threads

  // const Matrix3r F = particles_F_gpu[tid];  // Fn+1

  // const Matrix3r FP = particles_FP_gpu[tid];  // FPn

  // const Matrix3r Fe_tr = F * FP.inverse();  // Fe_tr = Fn+1*FPn^-1

  // // Finding Ue_tr, we can get the strech part of the polar decomposition
  // Matrix3r Q, R;
  // do_qr_decomposition(Fe_tr, Q, R);
  // const Matrix3r Ue_Tr = R * Q.transpose();

  // // Compute the logarithmic mapping of matrix S
  // Eigen::SelfAdjointEigenSolver<Matrix3r> eigensolver(Ue_Tr);

  // // Compute Hencky elastic strain
  // Matrix3r E_tr = Matrix3r::Zero();  // Logarithmic mapping of S
  // logS(0, 0) = log(eig_vals(0));
  // logS(1, 1) = log(eig_vals(1));
  // logS(2, 2) = log(eig_vals(2));

  // const Matrix3r E_tr_dev =
  //     0.5 *
  //     (E_tr -
  //      E_tr.trace() *
  //          Matrix3r::Identity());  // deviatoric part of Hencky elastic strain

  // const Matrix3r M_tr = 2 * shear_modulus * E_tr_dev +
  //                       bulk_modulus * E_tr.trace() *
  //                           Matrix3r::Identity();  // Hencky elastic stress

  // const Real p_tr = -M_tr.trace() / 3.;  // Hencky elastic pressure

  // if ((p_tr = < 0) | (1 / F.determinant() < original_density)) {
  //   particles_stresses_gpu[tid] = Matrix3r::Zero();
  //   particles_pressure_gpu[tid] = 0.0;
  //   particles_FP_gpu[tid] = F;
  //   return;
  // }

  // if (articles_pressure_gpu[tid] == 0) {
  //   const Matrix3r D = 0.5 * (F.transpose() * F + Matrix3r::Identity());  //

  //   particles_g_gpu[tid] = sqrt(2) * sqrt(D * D.trace()) / mu_2;
  //   particles_mu_gpu[tid] = mu_2;
  // } else {


    
  // }

  // deviatoric part of deformation gradient
}


}  // namespace pyroclastmpm
