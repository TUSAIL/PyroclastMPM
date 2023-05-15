#include "pyroclastmpm/materials/druckerprager/druckerpragermat_kernels.cuh"

namespace pyroclastmpm {

extern __constant__ Real dt_gpu;

__global__ void KERNEL_TRAIL_FE_DRUCKERPRAGER(
    // const Real * particles_Fe_tr_flat_gpu,
    Matrixr* particles_F_gpu,
    const Matrixr* particles_velocity_gradients_gpu,
    const uint8_t* particles_colors_gpu,
    const int num_particles,
    const int mat_id)
    {
        const int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= num_particles) {
            return;
        }  // block access threads

        const int particle_color = particles_colors_gpu[tid];

        if (particle_color != mat_id) {
            return;
        }
        const Matrixr vel_grad = particles_velocity_gradients_gpu[tid];
        const Matrixr Fe = particles_F_gpu[tid];
        const Matrixr Fe_tr = (Matrixr::Identity() + dt_gpu * vel_grad) *Fe;
        
        // TODO replace this with flattened array when using CUSOLVER to compute SVD
        particles_F_gpu[tid] = Fe_tr;

    }


__global__ void KERNEL_STRESS_UPDATE_DRUCKERPRAGER(
    Matrix3r * particles_stresses_gpu,
    Matrixr* particles_F_gpu,
    Real * particles_logJp_gpu,
    Real * particles_pressure_gpu,
    const Matrixr * U_gpu,
    const Matrixr * V_gpu,
    const Vectorr *S_gpu,
    const uint8_t* particles_colors_gpu,
    const Real alpha,
    const Real shear_modulus,
    const Real lame_modulus,
    const Real cohesion,
    const Real vcs,
    const int num_particles,
    const int mat_id)
    {

        const int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= num_particles) {
            return;
        }  // block access threads

        const int particle_color = particles_colors_gpu[tid];

        if (particle_color != mat_id) {
            return;
        }

        Vectorr S = S_gpu[tid];
        Matrixr U = U_gpu[tid];
        Matrixr V = V_gpu[tid];

        Real logJp = particles_logJp_gpu[tid];

        Vectorr eps;
        eps(0) = log(max(abs(S[0]), 1.e-4));

        #if DIM >1
        eps(1) = log(max(abs(S(1)), 1.e-4));
        #endif

        #if DIM >2
        eps(2) = log(max(abs(S(2)), 1.e-4));
        #endif

        Real sum_eps = eps.sum() + logJp;

        Vectorr eps_hat = eps - Vectorr::Ones() * (sum_eps / DIM);


        Real eps_norm = eps.norm();
        Real eps_hat_norm = eps_hat.norm();

        Real dlogJp=0.;
        Vectorr S_new;

        if (sum_eps >= 0.) {
            // Project to start of cone. Tension


            S_new(0) = exp(cohesion);
            #if DIM >1
            S_new(1) = exp(cohesion);
            #endif

            #if DIM >2
            S_new(2) = exp(cohesion);
            #endif

            dlogJp = vcs*sum_eps;
            
        } else {
            Real delta_gamma = eps_hat_norm + alpha * sum_eps *
                           (DIM * lame_modulus + 2. * shear_modulus) /
                           (2 * shear_modulus);

            if (delta_gamma <= 0) {
            // Elastic deformation gradient already on the yield surface
            Vectorr H = eps + Vectorr::Ones()*cohesion;

            S_new(0) = exp(H(0));
            #if DIM >1
            S_new(1) = exp(H(1));
            #endif
            #if DIM >2
            S_new(2) = exp(H(2));
            #endif
            } else {

                Vectorr H = eps - delta_gamma * (eps_hat / eps_hat_norm) + Vectorr::Ones()*cohesion;

                // printf("H = %.16f %.16f %.16f \n", H(0, 0), H(1, 1), H(2, 2));
     
                S_new(0) = exp(H(0));
     
                #if DIM >1
                S_new(1) = exp(H(1));
                #endif
                #if DIM >2
                S_new(2) = exp(H(2));
                #endif
            }
        }

        logJp+=dlogJp;

        
        Matrixr S_diag = Matrixr::Zero();
        S_diag(0,0) = S_new(0);
        #if DIM >1
        S_diag(1,1) = S_new(1);
        #endif
        #if DIM >2
        S_diag(2,2) = S_new(2);
        #endif

        // if (tid==0)
        // {
        //     // printf("S_new = %.16f %.16f %.16f \n", S_new(0), S_new(1), S_new(2));
        //     printf("S_diag = %.16f %.16f %.16f \n", S_diag(0, 0), S_diag(1, 1), S_diag(2, 2));
        // }
        
        const Matrixr V_T = V.transpose();

        // Reproject to find new Fe and Fp
        particles_F_gpu[tid] = U * S_diag * V_T;

        const Matrixr S_inverse = S_diag.inverse();
        
        Matrixr Se_log = Matrixr::Zero();
        Se_log(0,0) = log(S_diag(0,0));
        Se_log(1,1) = log(S_diag(1,1));
        Se_log(2,2) = log(S_diag(2,2));

        Matrixr Se_update = 2.0*shear_modulus*S_inverse*Se_log + lame_modulus*Se_log.trace()*S_inverse;

        Matrixr T = (U*Se_update*V_T)/particles_F_gpu[tid].determinant();

        // if (tid==0)
        // {
        //     printf("T = %.16f %.16f %.16f \n", T(0, 0), T(1, 1), T(2, 2));
        // }
        // printf("T = %.16f %.16f %.16f \n", T(0, 0), T(1, 1), T(2, 2));


        #if DIM ==3
        particles_stresses_gpu[tid] = T; // Cauchy stress
        #else
        particles_stresses_gpu[tid].block(0, 0, DIM, DIM) = T.block(0, 0, DIM, DIM); // Cauchy stress
        #endif

        particles_pressure_gpu[tid] =  -(1. /(Real)DIM) * T.trace();

    }

}

