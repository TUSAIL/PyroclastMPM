#pragma once

#include <variant>

// Common
#include "pyroclastmpm/common/types_common.cuh"

// Boundary conditions
#include "pyroclastmpm/boundaryconditions/boundaryconditions.cuh"
// #include "pyroclastmpm/boundaryconditions/bodyforce/bodyforce.cuh"
// #include "pyroclastmpm/boundaryconditions/gravity/gravity.cuh"
// #include "pyroclastmpm/boundaryconditions/noslipwall/noslipwall.cuh"
// #include "pyroclastmpm/boundaryconditions/periodicwall/periodicwall.cuh"
// #include "pyroclastmpm/boundaryconditions/rigidparticles/rigidparticles.cuh"
// #include "pyroclastmpm/boundaryconditions/slipwall/slipwall.cuh"
// #include "pyroclastmpm/boundaryconditions/planardomain/planardomain.cuh"
// #include "pyroclastmpm/boundaryconditions/nodedomain/nodedomain.cuh"

// Materials
#include "pyroclastmpm/materials/materials.cuh"
// #include "pyroclastmpm/materials/druckerprager/druckerpragermat.cuh"
#include "pyroclastmpm/materials/linearelastic.cuh"
// #include "pyroclastmpm/materials/localrheo/localrheomat.cuh"

// #include "pyroclastmpm/materials/newtonfluid/newtonfluidmat.cuh"


// Particles, Nodes and shapefunctions
#include "pyroclastmpm/nodes/nodes.cuh"
#include "pyroclastmpm/particles/particles.cuh"
#include "pyroclastmpm/shapefunction/shapefunction.cuh"

namespace pyroclastmpm
{

    /**
     * @brief Define the material type as a variant of all the possible materials
     *
     */
    using MaterialType = std::variant<Material,
                                      LinearElastic
                                    //   NewtonFluid,
                                    //   LocalGranularRheology,
                                    //   DruckerPrager
                                    >;

    /**
     * @brief Define the boundary condition type as a variant of all the possible
     * boundary conditions
     *
     */
    using BoundaryConditionType = std::variant<BoundaryCondition
    //                                            Gravity,
    //                                            RigidParticles,
    //                                            BodyForce,
    //                                            PlanarDomain,
    //                                            NodeDomain
                                               >;

    /**
     * @brief MPM solver base class
     *
     */
    class Solver
    {
    public:
        /**
         * @brief Construct a new Solver object
         *
         * @param _particles particles container
         * @param _nodes nodes container
         * @param _boundaryconditions a list of boundary conditions to be applied
         * @param _materials a list of materials to be applied
         */
        explicit Solver(
            ParticlesContainer _particles,
            NodesContainer _nodes,
            cpu_array<MaterialType> _materials =
                cpu_array<MaterialType>(),
            cpu_array<BoundaryConditionType> _boundaryconditions =
                cpu_array<BoundaryConditionType>());

        /**
         * @brief Destroy the Solver object
         *
         */
        ~Solver();

        /**
         * @brief Solve the main loop for n_steps
         *
         * @param n_steps
         */
        void solve_nsteps(int n_steps);

        /**
         * @brief Output the results (particles,nodes,boundaryconditions, etc. )
         *
         */
        void output();

        /**
         * @brief Do stress update for all particles (using constitutive law)
         *
         */
        void stress_update();

        /**
         * @brief reset (temporary) arrays to initial state
         *
         */
        virtual void reset(){};

        /**
         * @brief main loop of the solver
         *
         */
        virtual void solve(){};

        /*!
         * @brief Smart pointer to nodes container
         */
        NodesContainer nodes;

        /*!
         * @brief Smart pointer to particles container
         */
        ParticlesContainer particles;

        /*! @brief list of materials */
        cpu_array<MaterialType> materials;

        /*!
         * @brief a list of pointers to the boundary conditions
         */
        cpu_array<BoundaryConditionType> boundaryconditions;

        /** current step of the main loop */
        int current_step;
        
    };

} // namespace pyroclastmpm