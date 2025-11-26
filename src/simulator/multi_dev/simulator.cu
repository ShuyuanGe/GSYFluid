#include <format>
#include <vector>
#include "vec3.hpp"
#include <ostream>
#include "CLI11.hpp"
#include "config.hpp"
#include "logger.hpp"
#include "simulator.hpp"
#include <cuda_runtime.h>
#include "velocity_set.hpp"
#include "cu_exception.cuh"


namespace gf::simulator::multi_dev
{
    class Simulator::Data
    {
        friend class Simulator;

        struct SingleDevData
        {
            flag_t *flagBuf = nullptr;
            real_t *rhoBuf  = nullptr;
            real_t *vxBuf   = nullptr;
            real_t *vyBuf   = nullptr;
            real_t *vzBuf   = nullptr;
            ddf_t  *ddfBuf0 = nullptr;
            ddf_t  *ddfBuf1 = nullptr;
        };

        private:
            std::uint32_t _step = 0;
            float _invTau = 0;
            std::uint32_t _nStep;
            gf::basic::Vec3<std::uint32_t> _devDim;
            gf::basic::Vec3<std::uint32_t> _blockDim;
            gf::basic::Vec3<std::uint32_t> _gridDim;
            std::vector<SingleDevData>     _singleDevData;
        public:
            Data(int argc, char** argv)
            {
                CLI::App app {"Multiple Device Fluid Solver"};
                app.add_option("--invTau", _invTau, "Reciprocal of Tau.")
                    ->default_val(0.5);
                app.add_option("--nstep", _nStep, "Total time steps of the simulation.")
                    ->default_val(100);
                app.add_option("--devDim", _devDim.data, "Dimension of GPUs involved in the simulation.")
                    ->default_val(std::array<std::uint32_t,3>{1,1,1});
                app.add_option("--blkDim", _blockDim.data, "Dimension of blocks per grid.")
                    ->default_val(std::array<std::uint32_t,3>{32, 8, 4});
                app.add_option("--gridDim", _gridDim.data, "Dimension of grids per kernel.")
                    ->default_val(std::array<std::uint32_t,3>{16, 32, 64});
                try
                {
                    app.parse(argc, argv);
                }
                catch(const CLI::ParseError& e)
                {
                    std::exit(app.exit(e));
                }
                _singleDevData.resize(getDevNum());
            }

            std::uint32_t getDevNum() const
            {
                return _devDim.x * _devDim.y * _devDim.z;
            }

            gf::basic::Vec3<std::uint32_t> getDomainDim() const
            {
                return
                {
                    _blockDim.x * _gridDim.x, 
                    _blockDim.y * _gridDim.y,
                    _blockDim.z * _gridDim.z
                };
            }

            dim3 getBlockDim() const
            {
                return { _blockDim.x, _blockDim.y, _blockDim.z };
            }

            dim3 getGridDim() const
            {
                return { _gridDim.x, _gridDim.y, _gridDim.z };
            }

            std::ostream& log(std::ostream& os) const
            {
                auto domainDim = getDomainDim();
                os << std::format(
                    "InvTau: {}\nTotal Time Step: {}\nDevice Dimension: [{},{},{}]\nSingle Device Domain Dimension: [{},{},{}]\nAll Device Domain Dimension: [{},{},{}]\n",
                    _invTau, _nStep,
                    _devDim.x, _devDim.y, _devDim.z, 
                    domainDim.x, domainDim.y, domainDim.z, 
                    _devDim.x * domainDim.x, _devDim.y * domainDim.y, _devDim.z * domainDim.z
                ) << std::endl;
                return os;
            }
    };

    Simulator::Simulator(int argc, char** argv) : 
        _data(std::make_unique<Data>(argc, argv)), 
        _pool(_data->getDevNum())
    {
        _data->log(std::cout);

        auto initDevData = [this](std::uint32_t devId, gf::basic::Logger& logger, std::barrier<>& barrier)
        {
            CU_CHECK(cudaSetDevice(devId));
            const gf::basic::Vec3<std::uint32_t> devDim = _data->_devDim;
            const std::int32_t devIdxX = devId%devDim.x;
            const std::int32_t devIdxY = (devId/devDim.x)%devDim.y;
            const std::int32_t devIdxZ = devId/(devDim.x * devDim.y);
            for(std::uint32_t dir=0 ; dir<NDIR ; ++dir)
            {
                if(dir == gf::basic::VelSet3D::getCentIdx()) continue;
                const std::int32_t nbrIdxX = devIdxX + gf::basic::VelSet3D::getDxArr()[dir];
                const std::int32_t nbrIdxY = devIdxY + gf::basic::VelSet3D::getDyArr()[dir];
                const std::int32_t nbrIdxZ = devIdxZ + gf::basic::VelSet3D::getDzArr()[dir];
                if(
                    0<=nbrIdxX and nbrIdxX<devDim.x and 
                    0<=nbrIdxY and nbrIdxY<devDim.y and 
                    0<=nbrIdxZ and nbrIdxZ<devDim.z
                )
                {
                    const std::int32_t nbrIdx = nbrIdxX+devDim.x*(nbrIdxY+devDim.y*nbrIdxZ);
                    CU_CHECK(cudaDeviceEnablePeerAccess(nbrIdx, 0));
                }
            }
            LOG_INFO(logger, "Initialize cuda environment successfully!");
        };

        _pool.addTask(initDevData);
    }

    void Simulator::run()
    {

    }

    Simulator::~Simulator()
    {

    }
}