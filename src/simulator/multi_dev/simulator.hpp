#pragma once

#include <memory>
#include "thread_pool.hpp"

namespace gf::simulator::multi_dev
{
    class Simulator
    {
        private:
            class Data;
            std::unique_ptr<Data> _data;
            gf::basic::ThreadPool _pool;
        public:
            Simulator(int argc, char** argv);
            void run();
            ~Simulator();
    };
}