#pragma once

#include <memory>

namespace gf::simulator::single_dev::mix_block
{
    class Simulator
    {
        private:
            class Data;
            std::unique_ptr<Data> _data;
        public:
            Simulator(int argc, char** argv);
            void run();
            ~Simulator();
    };
}