#pragma once

#include <cstdint>

namespace gf::simulator::single_dev_expt
{
    using idx_t = std::int32_t;
    using flag_t = std::uint32_t;
    using real_t = float;
    using ddf_t = float;
    constexpr flag_t LOAD_DDF_BIT    = static_cast<flag_t>(1) << 0;
    constexpr flag_t EQU_DDF_BIT     = static_cast<flag_t>(1) << 1;
    constexpr flag_t COLLIDE_BIT     = static_cast<flag_t>(1) << 2;
    constexpr flag_t BOUNCE_BACK_BIT = static_cast<flag_t>(1) << 3;
    constexpr flag_t STORE_DDF_BIT   = static_cast<flag_t>(1) << 4;

    constexpr flag_t CORRECT_BIT     = static_cast<flag_t>(1) << 31;
}