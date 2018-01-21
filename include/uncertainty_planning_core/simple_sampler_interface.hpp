#include <stdio.h>
#include <vector>
#include <map>
#include <random>

#ifndef SIMPLE_SAMPLER_INTERFACE_HPP
#define SIMPLE_SAMPLER_INTERFACE_HPP

namespace simple_sampler_interface
{
    template <typename Configuration, typename Generator>
    class SimpleBaseSampler
    {
    public:

        SimpleBaseSampler() {}

        virtual Configuration Sample(Generator& prng) = 0;
    };
}

#endif // SIMPLE_SAMPLER_INTERFACE_HPP
