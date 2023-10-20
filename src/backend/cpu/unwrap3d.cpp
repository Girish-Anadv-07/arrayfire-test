/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/dispatch.hpp>
#include <common/half.hpp>
#include <kernel/unwrap3d.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <unwrap3d.hpp>

using arrayfire::common::half;

namespace arrayfire {
namespace cpu {
template<typename T>
Array<T> unwrap3d(const Array<T> &in, const dim_t wx, const dim_t wy, const dim_t wz,
                const dim_t sx, const dim_t sy, const dim_t sz, const dim_t px, const dim_t py,
                const dim_t pz, const dim_t dx, const dim_t dy, const dim_t dz, const bool is_column) {
    af::dim4 idims = in.dims();

    dim_t nx = 1 + (idims[0] + 2 * px - (((wx - 1) * dx) + 1)) / sx;
    dim_t ny = 1 + (idims[1] + 2 * py - (((wy - 1) * dy) + 1)) / sy;
    dim_t nz = 1 + (idims[1] + 2 * pz - (((wz - 1) * dz) + 1)) / sz;

    af::dim4 odims(wx * wy * wz, nx * ny * nz, 1, 1);

    if (!is_column) { std::swap(odims[0], odims[1]); }

    Array<T> outArray = createEmptyArray<T>(odims);

    const int d = (is_column) ? 1 : 0;
    getQueue().enqueue(kernel::unwrap3d_dim<T>, outArray, in, wx, wy, wz, sx, sy, sz, px,
                       py, pz, dx, dy, dz, d);

    return outArray;
}

#define INSTANTIATE(T)                                                      \
    template Array<T> unwrap3d<T>(                                            \
        const Array<T> &in, const dim_t wx, const dim_t wy, const dim_t wz, \
        const dim_t sx, const dim_t sy, const dim_t sz, const dim_t px,     \
        const dim_t py, const dim_t pz, const dim_t dx, const dim_t dy,     \
        const dim_t dz, const bool is_column);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(half)
#undef INSTANTIATE

}  // namespace cpu
}  // namespace arrayfire
