/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <err_cpu.hpp>
#include <math.hpp>

namespace arrayfire {
namespace cpu {
namespace kernel {

template<typename T>
void unwrap3d_dim(Param<T> out, CParam<T> in, const dim_t wx, const dim_t wy, const dim_t wz,
                const dim_t sx, const dim_t sy, const dim_t sz, const dim_t px, const dim_t py,
                const dim_t pz, const dim_t dx, const dim_t dy, const dim_t dz, const int d) {
    const T *inPtr = in.get();
    T *outPtr      = out.get();

    af::dim4 idims    = in.dims();
    af::dim4 odims    = out.dims();
    af::dim4 istrides = in.strides();
    af::dim4 ostrides = out.strides();

    dim_t nx = 1 + (idims[0] + 2 * px - (((wx - 1) * dx) + 1)) / sx;
    dim_t ny = 1 + (idims[0] + 2 * py - (((wy - 1) * dy) + 1)) / sy;

    for (dim_t w = 0; w < odims[3]; w++) {
        for (dim_t z = 0; z < odims[2]; z++) {
            dim_t cOut    = w * ostrides[3] + z * ostrides[2];
            dim_t cIn     = w * istrides[3] + z * istrides[2];
            const T *iptr = inPtr + cIn;
            T *optr_      = outPtr + cOut;

            for (dim_t col = 0; col < odims[d]; col++) {
                // Offset output ptr
                T *optr = optr_ + col * ostrides[d];

                // Calculate input window index
                dim_t winz = (col / (ny * nx));
                dim_t winy = (col / nx) % ny;
                dim_t winx = (col % nx);

                dim_t startx = winx * sx;
                dim_t starty = winy * sy;
                dim_t startz = winz * sz;

                dim_t spx = startx - px;
                dim_t spy = starty - py;
                dim_t spz = startz - pz;

                // Short cut condition ensuring all values within input
                // dimensions
                bool cond = (spx >= 0 && spx + (wx * dx) < idims[0] &&
                             spy >= 0 && spy + (wy * dy) < idims[1] &&
                             spz >= 0 && spz + (wz * dz) < idims[2]);

                for(dim_t z = 0; z < wz; z++) {
                    dim_t zpad = spz + z * dz;
                    for (dim_t y = 0; y < wy; y++) {
                        dim_t ypad = spy + y * dy;
                        for (dim_t x = 0; x < wx; x++) {
                            dim_t xpad = spx + x * dx;

                            dim_t oloc = (z * wy * wx + y * wx + x);
                            if (d == 0) oloc *= ostrides[1];

                            if (cond || (xpad >= 0 && xpad < idims[0] &&
                                         ypad >= 0 && ypad < idims[1] &&
                                         zpad >= 0 && zpad < idims[2])) {
                                dim_t iloc =
                                    (zpad * istrides[2] + ypad * istrides[1] + xpad * istrides[0]);
                                optr[oloc] = iptr[iloc];
                            } else {
                                optr[oloc] = scalar<T>(0.0);
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace arrayfire
