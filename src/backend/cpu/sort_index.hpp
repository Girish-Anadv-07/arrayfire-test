/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>

namespace arrayfire {
namespace cpu {
template<typename T>
void sort_index(Array<T> &okey, Array<unsigned> &oval, const Array<T> &in,
                const unsigned dim, bool isAscending);
}  // namespace cpu
}  // namespace arrayfire