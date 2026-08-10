#include "sdl/transform/scale_transform.hpp"

namespace sdl {

Sample ScaleTransform::apply(Sample sample) const {
    for (auto& v : sample.data) {
        v *= factor_;
    }
    return sample;
}

} // namespace sdl
