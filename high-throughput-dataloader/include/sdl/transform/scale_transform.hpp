#pragma once

#include "sdl/sample.hpp"
#include "sdl/transform/transform.hpp"

namespace sdl {

// Multiplies every element of Sample::data by a fixed factor. Stateless
// and deterministic; exists primarily to exercise the Transform
// composition mechanism.
class ScaleTransform : public Transform {
public:
    explicit ScaleTransform(float factor) : factor_(factor) {}

    Sample apply(Sample sample) const override;

private:
    float factor_;
};

} // namespace sdl
