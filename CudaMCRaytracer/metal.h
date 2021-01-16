#pragma once
#include <curand_kernel.h>

#include "Material.h"
#include "vec3.h"
#include "ray.h"
#include "hitable.h"
#include "utility.h"

class metal : public material
{
public:
	__device__ metal(const vec3& a, const float& f) : albedo{a}, fuzz{f < 1 ? f : 1}
	{
	}

	__device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered,
	                        curandState* rand_state) const override
	{
		const vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
		scattered = ray{rec.p, reflected + fuzz * random_in_unit_sphere(rand_state)};
		attenuation = albedo;
		return (dot(scattered.direction(), rec.normal) > 0.0f);
	}

	vec3 albedo;
	float fuzz;
};
