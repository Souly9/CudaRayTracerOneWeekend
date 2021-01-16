#pragma once
#include "Material.h"
#include "hitable_list.h"
#include "vec3.h"
#include "utility.h"
class lambertian : public material
{
public:
	vec3 albedo;
	__device__ lambertian(const vec3& a) : albedo{a}
	{
	}

	__device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered,
	                        curandState* rand_state) const override
	{
		vec3 target = rec.p + rec.normal + random_in_unit_sphere(rand_state);

		scattered = ray{rec.p, target - rec.p};
		attenuation = albedo;
		return true;
	}
};
