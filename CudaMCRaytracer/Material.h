#pragma once

class vec3;
struct hit_record;
class ray;

class material
{
public:
	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered,
	                                curandState* rand_state) const = 0;
};
