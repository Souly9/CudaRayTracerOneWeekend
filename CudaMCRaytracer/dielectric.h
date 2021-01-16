#pragma once
#include <curand_kernel.h>
#include "hitable_list.h"

#include "Material.h"
#include "utility.h"

class dielectric : public material
{
	private:
	__device__ static float schlick(float cos, float ref_idx)
	{
		auto r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
		r0 = r0 * r0;
		return r0 + (1.0f - r0) * powf((1.0f - cos), 5.0f);
	}

	__device__ inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) const
	{
	const auto cos_theta = fmin(dot(-uv, n), 1.0f);
	
	const auto r_out_perp = etai_over_etat * (uv + cos_theta * n);
	
	const auto r_out_parallel = -sqrt(fabs(1.0f - r_out_perp.length_squared())) * n;
	
	return r_out_perp + r_out_parallel;
}
public:
	float m_ir;

	__device__ dielectric(float ir) : m_ir{ir}
	{
	}

	__device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered,
	                        curandState* rand_state) const override
	{
		attenuation = vec3{1, 1, 1};
		const auto ref_ratio = rec.front_face ? (1.0f / m_ir) : m_ir;

		const vec3 unit_dir = unit_vector(r_in.direction());

		const auto cos_theta = fmin(dot(-unit_dir, rec.normal), 1.0f);
		const auto sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

		const bool cannot_refract = (ref_ratio * sin_theta > 1.0f) > 0;
		const bool ref = cannot_refract || 
			(schlick(cos_theta, ref_ratio) > curand_uniform(rand_state));
		const auto refracted = (ref * reflect(unit_dir, rec.normal))
		+ (!ref * refract(unit_dir, rec.normal, ref_ratio));

		scattered = {rec.p, refracted};
		return true;
	}


};
