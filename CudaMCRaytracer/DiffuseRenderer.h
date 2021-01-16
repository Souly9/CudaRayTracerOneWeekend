#pragma once
#include "hitable.h"
#include "ray.h"
#include "vec3.h"


class DiffuseRenderer
{
public:
	

	__device__ static vec3 color_unit_sphere(const ray& r, hitable** world,
	                                         curandState* rand_state)
	{
		auto cur_ray = r;

		vec3 cur_attentuation = vec3{1, 1, 1};

		for (int i = 0; i < 50; ++i)
		{
			hit_record rec;
			if ((*world)->hit(cur_ray, 0.01f, FLT_MAX, rec))
			{
				ray scattered;
				vec3 attentuation;
				if (rec.mat_ptr->scatter(cur_ray, rec, attentuation, scattered, rand_state))
				{
					cur_ray = scattered;
					cur_attentuation = cur_attentuation * attentuation;
				}
				else
				{
					return {0, 0, 0};
				}
			}
			else
			{
				const vec3 unit_direction = unit_vector(cur_ray.direction());
				const float t = 0.5f * (unit_direction.y() + 1.0f);
				const vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
				return cur_attentuation * c;
			}
		}
		return {0, 0, 0};
	}
};
