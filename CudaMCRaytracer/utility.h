#pragma once
#include "vec3.h"
#define PI 3.14159265359f 
__device__ inline vec3 random_in_unit_sphere(curandState* state)
{
	vec3 p;
	do
	{
		p = 2.0f * vec3::random(state) - vec3(1, 1, 1);
	}
	while (p.length_squared() >= 1.0f);
	return unit_vector(p);
}

__host__ __device__ inline vec3 reflect(const vec3& v1, const vec3& n)
{
	return v1 - 2.0f * dot(v1, n) * n;
}

__device__ inline vec3 randomUnitDisk(curandState* state)
{
	vec3 p;
	do
	{
		p = 2.0f * vec3{curand_uniform(state), curand_uniform(state), 0} - vec3{1,1,0};
	} while(p.length_squared() >= 1.0f);
	return p;
}
