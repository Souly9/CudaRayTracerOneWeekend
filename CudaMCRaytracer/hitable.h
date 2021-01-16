#pragma once
#ifndef HITTABLE_H
#define HITTABLE_H

#include "Material.h"
#include "ray.h"

struct hit_record
{
	vec3 p;
	float t;
	vec3 normal;
	material* mat_ptr;
	bool front_face;

	 __device__ inline void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal :-outward_normal;
    }
};

class hitable
{
public:
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif