#pragma once
#ifndef SPEHRE_H
#define SPHERE_H

#include <utility>


#include "hitable.h"
#include "vec3.h"

class sphere : public hitable
{
public:
	__device__ sphere()
	{
	}

	__device__ sphere(vec3 cen, float r, material* m) : center{cen}, radius{r}, mat_ptr{m}
	{
	}

	__device__ bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const override;

	__device__ hit_record createHit(const float& temp, const ray& r) const
	{
		hit_record rec;
		rec.t = temp;
		rec.p = r.point_at_parameter(rec.t);
		rec.normal = (rec.p - center) / radius;
		rec.mat_ptr = mat_ptr;
		const auto outward_normal = (rec.p - center) / radius;
		rec.set_face_normal(r, outward_normal);
		return rec;
	}

public:
	vec3 center;
	float radius;
	material* mat_ptr;
};


__device__ bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - a * c;
	if (discriminant > 0)
	{
		float temp = (-b - sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min)
		{
			rec = createHit(temp, r);
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min)
		{
			rec = createHit(temp, r);
			return true;
		}
	}
	return false;
}

#endif
