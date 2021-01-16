#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"
#include "utility.h"

#define degrees_to_radians(degrees) degrees * (PI / 180.0f)

class camera
{
public:
	__device__ camera(const vec3& pos, const vec3& target, const vec3& up, 
		const float& fov, const float& aspect_r, float aperture, float focus_dist)
		: m_origin {pos}, m_lens_radius{aperture / 2.0f}
	
	{
		const auto theta = degrees_to_radians(fov);
		const auto h = tan(theta/2.0f);
		const auto viewport_height = 2.0f * h;
		const auto viewport_width = aspect_r * viewport_height;

		m_w = unit_vector(pos - target);
		m_u = unit_vector(cross(up, m_w));
		m_v = cross(m_w, m_u);
		
		m_horizontal = focus_dist * viewport_width * m_u;
		m_vertical = focus_dist * viewport_height * m_v;
		m_lower_left_corner = m_origin - m_horizontal/2.0f - m_vertical/2.0f - focus_dist * m_w;
	}

	__device__ ray get_ray(float s, float t, curandState* state) const
	{
		const auto rd = m_lens_radius * randomUnitDisk(state);
		const auto offset = m_u * rd.x() + m_v * rd.y();
		
		return {
			m_origin + offset,
			m_lower_left_corner + s * m_horizontal + t * m_vertical - m_origin - offset
		};
	}

	vec3 m_origin;
	vec3 m_lower_left_corner;
	vec3 m_horizontal;
	vec3 m_vertical;
	vec3 m_u, m_v, m_w;
	float m_lens_radius;
};

#endif
