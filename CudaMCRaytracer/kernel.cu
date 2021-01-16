#include <iostream>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <memory>


#include "camera.h"
#include "dielectric.h"
#include "DiffuseRenderer.h"
#include "hitable.h"
#include "hitable_list.h"
#include "lambertian.h"
#include "metal.h"
#include "vec3.h"
#include "ray.h"
#include "sphere.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
#define SPHERE_NUM 4
static auto R = cos(PI/4.0f);

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

__device__ bool hit_sphere(const vec3& center, float radius, const ray& r)
{
	vec3 oc = r.origin() - center;
	//mit
	const float a = r.direction().length_squared();
	const float b = dot(oc, r.direction());
	const float c = oc.length_squared() - radius * radius;

	const float rslt = b * b - a * c;

	if (rslt < 0)
		return -1.0f;
	//equation to solve quadratic formula
	return (-b - sqrt(rslt) / a);
}


__global__ void render(vec3* fb, int max_x, int max_y, int ns,
                       camera** cam,
                       hitable** world,
                       curandState* rand_state)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y))
		return;
	const int pixel_index = j * max_x + i;
	curandState local_rand_state = rand_state[pixel_index];
	//fb[pixel_index] = vec3( float(i) / max_x, float(j) / max_y, 0.2f);

	vec3 col(0, 0, 0);
	for (int s = 0; s < ns; s++)
	{
		float u = static_cast<float>(i + curand_uniform(&local_rand_state)) / static_cast<float>(max_x);
		float v = static_cast<float>(j + curand_uniform(&local_rand_state)) / static_cast<float>(max_y);
		const ray r = (*cam)->get_ray(u, v, rand_state);
		col += DiffuseRenderer::color_unit_sphere(r, world, &local_rand_state);
	}
	rand_state[pixel_index] = local_rand_state;
	col /= static_cast<float>(ns);
	col[0] = sqrt(col[0]);
	col[1] = sqrt(col[1]);
	col[2] = sqrt(col[2]);
	fb[pixel_index] = col;
}

__global__ void create_world(hitable** d_list, hitable** d_world, camera** d_camera)
{
	
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		*(d_list) = new sphere(vec3(0, 0, -1), 0.5, new lambertian{vec3{0.8, 0.3, 0.3}});
		*(d_list + 1) = new sphere(vec3(0, -100.5, -1), 100, new lambertian{vec3{0.8, 0.8, 0.0}});
		*(d_list + 2) = new sphere(vec3(1, 0, -1), 0.5f, new metal{vec3{0.8, 0.6, 0.2}, 0.0f});
		*(d_list + 3) = new sphere(vec3(-1, 0, -1), 0.5f, new dielectric{1.5f});
		*d_world = new hitable_list(d_list, SPHERE_NUM);
		const auto camPos = vec3{0, 0, 5};
		const auto camTarget = vec3{0,0,-1};
		*d_camera = new camera(camPos, camTarget, {0,1,0},20.0f, 16.0f / 9.0f,
		0.25f, (camPos - camTarget).length());
	}
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_cam)
{
	for (int i = 0; i < SPHERE_NUM; ++i)
	{
		delete ((sphere*)d_list + i)->mat_ptr;
		delete *(d_list + i);
	}
	delete *d_world;
	delete *d_cam;
}

int main()
{
	int nx = 1280;
	int ny = 720;
	int ns = 200;
	int tx = 8;
	int ty = 8;

	const int num_pixels = nx * ny;
	const size_t fb_size = num_pixels * sizeof(vec3) * 3;

	// allocate FB
	vec3* fb;
	checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

	curandState* d_rand_state;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

	hitable** d_list;
	checkCudaErrors(cudaMalloc((void**)&d_list, SPHERE_NUM*sizeof(hitable *)));
	hitable** d_world;
	checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable *)));
	camera** d_cam;
	checkCudaErrors(cudaMalloc((void**)&d_cam, sizeof(camera * )));

	create_world<<<1,1>>>(d_list, d_world, d_cam);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Render our buffer
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);
	render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	render<<<blocks,threads>>>(fb, nx, ny, ns, d_cam, d_world, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Output FB as Image
	std::cout << "P3\n" << nx << " " << ny << "\n255\n";
	for (int j = ny - 1; j >= 0; j--)
	{
		for (int i = 0; i < nx; i++)
		{
			const size_t pixel_index = j * nx + i;
			const int ir = static_cast<int>(255.99 * fb[pixel_index].x());
			const int ig = static_cast<int>(255.99 * fb[pixel_index].y());
			const int ib = static_cast<int>(255.99 * fb[pixel_index].z());
			std::cout << ir << " " << ig << " " << ib << "\n";
		}
	}
	// clean up
	checkCudaErrors(cudaDeviceSynchronize());
	free_world<<<1,1>>>(d_list, d_world, d_cam);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaFree(d_cam));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(fb));

	// useful for cuda-memcheck --leak-check full
	cudaDeviceReset();
}
