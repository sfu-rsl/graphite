# Design (wip)


- Separate arrays for each type of variable or observation
- Pointers to array wrappers (underlying memory may be reallocated)
- For autodiff, need separate vectors for the gradients (or 2d array for variables)



## Error

```cpp

const CameraParams params(....);

template<typename T>
vec2<T> project_pinhole(const CameraPose<T>& camera, const Point<T>& point, const CameraParams<T> & params) {...}


__global__ void reprojection_kernel(CameraPoses* poses, Points* points, Observations* obs, Vec2* error, const CameraParams params, size_t N) {


    ...

    if (i < N) {
        return;
    }

    auto c = poses[pose_idx];
    auto p = points[points_idx];
    auto o = obs[obs_idx];

    error[err_idx] = obs - project_pinhole(c, p);
    
    ...

}

reset_gradients()

auto calc_reprojection_error = [=]() {
    reprojection_kernel<<num_blocks,threads_per_block>(poses, points, obs, error, params, num_constraints);
};

```
Issues 
- Gradient computation and storage
  - Vector valued function
  - Kernel needs to know how many parameters per error value
- What if we want analytic derivatives
- Read/Write Coalescing
