#pragma once
#include "projection_jacobians.cuh"
namespace graphite {

template <typename D, typename M, typename T>
__device__ static void bal_reprojection_error(const D *camera, const D *point,
                                              const M *obs, D *error) {
  // Compute the reprojection error according to the camera model

  /*
      https://grail.cs.washington.edu/projects/bal/
      Camera Model
      We use a pinhole camera model; the parameters we estimate for each camera
     area rotation R, a translation t, a focal length f and two radial
     distortion parameters k1 and k2. The formula for projecting a 3D point X
     into a camera R,t,f,k1,k2 is:

      P  =  R * X + t       (conversion from world to camera coordinates)
      p  = -P / P.z         (perspective division)
      p' =  f * r(p) * p    (conversion to pixel coordinates)

      where P.z is the third (z) coordinate of P.
      In the last equation, r(p) is a function that computes a scaling factor to
     undo the radial distortion:

      r(p) = 1.0 + k1 * ||p||^2 + k2 * ||p||^4.

      This gives a projection in pixels, where the origin of the image is the
     center of the image, the positive x-axis points right, and the positive
     y-axis points up (in addition, in the camera coordinate system, the
     positive z-axis points backwards, so the camera is looking down the
     negative z-axis, as in OpenGL).

  */

  Eigen::Map<const Eigen::Matrix<D, 3, 1>> X(point);
  Eigen::Map<const Eigen::Matrix<D, 9, 1>> cam(camera);
  Eigen::Map<const Eigen::Matrix<T, 2, 1>> observation(obs->data());
  Eigen::Matrix<D, 3, 1> P;

  // Rodrigues formula for rotation vector (angle-axis) with Taylor expansion
  // for small theta Adapted from g2o bal example
  Eigen::Matrix<D, 3, 1> rvec = cam.template head<3>();
  D theta = rvec.norm();

  if (theta > D(0)) {
    Eigen::Matrix<D, 3, 1> axis = rvec / theta;
    D cth = cos(theta);
    D sth = sin(theta);

    Eigen::Matrix<D, 3, 1> axx = axis.cross(X);
    D adx = axis.dot(X);

    P = X * cth + axx * sth + axis * adx * (D(1) - cth);

  } else {
    auto rxx = rvec.cross(X);
    P = X + rxx; // + D(0.5) * rvec.cross(rxx);
  }

  // Translate
  Eigen::Matrix<D, 3, 1> t = cam.template segment<3>(3);
  P += t;

  // Perspective division
  Eigen::Matrix<D, 2, 1> p = -P.template head<2>() / P(2);

  // Radial distortion
  D f = cam(6);
  D k1 = cam(7);
  D k2 = cam(8);

  D r2 = p.squaredNorm();
  D radial_distortion = D(1.0) + k1 * r2 + k2 * r2 * r2;

  // Project to pixel coordinates and compute reprojection error
  Eigen::Matrix<D, 2, 1> reprojection_error =
      f * radial_distortion * p - observation.template cast<D>();

  // Assign the error
  error[0] = reprojection_error(0);
  error[1] = reprojection_error(1);
}

template <typename D, typename M, typename T>
__device__ static void bal_reprojection_error_simple(const D *camera,
                                                     const D *point,
                                                     const M *obs, D *error) {

  Eigen::Map<const Eigen::Matrix<D, 3, 1>> X(point);
  Eigen::Map<const Eigen::Matrix<D, 9, 1>> cam(camera);
  Eigen::Map<const Eigen::Matrix<T, 2, 1>> observation(obs->data());

  // Extract rotation vector and build rotation matrix using Eigen's AngleAxis
  Eigen::Matrix<D, 3, 1> rvec = cam.template head<3>();
  D theta = rvec.norm();
  Eigen::Matrix<D, 3, 3> R = Eigen::Matrix<D, 3, 3>::Identity();

  if (theta > D(0)) {
    Eigen::AngleAxis<D> angle_axis(theta, rvec / theta);
    R = angle_axis.toRotationMatrix();
  }

  // Apply rotation and translation
  Eigen::Matrix<D, 3, 1> t = cam.template segment<3>(3);
  Eigen::Matrix<D, 3, 1> P = R * X + t;

  // Perspective division
  Eigen::Matrix<D, 2, 1> p = -P.template head<2>() / P(2);

  // Radial distortion
  D f = cam(6);
  D k1 = cam(7);
  D k2 = cam(8);
  D r2 = p.squaredNorm();
  D radial_distortion = D(1.0) + k1 * r2 + k2 * r2 * r2;
  // Project to pixel coordinates and compute reprojection error
  Eigen::Matrix<D, 2, 1> reprojection_error =
      f * radial_distortion * p - observation.template cast<D>();
  // Assign the error
  error[0] = reprojection_error(0);
  error[1] = reprojection_error(1);
}

template <typename T, typename D, size_t I>
__device__ static void bal_jacobian_simple(const T *camera, const T *point,
                                           const Eigen::Matrix<T, 2, 1> *obs,
                                           D *jacobian,
                                           const unsigned char *data) {
  Eigen::Map<const Eigen::Matrix<D, 3, 1>> X(point);
  Eigen::Map<const Eigen::Matrix<D, 9, 1>> cam(camera);

  Eigen::Matrix<T, 2, 9> Jcam;
  Eigen::Matrix<T, 2, 3> Jpoint;

  const Eigen::Matrix<T, 3, 1> rvec = cam.template head<3>();
  const Eigen::Matrix<T, 3, 1> t = cam.template segment<3>(3);
  const T f = cam(6);
  const T k1 = cam(7);
  const T k2 = cam(8);

  projection_simple<T>(rvec, t, f, k1, k2, X, Jcam, Jpoint);

  if constexpr (I == 0) {
    Eigen::Map<Eigen::Matrix<D, 2, 9>> Jcam_map(jacobian);
    Jcam_map = Jcam;
  } else if constexpr (I == 1) {
    Eigen::Map<Eigen::Matrix<D, 2, 3>> Jpoint_map(jacobian);
    Jpoint_map = Jpoint;
  }
}

} // namespace graphite