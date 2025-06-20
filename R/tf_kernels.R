# tensorflow implementations of common kernels


tf_cols <- function(X, active_dims) {
  X[, , active_dims, drop = FALSE]
}


tf_distance <- function(x1, x2, squared = FALSE) {
  n1 <- dim(x1)[[2]]
  n2 <- dim(x2)[[2]]

  x1 <- tf$tile(tf$expand_dims(x1, 3L),
                list(1L, 1L, 1L, n2))
  x2 <- tf$transpose(x2, perm = c(0L, 2L, 1L))
  x2 <- tf$tile(tf$expand_dims(x2, 1L), list(1L, n1, 1L, 1L))

  dists <- (x1 - x2)^2
  dist <- tf$reduce_sum(dists, axis = 2L)

  if (!squared) {
    dist <- tf$math$sqrt(dist)
  }

  dist
}


# calculate great circle distance - 
# may run into trouble for small distances (floating point precision)
# note geosphere::distHaversine implements Vicenty formula
# TODO check x1 and x2 are in radians before proceeding
# (ideally this is done outside of any greta interaction)
tf_great_circle_distance <- function(x1, x2, 
                                     active_dims = c(0L, 1L),
                                     circumference = 1L, 
                                     radians = TRUE){
  lon <- active_dims[1]
  lat <- active_dims[2]
  
  n1 <- dim(x1)[[2]]
  n2 <- dim(x2)[[2]]
  
  # message(tf$shape(x1))
  # message(tf$shape(x2))

  lam1 <- x1[, , lon]
  phi1 <- x1[, , lat]
  lam2 <- x2[, , lon]
  phi2  <- x2[, , lat]
  
  # message("lam phi")
  # message(tf$shape(phi1))
  # 
  phi1 <- tf$tile(tf$expand_dims(phi1, axis = 2L), list(1L, 1L, n2))
  phi2 <- tf$tile(tf$expand_dims(phi2, axis = 1L), list(1L, n1, 1L))
  
  lam1 <- tf$tile(tf$expand_dims(lam1, axis = 2L), list(1L, 1L, n2))
  lam2 <- tf$tile(tf$expand_dims(lam2, axis = 1L), list(1L, n1, 1L))
  
  bits <- tf$multiply(tf$math$sin(phi1), tf$math$sin(phi2)) + 
    tf$multiply(tf$multiply(tf$math$cos(phi1), tf$math$cos(phi2)), 
                tf$math$cos(lam1 - lam2))
  
  # dist <- tf$math$acos(
  #   tf$multiply(tf$math$sin(phi1), tf$math$sin(phi2)) + 
  #     tf$multiply(tf$multiply(tf$math$cos(phi1), tf$math$cos(phi2)), 
  #                 tf$math$cos(lam1 - lam2))
  # )
  
  # dist <- tryCatch(expr = {tf$math$acos(bits)},
  #                  warning = function(w){
  #                    message("Warn Lucy!: ", w)
  #                    tf$math$acos(tf$clip(bits, -1, 1))
  #                 },
  #                 error = function(e){
  #                   message("Error Lucy!: ", e)
  #                   tf$math$acos(tf$clip(bits, -1, 1))
  #                 })
  
  dist <- tf$math$acos(tf$clip_by_value(bits, -1, 1))
  
  # message("offending shape?")
  # message(tf$shape(tf$tensordot(tf$math$sin(phi1), tf$math$sin(phi2), axes = 0L)))
  # message("hopefully transpose")
  # message(tf$shape(tf$transpose(lam1) - lam2))
  # message("or this one?")
  # message(tf$shape(tf$expand_dims(tf$math$cos(tf$transpose(lam1) - lam2), axis = 0L)))
  # 
  # dist = tf$math$acos(
  #           tf$reshape(
  #             tf$tensordot(tf$math$sin(phi1), tf$transpose(tf$math$sin(phi2)), axes = 0L),
  #             c(1L, n1, n2)) +
  #           tf$reshape(
  #             tf$tensordot(tf$math$cos(phi1), tf$transpose(tf$math$cos(phi2)), axes = 0L),
  #             c(1L, n1, n2)) *
  #           tf$expand_dims(tf$math$cos(tf$transpose(lam1) - lam2), axis = 0L)
  # )

  # message("dist ok")
  dist * circumference
}


#' Convert degrees to radians
#'
#' @param degrees, numeric: number of decimal degrees to convert to radians
#'
#' @return `degrees` in radians (numeric)
#' @export
degrees_to_radians <- function(degrees){
  degrees * pi / 180L
}

# build a matrix with dimension given by the number of rows in X and the
# number of rows in X_prime, filled with the given *constant* value
tf_empty_along <- function(X,
                           X_prime = NULL,
                           fill = 1) {
  if (is.null(X_prime)) {
    dims_out <- tf$stack(c(tf$shape(X)[0], dim(X)[[2]]))
  } else {
    dims_out <- tf$stack(c(tf$shape(X)[0], dim(X)[[2]], dim(X_prime)[[2]]))
  }

  switch(as.character(fill),
    `1` = tf$ones(dims_out, dtype = tf_float()),
    `0` = tf$zeros(dims_out, dtype = tf_float())
  )
}


# bias (or constant) kernel
# k(x, y) = \sigma^2
tf_bias <- function(X,
                    X_prime,
                    variance,
                    active_dims) {
  # create and return covariance matrix
  tf_empty_along(X, X_prime, 1) * variance
}

# white kernel
# diagonal with specified variance if self-kernel, all 0s otherwise
tf_white <- function(X,
                     X_prime,
                     variance,
                     active_dims) {
  # only non-zero for self-covariance matrices
  if (identical(X, X_prime)) {
    variance <- tf$squeeze(variance, 2L)
    d <- tf_empty_along(X, X_prime = NULL, fill = 1) * variance
    d <- tf$linalg$diag(d)
  } else {
    d <- tf_empty_along(X, X_prime, 0)
  }

  # return constructed covariance matrix
  d
}

tf_iid <- function(X,
                   X_prime,
                   variance,
                   active_dims) {
  # pull out active dimensions
  X <- tf_cols(X, active_dims)
  X_prime <- tf_cols(X_prime, active_dims)

  # find where these values match and assign the variance as a covariance there
  # (else set it to 0)
  distance <- tf_distance(X, X_prime)
  tf_as_float(distance < fl(1e-12)) * variance
}

# squared exponential kernel (RBF)
tf_rbf <- function(X,
                   X_prime,
                   lengthscales,
                   variance,
                   active_dims) {
  # pull out active dimensions
  X <- tf_cols(X, active_dims)
  X_prime <- tf_cols(X_prime, active_dims)

  # calculate squared distances
  r2 <- squared_dist(X, X_prime, lengthscales)

  # construct and return RBF kernel
  variance * tf$math$exp(-r2 / fl(2))
}

# rational_quadratic kernel
tf_rational_quadratic <- function(X,
                                  X_prime,
                                  lengthscales,
                                  variance,
                                  alpha,
                                  active_dims) {
  # pull out active dimensions
  X <- tf_cols(X, active_dims)
  X_prime <- tf_cols(X_prime, active_dims)

  # calculate squared distances (scaled if needed)
  r2 <- squared_dist(X, X_prime, lengthscales)

  # construct and return rational quadratic kernel
  variance * (fl(1) + r2 / (fl(2) * alpha))^-alpha
}

# linear kernel (base class)
tf_linear <- function(X,
                      X_prime,
                      variances,
                      active_dims) {
  # pull out active dimensions
  X <- tf_cols(X, active_dims)
  X_prime <- tf_cols(X_prime, active_dims)

  # full kernel
  tf$linalg$matmul(
    tf$math$multiply(variances, X),
    X_prime,
    transpose_b = TRUE
  )
}

tf_polynomial <- function(X,
                          X_prime,
                          variances,
                          offset,
                          degree,
                          active_dims) {
  # pull out active dimensions
  X <- tf_cols(X, active_dims)
  X_prime <- tf_cols(X_prime, active_dims)

  # full kernel
  tf$math$pow(
    tf$linalg$matmul(
      tf$math$multiply(variances, X),
      X_prime,
      transpose_b = TRUE
    ) + offset,
    degree
  )
}

# exponential kernel (stationary class)
tf_exponential <- function(X,
                           X_prime,
                           lengthscales,
                           variance,
                           active_dims) {
  # pull out active dimensions
  X <- tf_cols(X, active_dims)
  X_prime <- tf_cols(X_prime, active_dims)

  # calculate squared distances (scaled if needed)
  r <- absolute_dist(X, X_prime, lengthscales)

  # construct and return exponential kernel
  variance * tf$math$exp(-fl(0.5) * r)
}

# Matern12 kernel (stationary class)
tf_Matern12 <- function(X,
                        X_prime,
                        lengthscales,
                        variance,
                        active_dims) {
  # pull out active dimensions
  X <- tf_cols(X, active_dims)
  X_prime <- tf_cols(X_prime, active_dims)

  # calculate squared distances (scaled if needed)
  r <- absolute_dist(X, X_prime, lengthscales)

  # construct and return Matern12 kernel
  variance * tf$math$exp(-r)
}

# Matern32 kernel (stationary class)
tf_Matern32 <- function(X,
                        X_prime,
                        lengthscales,
                        variance,
                        active_dims) {
  # pull out active dimensions
  X <- tf_cols(X, active_dims)
  X_prime <- tf_cols(X_prime, active_dims)

  # calculate squared distances (scaled if needed)
  r <- absolute_dist(X, X_prime, lengthscales)

  # precalculate root3
  sqrt3 <- fl(sqrt(3))

  # construct and return Matern32 kernel
  variance * (fl(1) + sqrt3 * r) * tf$math$exp(-sqrt3 * r)
}

# Matern52 kernel (stationary class)
tf_Matern52 <- function(X,
                        X_prime,
                        lengthscales,
                        variance,
                        active_dims) {
  # pull out active dimensions
  X <- tf_cols(X, active_dims)
  X_prime <- tf_cols(X_prime, active_dims)

  # calculate squared distances (scaled if needed)
  r <- absolute_dist(X, X_prime, lengthscales)

  # precalculate root5
  sqrt5 <- fl(sqrt(5))

  # construct and return Matern52 kernel
  variance * (fl(1) + sqrt5 * r + fl(5) / fl(3) * tf$math$square(r)) * tf$math$exp(-sqrt5 * r)
}

# Circular Matern kernel (stationary class)
# time should be implemented separately as tf_exponential() with fixed variance
# need to incorporate variance here
tf_circMatern <- function(X,
                          X_prime,
                          lengthscale,
                          variance,
                          active_dims, # removing c(1L, 2L) here - suspect another python-R shemozzle
                          circumference = 1L, # don't want circumference of Earth here - leaving option
                          radians = TRUE){ 
  # message(lengthscale)
  # message(variance)
  
  # active dimensions
  X <- tf_cols(X, active_dims)
  X_prime <- tf_cols(X_prime, active_dims)
  
  #message("in tf_cm")
  #message(active_dims, tf$shape(X), tf$shape(X_prime))

  # message("Warning: coordinates should be in radians")
  # need to do some testing on what exactly happens if we violate this 
  # ... include a test to display warning conditionally ...
  if (radians == FALSE){
    message("Caution: this feature needs testing")
    X <- degrees_to_radians(X)
    X_prime <- degrees_to_radians(X_prime)
  }

  # calculate great circle distances
  r <- great_circle_dist(X, X_prime, active_dims, circumference)
  #message(paste("Here", tf$shape(r)))

  # some of the types in here are a little confused ...
  ls_inv <- 1L / tf$cast(lengthscale, "float64")
  offset <- pi * ls_inv / 2L
  cosh_coef <- 1L + offset / tf$math$tanh(offset)
  scale_inv <- 1L / (tf$math$cosh(offset) + offset / tf$math$sinh(offset))
  # have removed fl()s from r and ls_inv
  diffs <- (r - fl(pi)) * ls_inv / 2L
  # message("diffs")
  # message(tf$shape(diffs))
  ret <- tryCatch({scale_inv * (cosh_coef * tf$math$cosh(diffs) - diffs * tf$math$sinh(diffs))},
                  warning = function(w){
                    message("warn!") 
                    NA
                  }, error = function(e){
                    message("errorr!")
                    NA
                  })
  # message("ret")
  # message(ret)
  ret
}


# cosine kernel (stationary class)
tf_cosine <- function(X,
                      X_prime,
                      lengthscales,
                      variance,
                      active_dims) {
  # pull out active dimensions
  X <- tf_cols(X, active_dims)
  X_prime <- tf_cols(X_prime, active_dims)

  # calculate squared distances (scaled if needed)
  r <- absolute_dist(X, X_prime, lengthscales)

  # construct and return cosine kernel
  variance * tf$math$cos(r)
}

# periodic kernel
tf_periodic <- function(X,
                        X_prime,
                        lengthscale,
                        variance,
                        period) {
  # calculate squared distances (scaled if needed)
  exp_arg <- fl(pi) * absolute_dist(X, X_prime) / period
  exp_arg <- tf$math$sin(exp_arg) / lengthscale

  # construct and return periodic kernel
  variance * tf$math$exp(-fl(0.5) * tf$math$square(exp_arg))
}


tf_Prod <- function(kernel_a, kernel_b) {
  tf$math$multiply(kernel_a, kernel_b)
}

tf_Add <- function(kernel_a, kernel_b) {
  tf$math$add(kernel_a, kernel_b)
}


# rescale, calculate, and return distance
get_dist <- function(X,
                     X_prime,
                     lengthscales = NULL,
                     squared = FALSE) {
  if (!is.null(lengthscales)) {
    X <- X / lengthscales
    X_prime <- X_prime / lengthscales
  }
  
  # let's not:
  # if (great_circle){
  #   message(paste0("Calling GC", tf$shape(X)))
  #   return(tf_great_circle_distance(X, X_prime, circumference))
  # } else {
  
  tf_distance(X, X_prime, squared = squared)
}

squared_dist <- function(X,
                         X_prime,
                         lengthscales = NULL) {
  get_dist(X, X_prime, lengthscales, squared = TRUE)
}


#' Calculate absolute distance
#'
#' @param X 2-column matrix of longitudes and latitudes
#' @param X_prime 2-column matrix of longitudes and latitude
#' @param lengthscales numeric: to be supplied to `get_dist()`
#'
#' @return tensor of distances between all points in `X` and all points in `X_prime`
#' @seealso [great_circle_dist()]
#' @export
#'
absolute_dist <- function(X,
                          X_prime,
                          lengthscales = NULL) {
  get_dist(X, X_prime, lengthscales, squared = FALSE)
}

#' Calculate great circle distance
#'
#' @param X 2-column matrix of longitudes and latitudes
#' @param X_prime 2-column matrix of longitudes and latitude
#' @param lengthscales numeric: to be supplied to `get_dist()`
#' @param circumference numeric: sphere circumference in user-specified units, e.g. Earth: 6378137 metres
#'
#' @return tensor of distances between all points in `X` and all points in `X_prime`
#' 
#' @details function expects `X` and `X_prime` to be expressed in radians.
#' 
#' @seealso [degrees_to_radians()], [circmat()], [absolute_dist()]
#' 
#' @export 
#'
great_circle_dist <- function(X,
                              X_prime,
                              active_dims,
                              circumference = 1L # sphere circumference (user-specified units)
                              ) {
  # get_dist(X, X_prime, lengthscales,
  #          great_circle = TRUE,
  #          circumference = circumference)
  # skipping the scaling step in get_dist() (have removed lengthscale from args)
  tf_great_circle_distance(X, X_prime, active_dims, circumference)
}

# combine as module for export via internals
# tf_kernels_module <- module(tf_static,
#                             tf_constant,
#                             tf_bias,
#                             tf_squared_exponential,
#                             tf_rational_quadratic,
#                             tf_linear,
#                             tf_polynomial,
#                             tf_exponential,
#                             tf_Matern12,
#                             tf_Matern32,
#                             tf_Matern52,
#                             tf_cosine)




