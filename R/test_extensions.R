x1 = matrix(c(145, -40, 150, -38, 140, -39, 130, -30), nrow=2)
x2 = matrix(c(145, -34, 142, -36, 147, -34), nrow=2)

x1_radians = degrees_to_radians(x1)
x2_radians = degrees_to_radians(x2)

x1r = tf$expand_dims(t(x1_radians), 0L)
x2r = tf$expand_dims(t(x2_radians), 0L)

tf_great_circle_distance(tf$expand_dims(t(x1_radians), 0L),
                         tf$expand_dims(t(x2_radians), 0L), 
                        circumference = 6378137)/1000

# Why aren't the functions in tf_kernels added to the package? How do I control that?

# wrapper of tf_great_circle_distance()
great_circle_dist(tf$expand_dims(t(x1_radians), 0L),
                    tf$expand_dims(t(x2_radians), 0L))

# what is the meaning of lengthscales parameter here ? 
# Is it different to how I understand it to be later?

tf_Matern52(x1r,x2r,
            active_dims = c(1, 2),
            lengthscales = c(1, 1),
            variance = 1)

1 * (as.double(1) + as.double(sqrt5) * tmpd + as.double(5) / as.double(3) * tf$math$square(tmpd)) * tf$math$exp(-1 * sqrt5 * tmpd)
# assuming that when I'm in Python, floats are what I want ...


tf_circMatern <- function(X,
                          X_prime,
                          lengthscale,
                          variance,
                          active_dims,
                          circumference = 1L){ # actually don't want circumference of Earth here
  # active dimensions
  X <- tf_cols(X, active_dims)
  X_prime <- tf_cols(X_prime, active_dims)

  # calculate great circle distances
  r <- great_circle_dist(X, X_prime, lengthscale, circumference = circumference)

  ls_inv <- 1L / tf$cast(lengthscale, "float32")
  message(ls_inv)
  offset <- pi * ls_inv / 2L
  message(offset)
  cosh_coef <- 1L + offset / tf$math$tanh(offset)
  message(cosh_coef)
  scale_inv <- 1L / (tf$math$cosh(offset) + offset / tf$math$sinh(offset))
  message(scale_inv)
  diffs <- (as.double(r) - pi) * ls_inv / 2L
  print(diffs)
 
  scale_inv * (cosh_coef * tf$math$cosh(diffs) - diffs * tf$math$sinh(diffs))
}

tf_circMatern(x1r,x2r,
            active_dims = c(1L, 2L),
            lengthscale = 1L,
            variance = 1L)

