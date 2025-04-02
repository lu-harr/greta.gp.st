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

great_circle_dist(tf$expand_dims(t(x1_radians), 0L),
                    tf$expand_dims(t(x2_radians), 0L), 
                    circumference = 6378137)

# what is the meaning of lengthscales parameter here ? Is it different to how I understand it to be later?

tf_Matern52(x1r,
            x2r,
            active_dims = c(0, 1),
            lengthscales = c(1, 1),
            variance = 1)
