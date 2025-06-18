remove.packages("greta.gp")

# greta.gp minimal working example
library(raster)
library(dplyr)

# simulate data
x1 <- runif(20, 1, 11)
x2 <- runif(20, 0, 10)
x <- data.frame(x = x1, y = x2)
z <- sin(x1) + cos(x2) + rnorm(20, 0, 0.5)
plot(x1, x2, cex = y - min(y) + 1)
# x_plot <- cbind(x1 = seq(-1, 11, length.out = 200),
#                 x2 = seq(-1, 11, length.out = 200))
x_plot <- raster(nrow = 200, ncol = 200, xmn = 0, xmx = 12, ymn = -1, ymx = 11)
x_coords <- rasterToPoints(x_plot)
values(x_plot) <- sin(x_coords[,1]) + cos(x_coords[,2]) + rnorm(nrow(x_coords), 0, 0.5)

plot(x_plot)
points(x1, x2, cex = z - min(z) + 1)

library(greta)
#library(greta.gp)

# or install local version ...:
setwd("~/")
devtools::install("greta.gp.st.on.earth")
library(greta.gp)

# here are the bits I've added
?circmat
?degrees_to_radians
?great_circle_dist

# hyperparameters
# rbf_var <- lognormal(0, 1)
# rbf_len <- lognormal(0, 1)
circmat_len <- lognormal(0, 1)
circmat_var <- lognormal(0, 1)
obs_sd <- lognormal(0, 1)

# kernel & GP
kernel <- circmat(c(circmat_len, circmat_len), circmat_var) + bias(1)
f <- gp(x, kernel)

# likelihood
distribution(z) <- normal(f, obs_sd)

# fit the model by Hamiltonian Monte Carlo
m <- model(f)
draws <- mcmc(m, n_samples = 250)

# gives trace for each of the latent variables:
# bayesplot::mcmc_trace(draws)
# if my random field isn't in the `model` call, how does that change model behaviour?
# it gives us something pretty similar?

# prediction
f_plot <- project(f, x_coords)
y_plot <- greta::calculate(f_plot,
                           values = draws)

# plot summaries of posterior samples for chain 1
med_vals <- apply(y_plot[[1]], 2, median)
sd_vals <- apply(y_plot[[1]], 2, sd)

med_ras <- x_plot %>%
  setValues(med_vals)
sd_ras <- x_plot %>%
  setValues(sd_vals)

par(mfrow=c(1,2))
plot(med_ras, main="Median posterior samp")
points(x1, x2, cex = z - min(z) + 1)
plot(sd_ras, main="SD posterior samp")
points(x1, x2, cex = z - min(z) + 1)



