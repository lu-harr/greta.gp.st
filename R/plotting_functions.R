plot.kernel <- function(kernel, 
                        dist_max){
  # plot change in correlation over distance
  # to write this I'll need to understand how greta kernels work !
  # might be fun to implement a 2D / pairs plot version
  # working under the assumption that kernel is isomorphic
  
  dists <- seq(0, dist_max, length.out=100)
  corrns <- calculate(kernel(0, dists))
  plot(dists, corrns, ylim = c(0, 1), type = "l", xlab="Distance", ylab="Correlation")
  
  # next I've got that I want "quantiles of realised correlation from the data"
  # what am I asking for here?
  
}

plot.effects <- function(draws,
                         data,
                         column){
  # plot response over covar
  # is there an existing plot that does this ?
}