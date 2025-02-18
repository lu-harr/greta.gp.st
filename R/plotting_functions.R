#' Title
#'
#' @param kernel greta kernel.
#' @param dist_max numeric. greatest distance to include on x-axis.
#' @param hparams list. where hyperparameters are provided with priors, `hparams` should be a named list of exact values for hyperparameters to take.
#' @param add bool. should this be laid over an existing plot?
#' @param col character. colour for line.
#'
#' @return TRUE
#' @export
#'
#' @examples 
#' \dontrun{
#' # here's a kernel
#' rbf_var <- lognormal(0, 1)
#' rbf_len <- lognormal(0, 1)
#' kernel <- rbf(rbf_len, rbf_var) + bias(1)
#' plot.kernel(kernel, dist_max = 11, hparams = list(rbf_var = 1, rbf_len = 1))
#' plot.kernel(kernel, dist_max = 11, hparams = list(rbf_var = 0.5, rbf_len = 1), add=TRUE, col="blue")
#' plot.kernel(kernel, dist_max = 11, hparams = list(rbf_var = 1, rbf_len = 2), add=TRUE, col="red")
#' }

plot.kernel <- function(kernel, 
                        dist_max,
                        hparams = list(),
                        add = FALSE,
                        col = "black"){
  # plot change in correlation over distance
  # to write this I'll need to understand how greta kernels work !
  # might be fun to implement a 2D / pairs plot version
  # working under the assumption that kernel is isomorphic
  
  dists <- seq(0, dist_max, length.out=100)
  if (length(hparams) == 0){
    # no priors on hyperparameters
    corrns <- calculate(kernel(0, dists))
  } else {
    # require specific values for hyperparameters to be specified
    corrns <- calculate(kernel(0, dists), values = hparams)
    # how does this work for multiple dimensions? e.g. different lengthscales for lat and lon?
  }
  
  if (add == TRUE){
    lines(dists, corrns[[1]][1,], col = col)
  } else {
    plot(dists, corrns[[1]][1,], type = "l", xlab="Distance", ylab="Correlation")
  }
  
  # next I've got that I want "quantiles of realised correlation from the data"
  # what am I asking for here?
  
  # also don't understand why we're decaying to one, rather than zero. Because I don't really know what I'm doing.
}


plot.effects <- function(draws,
                         data,
                         column){
  # plot response over covar
  # is there an existing plot that does this ?
}



