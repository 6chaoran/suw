#' shap value mtcars
#'
#' shap value matrix of a h2o.GBM model, which takse mpg as label and
#' the other variables as predictors using mtcars dataset
#' the code to generate the data is as following:
#' data(mtcars)
#' library(h2o)
#' library(dplyr)
#'
#' h2o.init()
#' hex <- as.h2o(mtcars)
#' model <- h2o.gbm(setdiff(colnames(mtcars), 'mpg'), mpg',min_rows = 3,
#' training_frame = hex, nfolds = 3)
#' shap_values_mtcars <- h2o.predict_contributions(model, hex) %>% as.data.frame()

"shap_values_mtcars"
