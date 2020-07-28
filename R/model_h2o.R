#' @import dplyr
#' @importFrom glue glue
#' @importFrom magrittr %>%
#' @importFrom knitr kable
#' @importFrom h2o h2o.getFrame h2o.auc h2o.predict h2o.cbind h2o.saveModel
#' @importFrom h2o h2o.varimp
#' @importFrom rlang sym
#' @importFrom data.table fwrite fread
NULL

#' save.h2o.cv.pred
#'
#' save.h2o.cv.pred
#'
#' @param model h2o.model
#' @param id [string] id column
#' @param label [string] label column
#' @param model.id [string] model.id
#' @param model.dir [path] path to save the model
#' @param verbose whether to show message
save.h2o.cv.pred <- function(model, model.id, model.dir, id, label, verbose = TRUE){
  df.cvpreds <- h2o.getFrame(model@model[["cross_validation_holdout_predictions_frame_id"]][["name"]]) %>%
    as.data.frame()
  df.id <- h2o.getFrame(model@allparameters$training_frame)[c(id, label)] %>%
    as.data.frame()
  cvpreds <- df.id %>%
    bind_cols(df.cvpreds) %>%
    mutate(preds = p1, label = !!sym(label)) %>%
    select(one_of(c(id, 'label', 'preds')))

  out <- glue('{model.dir}/cv_pred/{model.id}.csv')
  if(verbose) message(glue('> saving [cv preds] to [{out}]'))
  fwrite(cvpreds, out)
}

#' save.h2o.cv.score
#'
#' save cv score from h2o model
#'
#' @inheritParams save.h2o.cv.pred
save.h2o.cv.score <- function(model, model.id, model.dir, verbose = TRUE){
  res.cv <- model@model$cross_validation_metrics_summary['auc',] %>%
    as.data.frame() %>%
    select(starts_with('cv_')) %>%
    as.numeric()

  res.train <- h2o.auc(model, train = T)
  res.valid <- h2o.auc(model, valid = T)

  res <- list(train = res.train, xval = res.cv, valid = res.valid)

  out <- glue('{model.dir}/cv_score/{model.id}.json')
  if(verbose) message(glue('> saving [cv scores] to {out}'))
  list2json(res, out)
}

#' save.h2o.valid.pred
#'
#' Save valid set prediction from h2o.model
#'
#' @inheritParams save.h2o.cv.pred
#' @param X.valid.hex valid set in h2o.DataFrame
save.h2o.valid.pred <- function(model, model.id, model.dir,
                                X.valid.hex, id, label,
                                verbose = TRUE){
  preds <- h2o.predict(model, X.valid.hex)
  preds <- h2o.cbind(X.valid.hex[,c(id, label)], preds)
  preds <- preds %>%
    as.data.frame() %>%
    mutate(label = !!sym(label),
           preds = p1) %>%
    select(one_of(c(id, 'label', 'preds')))
  out <- glue('{model.dir}/valid_pred/{model.id}.csv')
  if(verbose) message(glue('> saving [valid preds] to {out}'))
  fwrite(preds, out)
}

#' h2o.save_model
#'
#' Save h2o.model and extra metrics, related results
#'
#' @inheritParams save.h2o.cv.pred
#' @inheritParams save.h2o.valid.pred
#' @export
#' @examples
#' library(h2o)
#' h2o.init(nthreads = 1,max_mem_size = '1g')
#' y <- 'am'
#' x <- setdiff(colnames(mtcars), y)
#' model.id <- 'h2o.model'
#' mtcars <- tibble::rownames_to_column(mtcars, 'id')
#' hex <- as.h2o(mtcars)
#' hex$am <- h2o.asfactor(hex$am)
#' h2o.model <- h2o.gbm(x, y , hex, model.id,
#'                      validation_frame = hex, nfolds = 2,
#'                      keep_cross_validation_predictions = TRUE,
#'                      min_rows = 1, max_depth = 3, ntrees = 10)
#' h2o.save_model(h2o.model, './saved_model', hex, id = 'id', label = 'am')
#' h2o.shutdown(FALSE)
h2o.save_model <- function(model, model.dir, X.valid.hex,
  id = 'LifeID', label = 'label',verbose = TRUE){

  model.id <- model@model_id

  # init dirs
  sub.dirs <- c('h2o_model',
                'cv_pred','cv_score','valid_pred',
                'var_imp')
  init.dirs(file.path(model.dir, sub.dirs), verbose = verbose)

  if(verbose){
    # report run time
    message(glue('> run time: {round(model@model$run_time / 100 / 60,2)} mins'))
    # report auc
    message('AUC summary:')
    print(kable(h2o.auc(model, train = T, xval = T, valid = T) %>% t(),
                format = 'markdown',
                digits = 3))
    # report variable importance
    message('Variable Importance:')
  }

  model.imp <- h2o.varimp(model)
  if(verbose){
    head(model.imp, 20) %>%
    kable(digits = 3, format = 'markdown') %>%
    print()
  }

  # save result to disk
  if(verbose) message(glue('> saving [model] to [{model.dir}/h2o_model/{model.id}]'))
  h2o.saveModel(model, glue('{model.dir}/h2o_model'), force = T)

  if(verbose) message(glue('> saving [variable importance] => [{model.dir}/var_imp/{model.id}.csv]'))
  fwrite(model.imp, file.path(model.dir,'var_imp',glue('{model.id}.csv')))

  save.h2o.cv.pred(model, model.id, model.dir, id, label, verbose)
  save.h2o.valid.pred(model, model.id, model.dir, X.valid.hex, id, label, verbose)
  save.h2o.cv.score(model, model.id, model.dir, verbose)
}
