#' @importFrom lightgbm lgb.cv lgb.train lgb.prepare_rules lgb.Dataset lgb.save
#' @import dplyr
#' @importFrom stats predict sd
#' @importFrom Metrics auc rmse
#' @importFrom glue glue
#' @importFrom magrittr %>%
#' @importFrom knitr kable
#' @importFrom rlang sym
#' @importFrom tibble rownames_to_column
#' @importFrom data.table fwrite fread
NULL

#' lgb.extract_cv_score
#'
#' extract cv prediction from lgb.cv object
#'
#' @param cv \code{lgb.cv} object
#' @param score.fun function for scoring, score.fun(label, preds)
#' @return list of (cv.score, cv.preds)
#' @examples
#' library(lightgbm)
#' data(agaricus.train, package = "lightgbm")
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label = train$label)
#' params <- list(objective = "regression", metric = "l2")
#' model <- lgb.cv(
#'   params = params
#'   , data = dtrain
#'   , nrounds = 10L
#'   , nfold = 3L
#'   , min_data = 1L
#'   , learning_rate = 1.0
#'   , early_stopping_rounds = 5L
#' )
#' res <- lgb.extract_cv_score(model, score.fun = Metrics::rmse)
#' res$cv.score
#' @export
lgb.extract_cv_score <- function(cv, score.fun = Metrics::auc){

  # extract cv predictions, folds, label
  rows <- length(cv$boosters[[1]]$booster$.__enclos_env__$private$valid_sets[[1]]$.__enclos_env__$private$used_indices)+
    length(cv$boosters[[1]]$booster$.__enclos_env__$private$train_set$.__enclos_env__$private$used_indices)
  preds <- numeric(rows)
  folds <- numeric(rows)
  label <- numeric(rows)
  train.score <- NULL
  valid.score <- NULL
  for(i in 1:length(cv$boosters)){
    preds[
      cv$boosters[[i]]$booster$.__enclos_env__$private$valid_sets[[1]]$.__enclos_env__$private$used_indices] <-
      cv$boosters[[i]]$booster$.__enclos_env__$private$inner_predict(2)
    folds[
      cv$boosters[[i]]$booster$.__enclos_env__$private$valid_sets[[1]]$.__enclos_env__$private$used_indices] <-
      i
    label[
      cv$boosters[[i]]$booster$.__enclos_env__$private$valid_sets[[1]]$.__enclos_env__$private$used_indices] <-
      cv$boosters[[i]]$booster$.__enclos_env__$private$valid_sets[[1]]$.__enclos_env__$private$info$label
    preds.train <- cv$boosters[[i]]$booster$.__enclos_env__$private$inner_predict(1)
    label.train <- cv$boosters[[i]]$booster$.__enclos_env__$private$train_set$.__enclos_env__$private$info$label
    train.score <- c(train.score, score.fun(label.train, preds.train))

    preds.valid <- cv$boosters[[i]]$booster$.__enclos_env__$private$inner_predict(2)
    label.valid <- cv$boosters[[i]]$booster$.__enclos_env__$private$valid_sets[[1]]$.__enclos_env__$private$info$label
    valid.score <- c(valid.score, score.fun(label.valid, preds.valid))
  }

  return(list(cv.score = list(train = train.score, xval = valid.score),
              cv.preds = data.frame(folds = folds,
                                    label = label,
                                    preds = preds,
                                    stringsAsFactors = F)))
}


#' lgb.train_with_cv
#'
#' Train LightGBM with cross-validation
#'
#' @export
#' @param df.train data.frame for train set
#' @param fnames all feature names
#' @param label label name
#' @param fnames.cat categorical feature names
#' @param id colnames of id, if not provide, default to use rowname
#' @param rules rules to index categorical features
#' @param params list of params for lightgbm
#' @param df.valid data.frame for valid set
#' @param cv.verbose whether verbose when cv
#' @param train.verbose whether verbose when train
#' @param cv whether perform a k-fold cross validation, default at FALSE
#' @param nfold number of folds of cross validation, if cv is TRUE
#' @param score.fun score functino for validation, defaults to auc
#' @param ... other parameters in \code{lgb.train}
#' @return list of (model, rules, fnames)
#' @examples
#' label <- 'label'
#' fnames <- c('Sex','Class','Age','Freq')
#' fnames.cat <- c('Sex','Age','Class')
#' df <- data.frame(Titanic)
#' df$label <- ifelse(df$Survived == "Yes", 1 ,0)
#' in.train <- runif(nrow(df)) < 0.8
#' df.train <- df[in.train, ]
#' df.valid <- df[!in.train, ]
#' bst <- lgb.train_with_cv(df.train, df.valid,
#'                        fnames = fnames,
#'                        label = label,
#'                        fnames.cat = fnames.cat,
#'                        rules = NULL, cv = TRUE,
#'                        num_leaves = 63,
#'                        learning_rate = 1.0,
#'                        nrounds = 10L,
#'                        min_data = 1L,
#'                        cv.verbose = 1,
#'                        train.verbose = 1,
#'                        eval = 'auc',
#'                        eval_freq = 10,
#'                        nfold = 2L,
#'                        early_stopping_rounds = 5L,
#'                        objective = "binary")
#' lgb.save_model(bst, './saved_model','lgb_baseline', verbose = TRUE)
#' bst.loaded <- lgb.load_model('./saved_model','lgb_baseline')
#' preds <- lgb.predict(bst.loaded, df.valid)

lgb.train_with_cv <- function(df.train, df.valid, fnames, label, fnames.cat, id = NULL,
                            rules = NULL, params = NULL,
                            cv.verbose = 0, train.verbose = 1,
                            cv = FALSE, nfold = 5, score.fun = Metrics::auc,
                            ...){
  t0 <- Sys.time()

  if('data.table' %in% class(df.train)){
    message('convert [df.train] from data.table to data.frame')
    df.train <- as.data.frame(df.train)
  }

  if('data.table' %in% class(df.valid)){
    message('convert [df.valid] from data.table to data.frame')
    df.valid <- as.data.frame(df.valid)
  }

  # clean colnames (make naming legel)
  message(glue('> rename columns for train set...'))
  colnames(df.train) <- clean.fnames(colnames(df.train))
  fnames <- clean.fnames(fnames)
  fnames.cat <- clean.fnames(fnames.cat)

  # prepare training data
  message(glue('> preparing train set ...'))
  df.train <- df.train %>% mutate_at(vars(one_of(fnames.cat)), as.character)
  train_w_rules <- lgb.prepare_rules(df.train %>% select(one_of(fnames)), rules)
  rules <- train_w_rules$rules
  train <- train_w_rules$data
  rm(train_w_rules)
  dtrain <- lgb.Dataset(data = as.matrix(train),
                        label = df.train %>% pull(!!rlang::sym(label)) ,
                        categorical_feature = fnames.cat)

  # prepare validation data if any
  valids <- NULL
  if(!is.null(df.valid)){
    message(glue('> rename columns for valid set ...'))
    colnames(df.valid) <- clean.fnames(colnames(df.valid))
    message(glue('> preparing valid set ...'))
    df.valid <- df.valid %>% mutate_at(vars(one_of(fnames.cat)), as.character)
    valid_w_rules <- lgb.prepare_rules(df.valid %>% select(one_of(fnames)), rules)
    valid <- valid_w_rules$data
    rm(valid_w_rules)
    dvalid <- lgb.Dataset(data = as.matrix(valid),
                          label = df.valid %>% pull(!!rlang::sym(label)) ,
                          categorical_feature = fnames.cat)
    valids <- list(train = dtrain, test = dvalid)
  }

  t1 <- Sys.time()

  # prepare id column
  if(is.null(id)) {
    warning('id column is not provided, use row.name as id')
    df.train <- df.train %>%
      tibble::rownames_to_column(var = 'id')
    df.valid <- df.valid %>%
      tibble::rownames_to_column(var = 'id')
    id <- 'id'
  }

  # train cross-validation
  if(cv & nfold > 1){
    message('> start cross-validation ....')
    params[['verbosity']] <- cv.verbose
    bst.cv <- lgb.cv(data = dtrain,
                     record = T,
                     nfold = nfold,
                     params = params,
                     ...)
    res.cv <- lgb.extract_cv_score(bst.cv, score.fun = score.fun)
    cv.score <- res.cv$cv.score
    cv.preds <- res.cv$cv.preds
    cv.preds[id] <- df.train %>% pull(id)
  }

  # train lgb model
  message(glue('> start training LightGBM ...'))
  params[['verbosity']] <- train.verbose
  bst <- lgb.train(data = dtrain,
                   valids = valids,
                   params = params,
                   ...)
  t2 <- Sys.time()

  # extract cv/valid auc metrics
  if(cv){
    lgb.auc.res <- data.frame(
      train_auc_mean = mean(res.cv$cv.score$train),
      cv_auc_mean = mean(res.cv$cv.score$xval),
      cv_auc_std = sd(res.cv$cv.score$xval),
      valid_auc_mean = bst$record_evals$test$auc$eval[[bst$best_iter]],
      stringsAsFactors = F)
  } else {
    lgb.auc.res <- data.frame(
      train_auc_mean = bst$record_evals$train$auc$eval[[bst$best_iter]],
      valid_auc_mean = bst$record_evals$test$auc$eval[[bst$best_iter]],
      stringsAsFactors = F)
  }

  # print result to console
  message('AUC summary:')
  kable(lgb.auc.res, digits = 3, format = 'markdown') %>% print()

  message(glue('preparing lightgbm model takes: {difftime(t1,t0, units = "min") %>% as.numeric() %>% round(2)} mins'))
  message(glue('training lightgbm model takes: {difftime(t2,t1, units = "min") %>% as.numeric() %>% round(2)} mins'))

  # save valid prediction
  valid.preds <- df.valid %>% select(one_of(c(id, label)))
  valid.preds$preds <- predict(bst, as.matrix(valid))

  out <- list(
    model = bst,
    rules = rules,
    fnames = fnames
  )

  # save cross-validation score and predictions
  if(cv){
    cv.score$valid <- lgb.auc.res$valid_auc_mean
    out[['cv.score']] <- cv.score
    out[['cv.preds']] <- cv.preds}

  # save valdiation predictions
  if(!is.null(df.valid)){
    out[['valid.preds']] <- valid.preds
  }

  return(out)
}


#' lgb.predict
#'
#' predict using LGB model
#'
#' @export
#' @param bst boosting model from \code{lgb.train.cv}
#' @param df.test data.frame of test set
#' @param ... predcit.lgb.Booster other params
#' @return array of predictions
#' @examples
#' \dontrun{
#' preds <- lgb.predict(bst, df.valid)
#' }
#'
lgb.predict <- function(bst, df.test, ...){
  # rename
  colnames(df.test) <- clean.fnames(colnames(df.test))
  # encoding
  X <- df.test %>% select(one_of(bst$fnames))
  res <- lgb.prepare_rules(X, rules = bst$rules)
  # predict
  predict(bst$model, data = as.matrix(res$data), ...)
}

#' lgb.save_model
#'
#' save trained LGB model together with related meta-data
#'
#' @export
#' @importFrom utils write.table head
#' @importFrom lightgbm lgb.importance
#' @inheritParams lgb.predict
#' @param model.dir base path for saving model object and meta-data
#' @param model.id identifier for model
#' @param verbose whether display saving information, defaults at TRUE
#' @examples
#' \dontrun{
#' save.lgb.model(bst, './saved_model', 'base_model')
#' }
lgb.save_model <- function(bst, model.dir, model.id = '', verbose = T){

  # define model, rules, fname id
  if(nchar(model.id) > 0)  {
    suffix <- paste0('_',model.id)
  } else {
    suffix <- model.id
  }
  model_id <- glue('model{suffix}.txt')
  rules_id <- glue('rules{suffix}.json')
  feature_id <- glue('fnames{suffix}.txt')

  # init dirs
  sub.dirs <- c('lgb_model',
                'cv_pred','cv_score','valid_pred',
                'var_imp')
  init.dirs(file.path(model.dir, sub.dirs), verbose = verbose)

  # define output names
  out.list <- list()
  for(sub.dir in sub.dirs) {
    in.json <- c('cv_score')
    ext <- ifelse(sub.dir %in% in.json, 'json', 'csv')
    out.list[[sub.dir]] <- file.path(model.dir, sub.dir, glue('{model.id}.{ext}'))
  }

  # save model
  if(verbose) message(glue('> saving model, rules, fnames to [{model.dir}/lgb_model]'))
  lgb.save(bst$model,
           filename = file.path(model.dir, 'lgb_model', model_id))
  # save rule as json
  save.rules(bst$rules,
             file.path(model.dir, 'lgb_model', rules_id))
  # save feature names
  writeLines(bst$fnames,
              file.path(model.dir, 'lgb_model', feature_id))

  # save variable importance
  lgb_var_imp <- lgb.importance(bst$model)
  out <- out.list[["var_imp"]]
  if(verbose) {
    cat(glue('## variable importance - {model.id}:\n'))
    lgb_var_imp %>%
      head(20) %>%
      kable(digits = 3, format = 'markdown') %>%
      print()}
  message(glue('> saving [variable importance] to [{out}]'))
  lgb_var_imp %>% fwrite(out)

  # save cv predictions (train set)
  if('cv.preds' %in% names(bst)){

    out <- out.list[["cv_pred"]]
    if(verbose) message(glue('> saving [cv preds] to [{out}]'))
    fwrite(bst$cv.preds, out)

    out <- out.list[['cv_score']]
    if(verbose) message(glue('> saving [cv scores] to [{out}]'))
    list2json(bst$cv.score, out)
  }

  # save validation predictions (valid set)
  if('valid.preds' %in% names(bst)){
    out <- out.list[["valid_pred"]]
    if(verbose) message(glue('> saving [valid preds] to [{out}]'))
    fwrite(bst$valid.preds, out)
  }
}

#' lgb.load_model

#' Load LGB model

#' @export
#' @importFrom utils read.table
#' @importFrom lightgbm lgb.load
#' @inheritParams lgb.save_model

lgb.load_model <- function(model.dir, model.id = ''){

  if(nchar(model.id) > 0){
    suffix <- paste0('_', model.id)
  } else {
    suffix <- model.id
  }
  model_id <- glue('model{suffix}.txt')
  rules_id <- glue('rules{suffix}.json')
  feature_id <- glue('fnames{suffix}.txt')

  model <- lgb.load(file.path(model.dir,'lgb_model', model_id))
  fnames <- readLines(file.path(model.dir, 'lgb_model', feature_id))
  rules <- load.rules(file.path(model.dir, 'lgb_model',rules_id))

  list(model = model, rules = rules, fnames = fnames)
}
