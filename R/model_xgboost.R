#' @import dplyr
#' @importFrom rlang sym
#' @importFrom lightgbm lgb.prepare_rules
#'
NULL


#' xgb.train_with_cv
#'
#' Train Xgboost with Cross-valiation
#'
#' @export
#' @importFrom xgboost xgb.DMatrix xgb.train xgb.cv
#' @importFrom lightgbm lgb.prepare_rules
#' @importFrom rlang sym
#' @importFrom tibble rownames_to_column
#' @importFrom knitr kable
#' @importFrom stats sd predict
#' @importFrom utils tail head
#' @inheritParams lgb.train_with_cv
#' @param params list of params for xgboost
#' @param ... other xgboost parameters
#' @return trained xgboost model
#' @examples
#' label <- 'label'
#' fnames <- c('Sex','Class','Age','Freq')
#' fnames.cat <- c('Sex','Age','Class')
#' df <- data.frame(Titanic)
#' df$label <- ifelse(df$Survived == "Yes", 1 ,0)
#' in.train <- runif(nrow(df)) < 0.8
#' df.train <- df[in.train, ]
#' df.valid <- df[!in.train, ]
#'
#' params <- list(max_depth = 2, eta = 1,
#'                nthread = 2,objective = "binary:logistic",
#'                eval_metric = "auc")
#' bst <- xgb.train_with_cv(df.train, df.valid, fnames,
#'                        label, fnames.cat, params = params,
#'                        cv.verbose = 1, train.verbose = 1,
#'                        cv = TRUE, nfold = 2,
#'                        print_every_n = 5L,
#'                        early_stopping_rounds = 5L,
#'                        nrounds = 100L)
#'
#' xgb.save_model(bst, 'saved_model','xgb_baseline')
#' bst_loaded <- xgb.load_model('saved_model','xgb_baseline')
#' preds <- xgb.predict(bst_loaded, df.valid)

xgb.train_with_cv <- function(df.train, df.valid, fnames, label, fnames.cat, id = NULL,
                         rules = NULL, params = NULL,
                         cv.verbose = 0, train.verbose = 1,
                         cv = FALSE, nfold = 5, score.fun = Metrics::auc,
                         ...){


  t0 <- Sys.time()

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
  train.x <- train_w_rules$data
  train.y <- df.train %>% pull(!!sym(label))
  dtrain <- xgb.DMatrix(data = as.matrix(train.x),
                        label = train.y,
                        missing = NA)

  # prepare validation data if any
  if(!is.null(df.valid)){
    message(glue('> rename columns for valid set ...'))
    colnames(df.valid) <- clean.fnames(colnames(df.valid))
    message(glue('> preparing valid set ...'))
    df.valid <- df.valid %>% mutate_at(vars(one_of(fnames.cat)), as.character)
    valid_w_rules <- lgb.prepare_rules(df.valid %>% select(one_of(fnames)), rules)
    valid.x <- valid_w_rules$data
    valid.y <- df.valid %>% pull(!!sym(label))
    dvalid <- xgb.DMatrix(data = as.matrix(valid.x),
                          label = valid.y,
                          missing = NA)
    watchlist <- list(train = dtrain, test = dvalid)
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
  if(nfold > 1 & cv){
    message(glue('> start cross-validation ...'))

    bst.cv <- xgb.cv(data = dtrain,
                     verbose = cv.verbose,
                     params = params,
                     nfold = nfold,
                     prediction = TRUE, ...)

    cv.preds <- df.train %>% select(one_of(c(id, 'label')))
    cv.preds[['preds']] <- bst.cv[['pred']]
    cv.preds[['folds']] <- NA
    for(i in 1:nfold){
      cv.preds[['folds']][bst.cv[['folds']][[i]]] <- i
    }
    cv.score <- cv.preds %>%
      group_by(folds) %>%
      summarise(metrics = score.fun(label, preds)) %>%
      pull('metrics')
    cv.score <- list(xval = cv.score)

  } else {
    cv.score <- list()
    cv.preds <- NULL
  }

  # train xgboost model
  message(glue('> start training Xgboost ...'))
  bst <- xgb.train(data = dtrain,
                   watchlist = watchlist,
                   verbose = train.verbose,
                   params = params,
                   ...)
  if(!'best_score' %in% names(bst)){
    bst$best_iteration <- nrow(bst$evaluation_log)
    bst$best_score <- bst$evaluation_log$test_auc %>% tail(1)
  }
  cv.score[['train']] <- bst$evaluation_log[bst$best_iteration, 2][[1]]
  cv.score[['valid']] <- bst$evaluation_log[bst$best_iteration, 3][[1]]
  valid.preds <- df.valid %>% select(one_of(c(id, 'label')))
  valid.preds$preds <- predict(bst, dvalid)
  t2 <- Sys.time()

  # extract cv/valid auc metrics
  if(cv){
    xgb.auc.res <- data.frame(
      train_auc_mean = mean(cv.score$train),
      cv_auc_mean = mean(cv.score$xval),
      cv_auc_std = sd(cv.score$xval),
      valid_auc_mean = bst$best_score[[1]],
      stringsAsFactors = F, row.names = NULL)
  } else {
    xgb.auc.res <- data.frame(
      train_auc_mean = mean(cv.score$train),
      valid_auc_mean = bst$best_score,
      stringsAsFactors = F, row.names = NULL)
  }

  # print result to console
  xgb.auc.res %>% kable(digits = 3, format = 'markdown') %>% print()
  print(glue('preparing xgboost model takes: {difftime(t1,t0, units = "min") %>% as.numeric() %>% round(2)} mins'))
  print(glue('training xgboost model takes: {difftime(t2,t1, units = "min") %>% as.numeric() %>% round(2)} mins'))

  out <- list(model = bst,
              valid.preds  = valid.preds,
              fnames = bst$feature_names,
              rules = rules)

  if(cv & nfold > 1){
    out[['cv.score']] <- cv.score
    out[['cv.preds']] <- cv.preds
  }

  return(out)
}


#' xgb.save_model
#'
#' save Xgboost model from \code{xgb.train_with_cv}
#'
#' @export
#' @importFrom xgboost xgb.importance xgb.save
#' @inheritParams lgb.save_model
#' @bst list of output from \code{xgb.train_with_cv}
#'
xgb.save_model <- function(bst, model.dir, model.id = '', verbose = T){

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
  sub.dirs <- c('xgb_model',
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
  if(verbose) message(glue('> saving model, rules, fnames to [{model.dir}/xgb_model]'))
  xgb.save(bst$model,
           fname=file.path(model.dir, 'xgb_model', model_id))
  # save rule as json
  save.rules(bst$rules,
             file.path(model.dir, 'xgb_model', rules_id))
  # save feature names
  writeLines(bst$fnames,
              file.path(model.dir, 'xgb_model', feature_id))

  # save variable importance
  var_imp <- xgb.importance(model = bst$model)
  out <- out.list[["var_imp"]]
  if(verbose) {
    cat(glue('## variable importance - {model.id}:\n'))
    var_imp %>%
      head(20) %>%
      kable(digits = 3, format = 'markdown') %>%
      print()}
  message(glue('> saving [variable importance] to [{out}]'))
  var_imp %>% fwrite(out)

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

#' xgb.load_model
#'
#' load model from saved directory
#'
#' @export
#' @importFrom  xgboost xgb.load
#' @inheritParams lgb.load_model
xgb.load_model <- function(model.dir, model.id = ''){

  if(nchar(model.id) > 0){
    suffix <- paste0('_', model.id)
  } else {
    suffix <- model.id
  }
  model_id <- glue('model{suffix}.txt')
  rules_id <- glue('rules{suffix}.json')
  feature_id <- glue('fnames{suffix}.txt')

  model <- xgb.load(file.path(model.dir,'xgb_model', model_id))
  fnames <- readLines(file.path(model.dir, 'xgb_model', feature_id))
  rules <- load.rules(file.path(model.dir, 'xgb_model',rules_id))

  list(model = model, rules = rules, fnames = fnames)
}

#' xgb.predict
#'
#' predict from xgboost model
#'
#' @export
#' @inheritParams lgb.predict
xgb.predict <- function(bst, df.test, ...){
  # rename
  colnames(df.test) <- clean.fnames(colnames(df.test))
  # encoding
  X <- df.test %>% select(one_of(bst$fnames))
  res <- lgb.prepare_rules(X, rules = bst$rules)
  # predict
  predict(bst$model,
          xgb.DMatrix(as.matrix(res$data), missing = NA),
          ...)
}





