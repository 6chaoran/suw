#' read.cb
#'
#' read text from clipboard. suitable to copy table from excel to R
#'
#' @importFrom utils read.table
#' @param header whether table has header
#' @param sep separator
#' @return data.frame
#' @export

read.cb <- function(header = T, sep = '\t'){
  read.table(file = "clipboard",
             header = header,
             sep = sep,
             stringsAsFactors = F)
}

#' clean.fnames
#'
#' clean up feature names to make them legal for modelling. Only keep numbers/letters/_.
#'
#' @param fnames feature names
#' @return array of character
#' @examples
#' clean.fnames(c("client's birthday","sum assured in thousand (SGD)"))
#' @export
clean.fnames <- function(fnames){
  fnames <- gsub('[^a-zA-Z0-9_ ]','',fnames)
  fnames <- gsub(' ','_', fnames)
  return(fnames)
}

#' list2json
#'
#' write list to json file
#'
#' @importFrom jsonlite toJSON
#' @export
#' @param x list
#' @param path path of json file
#' @name json_list
#' @examples
#' list2json(list(a = 1, b = 2), 'test.json')
#' json2list('test.json')
list2json <- function(x, path){
  dir <- dirname(path)
  if(!dir.exists(dir)) dir.create(dir, recursive = T)
  x <- toJSON(x, auto_unbox = T)
  write(x, file = path)
}

#' json2list
#'
#' read json file as list
#'
#' @importFrom jsonlite fromJSON
#' @export
#' @name json_list
json2list <- function(path){
  fromJSON(path)
}

#' save.rules
#'
#' save trained rules for categorical features as json file
#'
#' @inheritParams lgb.train_with_cv
#' @export
#' @param path path for rules json file
save.rules <- function(rules, path){
  rules.json <- lapply(rules, function(i) split(i, names(i)))
  list2json(rules.json, path)
}

#' load.rules
#'
#' load rules from json file
#' @export
#' @inheritParams save.rules
load.rules <- function(path){
  json <- json2list(path)
  lapply(json, unlist)
}

#' init.dirs
#'
#' create dirs if not exist
#'
#' @export
#' @param dirs directory or directories
#' @param verbose whether to show messages
#' @examples
#' init.dirs('./saved_model/lgb_model')
#' @importFrom glue glue
#'
init.dirs <- function(dirs, verbose = T){
  for(dir in dirs){
    if(!dir.exists(dir)) {
      dir.create(dir, recursive = T)
      if(verbose) message(glue('> dir[{dir}] is created'))
    }
  }
}
