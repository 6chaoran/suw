# suw

h2o, xgboost and lightgbm model wrapper for streamlined underwriting

## Installation

```r
# lightgbm package is required prior installation
install.packages('remotes')
remotes::install_github('6chaoran/suw')
```

## Usage

```r
library(suw)

# save h2o model and related extra results
h2o.save_model

# train xgboost model together with cross validation
xgb.train_with_cv

# save/load xgboost model and related extra results
xgb.save_model
xgb.load_model

# train lightgbm model together with cross validation
lgb.train_with_cv

# save/load lgboost model and related extra results
lgb.save_model
lgb.load_model
```
