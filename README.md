*NOTE:* This code is incomplete, and untested.

Python implementation of the rulefit (http://statweb.stanford.edu/~jhf/ftp/RuleFit.pdf) algorithm (with support for xgboost).

The algorithm is a multi-step process:
1) Generate a tree ensemble using random forest/gradient boosting
2) Use the trees to form rules, with each decision path in a tree forming one rule. 
3) Prune the rules and the original input features using L1-regularised regression (LASSO)

Largely written before discovering the more complete implementation here: https://github.com/christophM/rulefit
