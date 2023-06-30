library (tidymodels)
library(tidyr)
library(dplyr)
library(visdat)
library(car)
library(ggplot2)
library(pROC)
library(vip)
library(rpart.plot)
library(DALEXtra)

setwd("C:/Data/Project Data")

hr_train=read.csv('hr_train.csv')
hr_test=read.csv('hr_test.csv')

glimpse(hr_train)

table(hr_train$promotion_last_5years)

dp_pipe=recipe(left~., data=hr_train) %>% 
  update_role(c("sales", "salary"), new_role = "to_dummies") %>% 
  step_unknown(has_role("to_dummies"), new_level = "__missing__") %>% 
  step_other(has_role("to_dummies"), threshold = 0.02, other="__other__") %>% 
  step_dummy(has_role("to_dummies")) %>% 
  step_impute_median(all_numeric(), -all_outcomes())

dp_pipe=prep(dp_pipe)

train=bake(dp_pipe, new_data = NULL)
test=bake(dp_pipe, new_data = hr_test)

set.seed(2)
s=sample(1:nrow(train), 0.8*nrow(train))
t1=train[s,]
t2=train[-s,]

vis_dat(train)  

fit=lm(left~.-sales_X__other__-salary_X__other__, data=t1)
sort(vif(fit), decreasing=T)[1:3]

summary(fit)

log_fit= glm(left~.-sales_X__other__-salary_X__other__, data=t1, family='binomial')
log_fit=stats::step(log_fit)

summary(log_fit)
formula(log_fit)

log_fit=glm(left ~ satisfaction_level + last_evaluation + number_project + 
              average_montly_hours + time_spend_company + Work_accident + 
              sales_IT + sales_management + sales_RandD + sales_sales + 
              salary_low + salary_medium, 
            data=t1)
#AUC score=0.7235

#from feature extraction performed during decision tree
log_fit=glm(left ~ satisfaction_level + last_evaluation + number_project + 
              average_montly_hours + time_spend_company + Work_accident+ 
              sales_IT + promotion_last_5years + 
              salary_low + salary_medium,              , 
            data=t1)
#AUC score=0.7264

val.score=predict(log_fit, newdata = t2, type = "response")
pROC::auc(pROC::roc(t2$left, val.score))



#Decision Tree
setwd("C:/Data/Project Data")

hr_train=read.csv('hr_train.csv')
hr_test=read.csv('hr_test.csv')
glimpse (hr_train)

hr_train$left=as.factor(as.numeric(hr_train$left==1)) 

dp_pipe=recipe(left~., data=hr_train) %>% 
  update_role(c("sales", "salary"), new_role = "to_dummies") %>% 
  step_unknown(has_role("to_dummies"), new_level = "__missing__") %>% 
  step_other(has_role("to_dummies"), threshold = 0.02, other="__other__") %>% 
  step_dummy(has_role("to_dummies")) %>% 
  step_impute_median(all_numeric(), -all_outcomes())

dp_pipe=prep(dp_pipe)

train=bake(dp_pipe, new_data = NULL)
test=bake(dp_pipe, new_data = hr_test)

tree_model=decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

folds=vfold_cv(train, v=10)

tree_grid=grid_regular(cost_complexity(), tree_depth(), 
                       min_n(), levels = 4)

my_res=tune_grid(
  tree_model,
  left~.,
  resamples=folds,
  rid=tree_grid,
  metrics = metric_set(roc_auc),
  control=control_grid(verbose=T)
)

autoplot(my_res)+theme_light()

my_res %>% collect_metrics()

my_res %>% show_best()

final_tree_fit= tree_model %>% 
  finalize_model(select_best(my_res)) %>% 
  fit(left~., data=train)

final_tree_fit %>% 
  vip(geom='col', aesthetics=list(fill='midnightblue', alpha=0.8))+
  scale_y_continuous(expand=c(0,0))

rpart.plot(final_tree_fit$fit)

train.pred=predict(final_tree_fit, new_data = train, type = 'prob') %>% select(.pred_1)
test.pred=predict(final_tree_fit, new_data = test, type = 'prob') %>% select(.pred_1)

write.csv(test.pred, 'Nimmi_Tulsyan_P4_part2.csv', row.names=F)
#AUC score= 0.8380


#Random Forest
rf_tree=rand_forest(
  mtry=tune(),
  trees=tune(),
  min_n=tune()
) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

folds=vfold_cv(train, v=5)

rf_grid=grid_regular(mtry(c(5,15)), trees(c(100,500)), 
                     min_n(c(2,10)), levels=3)
my_res=tune_grid(
  rf_tree,
  left~.,
  resamples=folds,
  grid=rf_grid,
  metrics=metric_set(roc_auc),
  control=control_grid(verbose=T)
)

autoplot(my_res)+theme_light()

my_res %>% show_best()

final_rf_fit=rf_tree %>% 
  set_engine('ranger', importance='permutation') %>% 
  finalize_model( select_best(my_res, "roc_auc")) %>% 
  fit(left~., data=train)

final_rf_fit %>% 
  vip(geom='col', aesthetics=list(fill='midnightblue', alpha=0.8))+
  scale_y_continuous(expand = c(0,0))

train.pred=predict(final_rf_fit, new_data = train, type = 'prob') %>% select(.pred_1)
test.pred=predict(final_rf_fit, new_data = test, type = 'prob') %>% select(.pred_1)

train.score=train.pred$.pred_1
real=train$left

rocit=ROCit::rocit(score = train.score,
                   class=real)
kplot=ROCit::ksplot(rocit)

my_cutoff=kplot$'KS Cutoff'


#partial dependence plots

model_explainer=explain_tidymodels(
  final_rf_fit, 
  data=dplyr::select(train, -left),
  y=as.numeric(train$left),
  verbose=F
)

pdp=model_profile(
  model_explainer,
  variables='satisfaction_level',
  N=2000,
  groups='number_project'
)

plot(pdp)

