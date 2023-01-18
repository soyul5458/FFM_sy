# FFM
frequency feature masking

## Dataset 
* ASV2019 - LA, PA
* ADD

## LA, ADD 데이터 셋에서 FFM의 성능 실험
* FFM_LA_SY.ipynb
* FFM_ADD_SY.ipynb
* Hyperparam_tunning_sy.ipynb

## hyperparameter search (cutout, cutmix, specaug, specmix )
음성에서 많이 사용되는 다양한 augmentation 실험
* run_FFM_hp_search_ADD
* run_FFM_hp_search_LA
* Hyperparam_tunning_sy_augmentation

## filter augmentation (line, step)
* run_FFM_hp_search_ADD
* run_FFM_hp_search_LA
* Hyperparam_tunning_sy_augmentation


## 데이터 크기에 따른 모델별 성능 차이 실험
* FFM_LA_Experiments by Data Size.ipynb  
* FFM_LA_Experiments by Data Size_0.1_0.9.ipynb  
* Hyperparam_tunning_sy_DatsSize.ipynb


## Blur masking 실험
* run_FFM_blur_final.ipynb
* Hyperparam_tunning_sy.ipynb
* 차이검정 t-test
  * Blur_T-test_ADD.ipynb
  * Blur_T-test_ASV.ipynb
