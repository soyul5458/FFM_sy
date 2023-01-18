# FFM
frequency feature masking

## Dataset 
* ASV2019 - LA, PA
* ADD

## LA, ADD 데이터 셋에서 FFM의 성능 실험
* FFM_LA_SY.ipynb
* FFM_ADD_SY.ipynb
* run_FFM_final.ipynb
* Hyperparam_tunning_sy.ipynb

## hyperparameter search of data augmentation (cutout, cutmix, specaug, specmix )
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
  * Results  
   * DataSize_None.csv 
   * DataSize_HP.csv  
   * DataSize_HP_LP.csv  


## Blur masking 실험
* run_FFM_blur_final.ipynb
* Hyperparam_tunning_sy.ipynb
* 차이검정 t-test
  * Blur_T-test_ADD.ipynb
  * Blur_T-test_ASV.ipynb
   * Results
    *  Blur_EER_ADD(mask, blur).xlsx
    *  Blur_EER_ASV(mask, blur).xlsx
    * BarPlot_mask_blur_eer.png: blur vs mask 모델별 augmentation별 성능 막대그래프  
    * ADD_EER, ASV_EER : 데이터별 blur vs mask 모델별 augmentation별 성능 막대그래프
    *  FFM_Result_220609.csv : Mask vs blur 모델(5)별 augmentation(8)별 EER table
    *  FFM_Result_trans_220609.csv : 위 엑셀에서 transpose 한 table
