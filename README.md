# Local explainability of NER task Masters thesis

1. Source code folder structure
explainable_ner_source_code  
	|- data  
	    |- CoNLL-2003  
	|- src  
	|- models  
	    |- ner  
	    |- glove.840B.300d  
	|- models_improved  
	    |- ner  
	|- models_improved2  
	    |- ner  

2. Place all 3 files for training and testing data inside explainable_ner_source_code> data > CoNLL-2003
3. Download glove embedding from http://nlp.stanford.edu/data/glove.840B.300d.zip and place in explainable_ner_source_code> models > glove.840B.300d
4. Then execute following jupyter notebooks from src directory.  
a. 01_NER_model_building.ipynb – NER model building, will save the models inside models folder  
b. 02_LIME_NER_as_per_original_paper.ipynb – LIME-NER explanation as proposed in original paper on this.  
c. 03_LIME_NER_only_entity_level_prediction_probability.ipynb –LIME-NER explanation with entity level explanation only  
d. 04_KernelSHAP_NER.ipynb – Kernel SHAP explanation for NER  
e. 05_LIME_KSHAP_LIME-NER_all_together.ipynb – this is optional. To see explanation from all 3 methods together  
f. 06_LIME_KSHAP_NER_unkown_word_entity.ipynb – visualize explanation for instances where entity words are unknown to training data set  
g. 07_LIME_KSHAP_explanation_accuracy.ipynb – comparison of explanation accuracy  
h. 08_LIME_KSHAP_explanation_accuracy_unknown_word_entity.ipynb – explanation accuracy for instances where entity words are unknown to training data set  
i. 09_LIME_KSHAP_Time_Efficiency_comparison.ipynb – time efficiency comparison  
j. 10_LIME_KSHAP_word_visualization.ipynb – comprehensibility comparisons  
k. 11_LIME_KSHAP_AOPC_comparison.ipynb – AOPC comparisons  
l. 12_NER_model_improvement_step1_no_context_word_spelling.ipynb – NER model training after reducing spelling and context features to single word. Model will be stored in models_improved directory.  
m. 13_NER_model_improvement_step2_no_spelling_context_feature.ipynb - NER model training after removing spelling and context features. Model will be stored in models_improved2 directory.  
n. 14_NER_test-b_validation_initial_model.ipynb – Initial NER model validation on test-b dataset.  
o. 15_LIME_KSHAP_LIME-NER_individual_explanation_generation.ipynb – This is to generate explanation in ad-hoc basis.. generate explanation for all methods.  
p. ner_util.py – this code does not need to be executed directly. This contains custom functions for NER model predictions and gets invoked from different jupyter notebooks   
q. explanation_util.py – this code does not need to be executed directly. This contains custom functions for explanation generations and gets invoked from different jupyter notebooks  
r. explanation_validation_util.py – this code does not need to be executed directly. This contains custom functions for explanation validations and gets invoked from different jupyter notebooks   
