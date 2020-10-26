import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

import re
import copy

from lime.lime_text import LimeTextExplainer
import shap

from ner_util import format_aux_input, \
                        pred2label, \
                        sent2features


time_efficiency_scoring_weights = {
    '__raw_word': 1,
    'word.len()':1.1,
    'word.isupper()': 0.8,
    'word.istitle()': 0.8,
    'word.isdigit()': 0.8,
    'postag': 0.7,
    'chunktag': 1.2,
    'has_number': 0.8,
    'has_upper': 0.8,
    'has_symbol': 0.9,
    'word_format': 1.2,
    'word_format_summary': 1.2,
    'BOS+':1.5,
    'EOS-':1.5
}

def get_KSHAP_explanation_feature_list(explainer_generator, ref_flat_input, validate_sentences, 
                                    expl_dict, l1_reg = "bic", ent2idx = None, grnd_truth_explanation = False):
    expl_word_list = []
    for (sent_indx, pred_tag,max_p) in expl_dict.keys():
        #label = tag2idx[index]
        text = validate_sentences[sent_indx]

        # 1st Q is 6 for the number of words in sentence in training data
        if len(text) < 6:
            continue

        word_indx_list = expl_dict[(sent_indx, pred_tag,max_p)]['word_indx']
        
        if grnd_truth_explanation: # LeRF explanation needs explanation on the ground truth class
            ground_truth_tag = get_grnd_trth_tag(expl_dict[(sent_indx, pred_tag,max_p)]['actual_tag'])
            p_i = ent2idx[ground_truth_tag]
        else:
            p_i = expl_dict[(sent_indx, pred_tag,max_p)]['p_i']
            
        actual_tag_list = expl_dict[(sent_indx, pred_tag,max_p)]['actual_tag']

        explain_text = " ".join([x[0] for x in text])
        
        # prediction function still on singe word, but that will take care of multiword entities
        predict_func = explainer_generator.get_predict_function(word_index_list= word_indx_list)

        # get 2 dimentional input with all input feartures using pre-process
        flat_input_list, flat_feature_list  = explainer_generator.preprocess([explain_text])

        # we use the reference input as our background dataset to integrate over
        explainer = shap.KernelExplainer(predict_func, ref_flat_input, link = 'logit')

        # explain the first 10 predictions
        # explaining each prediction requires 2 * background dataset size runs
        shap_values = explainer.shap_values(flat_input_list, l1_reg = l1_reg) #"num_features(20)")


        #top 50 features and SHAP values - the number here should not be more than the num_features number above
        explaining_features_all = pd.DataFrame(data=shap_values[p_i], 
                 columns=flat_feature_list[0], 
                 index=[p_i]).transpose().sort_values(by=p_i, ascending=False)

        explaining_features = explaining_features_all.loc[explaining_features_all[p_i] > 0, :]
        #explaining_features_all.iloc[0:20,:]
        
        kshap_expl_feature_list = list(explaining_features.index)
        
        dict_for_metric = {'sent_indx':sent_indx,
                            'explain_text': explain_text,
                          'word_indx_list':word_indx_list,
                          'explain_features': kshap_expl_feature_list,
                          'actual_tag': actual_tag_list,
                          'pred_tag': pred_tag}

        expl_word_list.append(dict_for_metric)
    return expl_word_list
       
def KSHAP_perturb_for_AOPC(explainer_generator, expl_metric_dict, tag, ent2idx, vocab_to_int, L = None, is_MoRF = True):
    
    explain_features_list = expl_metric_dict['explain_features']
    #if L is None: L = len(explain_features_list) 
    
    # prediction function still on singe word, but that will take care of multiword entities
    predict_func = explainer_generator.get_predict_function(word_index_list= expl_metric_dict['word_indx_list'])
    
    initial_text = expl_metric_dict['explain_text']
    
    # get 2 dimentional input with all input feartures using pre-process
    flat_input_list, flat_feature_list  = explainer_generator.preprocess([initial_text])

    # indices of the features in the fat input that neeed to be perturbed
    feature_indexes = [flat_feature_list[0].index(x) for x in explain_features_list]

    # perturbed lists will be added onto this
    perturbed_feature_list = copy.deepcopy(flat_input_list)
    temp_input_list = flat_input_list[0]
    if is_MoRF:
        for i, feature_index in enumerate(feature_indexes):

            feature_name = flat_feature_list[0][feature_index]
            if feature_name.count('>') >= 3: # this means a word feature
                temp_input_list[feature_index] = vocab_to_int.get('<UNK>')
            else: # this means a linguistic feature
                if feature_name.endswith(('_NA','_True','_False')): # this is boolean linguistic feature
                    temp_input_list[feature_index] = 1 - int(temp_input_list[feature_index])
                else: # this is numeric linguistic feature
                    temp_input_list[feature_index] = 0
                
            perturbed_feature_list = np.vstack([perturbed_feature_list,temp_input_list])
            if i + 1 >= L: break
    else: # i.e. LeRF
        for i, feature_index in enumerate(reversed(feature_indexes)):
            feature_name = flat_feature_list[0][feature_index]
            if feature_name.count('>') >= 3: # this means a word feature
                temp_input_list[feature_index] = vocab_to_int.get('<UNK>')
            else: # this means a linguistic feature
                if feature_name.endswith(('_NA','_True','_False')): # this is boolean linguistic feature
                    temp_input_list[feature_index] = 1 - int(temp_input_list[feature_index])
                else: # this is numeric linguistic feature
                    temp_input_list[feature_index] = 0
                
            perturbed_feature_list = np.vstack([perturbed_feature_list,temp_input_list])
            if i + 1 >= L: break
    #print(perturbed_feature_list.shape)
    predict_scores = predict_func(perturbed_feature_list)
    predict_scores_tag = predict_scores[:,ent2idx[tag]]
    # copying the prediction score for last purterbation if all iteration are not complete
    # due to less number of words in explanation
    while len(predict_scores_tag) < (L+1) : 
        predict_scores_tag = np.append(predict_scores_tag,predict_scores_tag[-1])
    return predict_scores_tag

def one_minus_all(ma):
    # substract 1st coulm from all columns including 1st
    return np.reshape(ma[:,0], (ma.shape[0],1)) - ma

def iterative_row_sum_mean_normalize(mat):
    # iterative row sum (addding one at a time)
    # mean
    # devide by L+1
    return [np.mean(np.sum(mat[:,:(l+1)], axis = 1))/(l+2) for l in range(mat.shape[1])]

def get_grnd_trth_tag(tag_list, exclude = 'O'):
    unq_tag = set(tag_list)
    if len(unq_tag) > 1:
        if exclude in unq_tag: unq_tag.remove(exclude)
            
    return list(unq_tag)[0]    
    
    
def AOPC(explainer_generator, expl_metric_dict_list, ent2idx, vocab_to_int,
            expalanation_func = 'KSHAP', L = None):
    MoRF_list = []
    LeRF_list = []
    for expl_metric_dict in expl_metric_dict_list:
        # focus on correctly predicted cases
        tag = get_grnd_trth_tag(expl_metric_dict['actual_tag'])
        if expalanation_func == 'KSHAP':
            MoRF_list.append(KSHAP_perturb_for_AOPC(explainer_generator,
                                expl_metric_dict, tag = tag, ent2idx = ent2idx, 
                                vocab_to_int = vocab_to_int, is_MoRF = True, L = L))
        elif expalanation_func == 'LIME':
            MoRF_list.append(LIME_perturb_for_AOPC(explainer_generator,
                                            expl_metric_dict, tag = tag, ent2idx = ent2idx, 
                                            is_MoRF = True, L = L))
        
        
        # focus on incorrect predictions, thus exclude the correctly predicted tag 
        tag = get_grnd_trth_tag(expl_metric_dict['actual_tag'], exclude = expl_metric_dict['pred_tag'])
        if expalanation_func == 'KSHAP':
            LeRF_list.append(KSHAP_perturb_for_AOPC(explainer_generator, expl_metric_dict, 
                                                   tag = tag, ent2idx = ent2idx, 
                                                   vocab_to_int = vocab_to_int, is_MoRF = False, L = L))
        
        elif expalanation_func == 'LIME':
            LeRF_list.append(LIME_perturb_for_AOPC(explainer_generator, expl_metric_dict,
                                            tag = tag, ent2idx = ent2idx, 
                                            is_MoRF = False, L = L))
    # convert list of ndarrays for further compute
    MoRF_arr = np.array(MoRF_list)
    LeRF_arr = np.array(LeRF_list)
    
    # diffference in probability score between without pertubation and after each iteration of perturbation
    AOPC_MoRF_mat = one_minus_all(MoRF_arr)
    
    # for an iteration of perturbation (e.g. 2 features perturbed for all instances) add the differences of each instance
    # take mean and devide by the number of iteration (i.e. number of features perturbed) + 1
    AOPC_MoRF_list = iterative_row_sum_mean_normalize(AOPC_MoRF_mat)
    
    # diffference in probability score between without pertubation and after each iteration of perturbation
    AOPC_LeRF_mat = one_minus_all(LeRF_arr)
    
    # for an iteration of perturbation (e.g. 2 features perturbed for all instances) add the differences of each instance
    # take mean and devide by the number of iteration (i.e. number of features perturbed) + 1
    AOPC_leRF_list = iterative_row_sum_mean_normalize(AOPC_LeRF_mat)
    
    return AOPC_MoRF_list, AOPC_leRF_list

def get_LIME_explanation_word_list(explainer_generator, explainer, validate_sentences,expl_dict,
                                    ent2idx  = None, grnd_truth_explanation = False):#, num_features=10):
    expl_word_list = []
    for (sent_indx, pred_tag,max_p) in expl_dict.keys():
        #label = tag2idx[index]
        text = validate_sentences[sent_indx]

        # 1st Q is 6 for the number of words in sentence in training data
        if len(text) < 6:
            continue

        word_indx_list = expl_dict[(sent_indx, pred_tag,max_p)]['word_indx']
        
        if grnd_truth_explanation: # LeRF explanation needs explanation on the ground truth class
            ground_truth_tag = get_grnd_trth_tag(expl_dict[(sent_indx, pred_tag,max_p)]['actual_tag'])
            p_i = ent2idx[ground_truth_tag]
        else: 
            p_i = expl_dict[(sent_indx, pred_tag,max_p)]['p_i']
            
        
        explain_text = " ".join([x[0] for x in text])

        # prediction function still on singe word, but that will take care of multiword entities
        predict_func = explainer_generator.get_predict_function(word_index_list= word_indx_list)

        exp = explainer.explain_instance(explain_text, predict_func, 
                                         #num_features=num_features, 
                                            labels=[p_i])
                                            #entity = word_indx_list) # not passing entity list as not using LIME-NER
        lime_expl_list=exp.as_list(p_i)
        # sort descending by score
        lime_expl_list = sorted(lime_expl_list, key=lambda x: x[1], reverse = True)
        lime_expl_word_list = [x[0] for x in lime_expl_list]
        
        dict_for_metric = {'sent_indx':sent_indx,
                            'explain_text': explain_text,
                          'word_indx_list':word_indx_list,
                          'explain_words': lime_expl_word_list,
                          'actual_tag': expl_dict[(sent_indx, pred_tag,max_p)]['actual_tag'],
                          'pred_tag': pred_tag}

        expl_word_list.append(dict_for_metric)
    return expl_word_list
    
    
def LIME_perturb_for_AOPC(explainer_generator, expl_metric_dict, tag, ent2idx, L = None, is_MoRF = True):
    
    if L is None: L = len(expl_metric_dict['explain_words']) 
    
    # prediction function still on singe word, but that will take care of multiword entities
    predict_func = explainer_generator.get_predict_function(word_index_list= expl_metric_dict['word_indx_list'])
    
    initial_text = expl_metric_dict['explain_text']
    texts = [initial_text]
    temp_text = initial_text
    if is_MoRF:
        for i, word in enumerate(expl_metric_dict['explain_words']):
            # replace only 1st instance
            temp_text = re.sub(r'(?<!\S)(%s)(?!\S)' % re.escape(str(word)), '<UNK>',temp_text, 1)
            texts.append(temp_text)
            if i + 1 >= L: break
    else: # i.e. LeRF
        for i, word in enumerate(reversed(expl_metric_dict['explain_words'])):
            # replace only 1st instance
            temp_text = re.sub(r'(?<!\S)(%s)(?!\S)' % re.escape(str(word)), '<UNK>',temp_text, 1)
            texts.append(temp_text)
            if i + 1 >= L: break
    
    predict_scores = predict_func(texts)
    predict_scores_tag = predict_scores[:,ent2idx[tag]]
    # copying the prediction score for last purterbation if all iteration are not complete
    # due to less number of words in explanation
    while len(predict_scores_tag) < (L+1) : 
        predict_scores_tag = np.append(predict_scores_tag,predict_scores_tag[-1])
    return predict_scores_tag


# copied from https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne
def tsne_transform(words_embeddings):
    "Creates and TSNE model"
    labels = []
    tokens = []

    for word, vector in words_embeddings.items():
        tokens.append(vector)
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

#     x = []
#     y = []
    tsne_dict = {}
    for i, value in enumerate(new_values):
#         x.append(value[0])
#         y.append(value[1])
        tsne_dict[labels[i]] = [value[0], value[1]]
    return tsne_dict
    
    
import itertools
def get_top_n_words(expl_dict_list, tsne_model_dict,
                    n=1, is_SHAP_feature = False):
    
     
    feature_list_key = 'explain_features' if is_SHAP_feature else 'explain_words'
            
    top_n_list = []
    top_n_list.extend([expl_dict[feature_list_key][0:n] for expl_dict in expl_dict_list])
    
    merged = list(itertools.chain(*top_n_list))
    top_n_set = set(merged)
    
    unique_feature_list = list(top_n_set)
    
    # if shap feature extract words from the feature
    if is_SHAP_feature:
        unique_feature_list = [feature.split('>')[0] for feature in unique_feature_list \
                               if feature.count('>') >= 3] # this means a word feature

    selected_word_tsne_dict ={k:tsne_model_dict.get(k,tsne_model_dict['<UNK>']) for k in unique_feature_list}
    
    return selected_word_tsne_dict

def get_word_entity_dict(word_dict_list1, word_dict_list2, list1_entity, list2_entity):
    word_entity_dict = {}
    word_dict = {}
    for word, values in word_dict_list1.items():
        word_entity_dict[word] = list1_entity
        word_dict[word] = values
    
    for word in word_dict_list2.keys():
        if word in word_entity_dict.keys():
            word_entity_dict[word] = 'COMMON'
        else:
            word_entity_dict[word] = list2_entity
            word_dict[word] = values
            
    return word_dict, word_entity_dict

from matplotlib import pyplot as plt    
def tsne_plot(tsne_transformed_dict, word_entity_dict = {},
             cdict = {'LOC': 'red', 'MISC': 'blue', 'COMMON': 'green', None:'grey'}):
    
    plt.figure(figsize=(16, 16))
    #ax = plt.gca()
    #fig, ax = plt.subplots()
    for label, x_y in tsne_transformed_dict.items():
        plt.scatter(x_y[0], x_y[1], c = cdict[word_entity_dict.get(label,None)], 
                   s = 100)
        plt.annotate(label,
                     xy=(x_y[0], x_y[1]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    #ax.legend()
    plt.show()
    
def get_time_efficiency_score(features_list,time_efficiency_scoring_weights):
    scores_list = []
    for features in features_list:
        score = 0
        for feature in features:
            if feature.count('>') >= 3:
                score += 1 * time_efficiency_scoring_weights.get('__raw_word', 1)
            elif any(substring in feature for substring in time_efficiency_scoring_weights.keys()):
                substring_index = [x in feature for x in time_efficiency_scoring_weights.keys()].index(True)
                substring = list(time_efficiency_scoring_weights.keys())[substring_index]
                print(substring, time_efficiency_scoring_weights.get(substring, 1))
                score += 1 * time_efficiency_scoring_weights.get(substring, 1)
            else:
                score += 1
        scores_list.append(-1 * score)
            
    return np.mean(scores_list) 
    
def visualize_kernel_shap_explanation(explainer_generator, ref_flat_input, validate_sentences, 
                                    idx2ent, expl_dict, l1_reg = "bic", additional_highlight = None, 
                                    multiplot_legend_location='center right', min_length = 6):
    shap.initjs()
    for (sent_indx, pred_tag,max_p) in expl_dict.keys():
        #label = tag2idx[index]
        text = validate_sentences[sent_indx]

        # 1st Q is 6 for the number of words in sentence in training data
        if len(text) < min_length:
            continue

        word_indx_list = expl_dict[(sent_indx, pred_tag,max_p)]['word_indx']
        p_i = expl_dict[(sent_indx, pred_tag,max_p)]['p_i']
        p_i_next = expl_dict[(sent_indx, pred_tag,max_p)]['p_i_next']
        multiplot_highlight = [p_i,p_i_next]
        if additional_highlight is not None:
            multiplot_highlight.append(get_key(idx2ent,additional_highlight))
        
        actual_tag_list = expl_dict[(sent_indx, pred_tag,max_p)]['actual_tag']

        explian_text = " ".join([x[0] for x in text])
        print()
        print("====================================================================================")
        print("Sentence for explanation: {}".format(explian_text))
        print("Word for explanation: {}".format(" ".join([text[x][0] for x in word_indx_list])))
        print("Predicted named entity for the words: {}".format(pred_tag))
        print("Actual named entity for the words: {}".format(" ".join(actual_tag_list)))
        # prediction function still on singe word, but that will take care of multiword entities
        predict_func = explainer_generator.get_predict_function(word_index_list= word_indx_list)

        # get 2 dimentional input with all input feartures using pre-process
        flat_input_list, flat_feature_list  = explainer_generator.preprocess([explian_text])

        # we use the reference input as our background dataset to integrate over
        explainer = shap.KernelExplainer(predict_func, ref_flat_input, link = 'logit')

        # explain the first 10 predictions
        # explaining each prediction requires 2 * background dataset size runs
        shap_values = explainer.shap_values(flat_input_list, l1_reg = l1_reg)#"num_features(10)")


    #     # plot the explanation of the first prediction
        # Note the model is "multi-output" because it is rank-2 but only has one column
        print("Force plot for entity {}:".format(idx2ent[p_i]))
        display(shap.force_plot(explainer.expected_value[p_i], shap_values[p_i], 
                                    flat_feature_list[0], link = 'logit')) #, matplotlib=True)

        #Display top 10 features and SHAP values
        print("Top 10 features and SHAP values for entity {}:".format(idx2ent[p_i]))
        top10_df = pd.DataFrame(data=shap_values[p_i], 
                         columns=flat_feature_list[0], 
                         index=[p_i]).transpose().sort_values(by=p_i, ascending=False)
        top10_df = top10_df.loc[top10_df[p_i] != 0, :]
        display(top10_df)

        print("Multiple output decision ploit highlighting entity {} and {}:".format(idx2ent[p_i], idx2ent[p_i_next]))
        r = shap.multioutput_decision_plot(explainer.expected_value.tolist(),
                                       shap_values,
                                       0,
                                       feature_names=flat_feature_list[0],
                                       feature_order='importance',
                                       highlight=multiplot_highlight ,
                                       legend_labels=idx2ent.values(),
                                       return_objects=True,
                                       legend_location=multiplot_legend_location)

        display(r)

        print("Summary plot with top 10 features and for entity {}:".format(idx2ent[p_i]))
        display(shap.summary_plot(shap_values[p_i], flat_feature_list[0],
                     max_display = top10_df.shape[0],
                     class_names=idx2ent,
                     class_inds=idx2ent.keys()))

        print("Summary plot for all classes together:")
        display(shap.summary_plot(shap_values, 
                      flat_feature_list[0],
                     max_display = 10,
                     class_names=idx2ent,
                     class_inds=idx2ent.keys()))

        #Display all features and SHAP values
        print("top 10 fetures and SHAP values for all classes together")
        df_list = []
        for ent_indx in idx2ent.keys():
            df_list.append(pd.DataFrame(data=shap_values[ent_indx], columns=flat_feature_list[0], index=[ent_indx]))

        df=pd.concat(df_list)
        df = df.transpose()
        df['total']= df.abs().sum(axis = 1)
        df = df.sort_values(by = 'total', ascending = False)
        df = df.loc[df['total'] > 0, :]
        df.drop(columns = 'total', inplace = True)
        df.rename(columns = idx2ent, inplace = True)
        display(df)

def get_key(my_dict,val): 
    for key, value in my_dict.items(): 
         if val == value: 
             return key 
  
    return "key doesn't exist"
        
def visualize_LIME_explanations(explainer_generator, explainer, validate_sentences, expl_dict, 
                                LIME_NER = False, min_length = 6, num_features=5):
    for (sent_indx, pred_tag,max_p) in expl_dict.keys():
        #label = tag2idx[index]
        text = validate_sentences[sent_indx]

        # 1st Q is 6 for the number of words in sentence in training data
        if len(text) < min_length:
            continue

        word_indx_list = expl_dict[(sent_indx, pred_tag,max_p)]['word_indx']
        p_i = expl_dict[(sent_indx, pred_tag,max_p)]['p_i']
        p_i_next = expl_dict[(sent_indx, pred_tag,max_p)]['p_i_next']
        actual_tag_list = expl_dict[(sent_indx, pred_tag,max_p)]['actual_tag']

        explian_text = " ".join([x[0] for x in text])

        print()
        print("====================================================================================")
        print("Sentence for explanation: {}".format(explian_text))
        print("Word for explanation: {}".format(" ".join([text[x][0] for x in word_indx_list])))
        print("Predicted named entity for the words: {}".format(pred_tag))
        print("Actual named entity for the words: {}".format(" ".join(actual_tag_list)))

    #     print("Sentence for explanation: {}".format(explian_text))
    #     print("Word for explanation: {}".format(" ".join([text[x][0] for x in word_indx_list])))
        # prediction function still on singe word, but that will take care of multiword entities
        predict_func = explainer_generator.get_predict_function(word_index_list= word_indx_list)
        if LIME_NER:
            exp = explainer.explain_instance(explian_text, predict_func, num_features=num_features, 
                                                labels=[p_i], #p_i_next],
                                                top_labels = 2,
                                                entity = word_indx_list) #  passing entity list as using LIME-NER
        else:
            exp = explainer.explain_instance(explian_text, predict_func, num_features=num_features, 
                                                labels=[p_i], #p_i_next],
                                                top_labels = 2)
                                                #entity = word_indx_list) # not passing entity list as not using LIME-NER
        exp.show_in_notebook(text=True)
        
