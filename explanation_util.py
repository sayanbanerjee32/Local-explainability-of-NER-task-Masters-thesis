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

def get_entity_prediction(validate_pred, idx2tag, root_tags = ['PER','LOC','ORG','MISC']):
    pred_df_col_names = [idx2tag.get(id) for id in sorted(idx2tag.keys())]
    entity_prob_arr_list = []
    entity_full_list = root_tags.copy()
    entity_full_list.append('O')
    for val_sent_pred in validate_pred:
        val_sent_pred_df = pd.DataFrame(val_sent_pred, columns= pred_df_col_names)
        #print(val_sent_pred_df.shape)
        val_sent_pred_df['pred_entity'] = val_sent_pred_df.idxmax(axis=1)
        
        entity_prob_dict = {}
        avg_entity_prob_dict = {}
        # for each row:
        for _, row in val_sent_pred_df.iterrows():
        #     if an entity prob is highest:
            if row['pred_entity'] is not 'O':
                for root_tag in root_tags:
                    b_tag = 'B-' + root_tag
                    i_tag = 'I-' + root_tag
                    #if B tag prob is higher:
                    if row[b_tag] > row[i_tag]:
                        # first check  if avg dict any of element left to be added to main dict
                        if avg_entity_prob_dict:
                            avg_list = avg_entity_prob_dict.get(root_tag, [])
                            if len(avg_list) > 0:

                                entity_avg = sum(avg_list)/len(avg_list)
                                # extend the original list by the number of probabilities being averaged
                                entity_prob_dict.setdefault(root_tag, []).extend([entity_avg] * len(avg_list))
                                # reset the list for doing average to emty list
                                avg_entity_prob_dict[root_tag] = []
                        # then add the b-tag prob in the list for averaging
                        avg_entity_prob_dict.setdefault(root_tag, []).append(row[b_tag])
                    else:
                        # if i-tag is higher keep the i-tag porib for averaging
                        avg_entity_prob_dict.setdefault(root_tag, []).append(row[i_tag])
                # need to retain probability of O for averaging
                avg_entity_prob_dict.setdefault('O', []).append(row['O'])
            else: # if row['pred_entity'] is 'O'
                # first check  if avg dict any of element left to be added to main dict
                if avg_entity_prob_dict:
                    # for every root tag calculate seperately
                    for root_tag in root_tags:
                        avg_list = avg_entity_prob_dict.get(root_tag, [])
                        if len(avg_list) > 0:

                            entity_avg = sum(avg_list)/len(avg_list)
                            # extend the original list by the number of probabilities being averaged
                            entity_prob_dict.setdefault(root_tag, []).extend([entity_avg] * len(avg_list))
                            # reset the list for doing average to emty list
                            avg_entity_prob_dict[root_tag] = []

                    # similarly averaging will need to happen for O tag
                    avg_list = avg_entity_prob_dict.get('O', [])
                    if len(avg_list) > 0:
                        entity_avg = sum(avg_list)/len(avg_list)
                        # extend the original list by the number of probabilities being averaged
                        entity_prob_dict.setdefault('O', []).extend([entity_avg] * len(avg_list))
                        # reset the list for doing average to emty list
                        avg_entity_prob_dict['O'] = []
                    # as O is encountered reset avg dict
                    avg_entity_prob_dict = {}
                # for root tags keep the max of B and I tags - no avg
                for root_tag in root_tags:
                    b_tag = 'B-' + root_tag
                    i_tag = 'I-' + root_tag
                    tag_prob = max(row[b_tag],row[i_tag])
                    # if the dict does not have list for the tag, then titialise with a dict
                    entity_prob_dict.setdefault(root_tag, []).append(tag_prob)
                # keep single probability for O tag
                entity_prob_dict.setdefault('O', []).append(row['O'])

            #prev_entity = row['pred_entity']
        # if avg dict hve some residual while processing last row.
        if avg_entity_prob_dict:
            # for every root tag calculate seperately
            for root_tag in root_tags:
                avg_list = avg_entity_prob_dict.get(root_tag, [])
                if len(avg_list) > 0:
                    #if len(avg_list) > 1: ready_to_break = True
                    entity_avg = sum(avg_list)/len(avg_list)
                    # extend the original list by the number of probabilities being averaged
                    entity_prob_dict.setdefault(root_tag, []).extend([entity_avg] * len(avg_list))
                    # reset the list for doing average to emty list
                    avg_entity_prob_dict[root_tag] = []
            # similarly averaging will need to happen for O tag
            avg_list = avg_entity_prob_dict.get('O', [])
            if len(avg_list) > 0:
                entity_avg = sum(avg_list)/len(avg_list)
                # extend the original list by the number of probabilities being averaged
                entity_prob_dict.setdefault('O', []).extend([entity_avg] * len(avg_list))
                # reset the list for doing average to emty list
                avg_entity_prob_dict['O'] = []
        # convert to dataframe
        entity_prob_df = pd.DataFrame(entity_prob_dict)
        entity_prob_df = entity_prob_df[entity_full_list]
        #print(entity_prob_df.columns)
        entity_prob_arr_list.append(entity_prob_df.to_numpy())

    # finally return numpy nd array
    entity_prob_mat = np.array(entity_prob_arr_list) 
    # new id to entity 
    idx2ent = {i: t for i, t in enumerate(entity_full_list)}
    return entity_prob_mat, idx2ent
    
def get_explanation_instances(validate_entity_prob_mat,validate_true_entity_prob_mat,
            idx2ent, idx2ent_true, 
        select_tags = ['LOC','MISC'],
        selection_prob = 0.95):
    
    expl_selected_dict ={}

    for sent_indx, (pred_i, true_pred_i) in enumerate(zip(validate_entity_prob_mat,validate_true_entity_prob_mat)):
        
        for word_indx, (p, true_p) in enumerate(zip(pred_i, true_pred_i)):
            # predicted
            p_i = np.argmax(p)
            max_p = max(p)
            pred_tag = idx2ent[p_i].replace("PAD", "O")
            
            # actual
            actual_p_i = np.argmax(true_p)
            actual_tag = idx2ent_true[actual_p_i].replace("PAD", "O")
            
            if pred_tag in select_tags and max_p >= selection_prob:
                p_i_next = np.argsort(-p)[1] #bn.argpartition(-p, 0)[1]
                p_next = p[p_i_next]
                pred_tag_next = idx2ent[p_i_next].replace("PAD", "O")
                if expl_selected_dict.get((sent_indx, pred_tag,max_p)) is None:
                    expl_selected_dict[(sent_indx, pred_tag,max_p)] = {'word_indx':[word_indx], 
                                                                    'p_i':p_i, 
                                                                    'p_i_next':p_i_next,
                                                                    'pred_tag_next':pred_tag_next,
                                                                    'p_next':p_next,
                                                                    'is_accurate': actual_tag == pred_tag,
                                                                    'actual_tag': [actual_tag]}
                else:
                    expl_selected_dict[(sent_indx, pred_tag,max_p)]['word_indx'].append(word_indx)
                    expl_selected_dict[(sent_indx, pred_tag,max_p)]['actual_tag'].append(actual_tag)
                    expl_selected_dict[(sent_indx, pred_tag,max_p)]['is_accurate'] = bool(expl_selected_dict[(sent_indx,
                                                                                            pred_tag,max_p)]['is_accurate']* (actual_tag == pred_tag))


    return expl_selected_dict

class NER_KSHAPExplainerGenerator(object):
    
    def __init__(self, model, word2idx, tag2idx, max_len, 
                 sent_getter_id_dict, sent_word_getter_id_dict, sentences, 
                 trained_preprocess_transform, num_word_next, num_word_prev,
                root_tags = ['PER','LOC','ORG','MISC']):
        self.model = model
        self.word2idx = word2idx
        self.idx2word = {v: k for k,v in word2idx.items()}
        self.tag2idx = tag2idx
        self.idx2tag = {v: k for k,v in tag2idx.items()}
        self.max_len = max_len
        self.sent_getter_id_dict = sent_getter_id_dict
        self.sent_word_getter_id_dict = sent_word_getter_id_dict
        self.sentences = sentences
        self.trained_preprocess_transform = trained_preprocess_transform
        self.root_tags = root_tags
        self.num_word_prev = num_word_prev
        self.num_word_next = num_word_next
        
    def preprocess(self, texts):
        #print(texts)
        X = [[self.word2idx.get(w, self.word2idx["<UNK>"]) for w in t.split()]
             for t in texts]
        X = pad_sequences(maxlen=self.max_len, sequences=X,
                          padding="post", value=self.word2idx["<PAD>"])
        
        # trying to find out the text from validation sentences, 
        # assumtion we are using validation sentences for explanation
        X_sents_idx = [self.sent_getter_id_dict.get(text) for text in texts]
        
        # ideally it should match with only one sentence 
        # othere are perturbed sentences of the original sentence
        X_sent_idx = [x for x in X_sents_idx if x is not None]
        
        if len(X_sent_idx) > 0: # this means the sentence is from validation set
            # get the tuple of POS and chunk for the words in that sentence
            original_sent_word_dict = self.sent_word_getter_id_dict.get(X_sent_idx[0],{})
        else: # generating reference / random data
            original_sent_word_dict = {}
        
        # even for purturbed sentence, use the POS and CHUNK for the words that are still unchanged in the sentence
        X_sents = [[original_sent_word_dict.get((i,word),('','unk','unk','unk')) for i, 
                    word in enumerate(text.split())]for text in texts]
        
        X_aux =  [sent2features(X_sent, self.num_word_prev, self.num_word_next, self.max_len) for X_sent in X_sents]

        X_aux_input, _ = format_aux_input(X_aux, max_len = self.max_len, 
                                             preproc_transform = self.trained_preprocess_transform)
                                           #oh_encoder = trained_oh_encoder, standard_transform = trained_standard_transform)
        
        flat_input = self._flatten_processed_input((X, X_aux_input))
        
        # prediction for feature enrichment, so that predicted tag be used for contect words
        pred_for_feature_enrichment = self.model.predict([X, X_aux_input], verbose=1)
        entity_prob_mat, idx2ent = get_entity_prediction(pred_for_feature_enrichment, self.idx2tag)
        entity_tags_list = pred2label(entity_prob_mat, idx2ent)

        # creating flat_feature_list (list of list), this will contain the words and CRF features for all those words repeated 
        # for number of word, aspiration is to add te words in the CRF feature names
        flat_feature_list = self._flatten_feature_list(X, X_sents, texts, entity_tags_list)
        
        return flat_input, flat_feature_list
    
    def _flatten_crf_features(self,word_list):
        crf_feature_list_all_words = []
        # for every word add CRF feature list 
        for i, word in enumerate(word_list):
            if word == '': word = 'UNK'
#                 crf_feature_list = [str(word) +'_'+ str(feature) for feature in \
#                                     self.trained_preprocess_transform.get('transformed_feature_list')]

            crf_feature_list = []   
            for feature in self.trained_preprocess_transform.get('transformed_feature_list'):
                # split based on the ':' between relative word index and the feature iside CRF feature names
                split_feature = re.findall("^(\+\d+|^-\d+)\:(.*)",feature)
                if len(split_feature) > 0:
                    # extract main word and context word
                    context_word_relative_indx = int(split_feature[0][0])

                    # check of out of index
                    #print(i, context_word_relative_indx)
                    if (i + context_word_relative_indx) >= 0 and  (i + context_word_relative_indx) < len(word_list):
                        context_word = word_list[i + context_word_relative_indx]
                    else: 
                        context_word = 'OutOfIndex'

                    processed_feature_name = str(word) +':'+ split_feature[0][0] + \
                                        ':' + context_word + '__' + split_feature[0][1]
                    crf_feature_list.append(processed_feature_name)
                else:
                    crf_feature_list.append(str(word) +'__'+ str(feature))
                
            crf_feature_list_all_words.extend(crf_feature_list)
                
        return crf_feature_list_all_words
    
    def _enrich_word_features(self,word_feature_list,sent_pos_chunk, entity_tags, seperator_char = '>'):
        enriched_word_feature_list = []

        for word_position, word in enumerate(word_feature_list):
            
            # enrich feature with word prostion
            enriched_feature = str(word) + seperator_char + str(word_position)+ seperator_char
            
            # enrich feature with POS tag
            if word_position < len(sent_pos_chunk): 
                enriched_feature += sent_pos_chunk[word_position][1]
             
            enriched_feature += seperator_char
            
            # enrich feature with predicted NER tag
            if word_position < len(entity_tags): 
                enriched_feature += entity_tags[word_position]
             
                       
            enriched_word_feature_list.append(enriched_feature)
        return enriched_word_feature_list
                    
    def _flatten_feature_list(self, word_indx_nd_array, sentences_pos_chunk, texts, entity_tags_list):
        
        feature_nd_list = []
        
        for word_indx, single_sent_pos_chunk, single_sentence, entity_tags in \
                        zip(word_indx_nd_array, sentences_pos_chunk, texts, entity_tags_list):
            # get the original words as well
            org_text_words_list = [t for t in single_sentence.split()]
            # get the words from word indices
            embd_word_list = [self.idx2word.get(indx,'NOT_EXPECTED') for indx in word_indx]
            
            # adding original word as well, so that we can distingusih between known and unknown embedding words
            word_list = []
            for emb_indx, word in enumerate(embd_word_list):
                if emb_indx < len(org_text_words_list):
                    if word != org_text_words_list[emb_indx]: 
                        # for unknown words with respect to embedding, keep original version as well
                        word_list.append(str(org_text_words_list[emb_indx]) + '|' + str(word))
                    else:
                        word_list.append(word)
                else:
                    word_list.append(word)
            
            feature_list = word_list.copy()
            
            enriched_word_feature_list = self._enrich_word_features(feature_list, single_sent_pos_chunk, entity_tags)
            
            crf_feature_list = self._flatten_crf_features(word_list)
            
            enriched_word_feature_list.extend(crf_feature_list)

            feature_nd_list.append(enriched_word_feature_list)
        
        return feature_nd_list
    
    def _flatten_processed_input(self, input_tup):
        X = input_tup[0]
        X_aux_input = input_tup[1]
        X_aux_input_flat = X_aux_input.transpose(0,1,2).reshape(X_aux_input.shape[0],-1)
        X_all_array = np.column_stack((X,X_aux_input_flat))
        return X_all_array
    
    def _unflatten_flat_input(self, flat_input):
        
        X = flat_input[:,0:self.max_len]
        X_aux_input = flat_input[:, self.max_len:].reshape(flat_input.shape[0],self.max_len, -1)
        return X, X_aux_input
    
    def _get_entity_i_prediction(self, validate_pred, entity_word_indx_list):
        pred_df_col_names = [self.idx2tag.get(id) for id in sorted(self.idx2tag.keys())]
        entity_prob_arr_list = []
        entity_full_list = self.root_tags.copy()
        entity_full_list.append('O')
        for val_sent_pred in validate_pred:
            val_sent_pred_df = pd.DataFrame(val_sent_pred, columns= pred_df_col_names)
            
            entity_prob_dict = {}
            avg_entity_prob_dict = {}
            # for each row:
            for row_indx, row in val_sent_pred_df.iterrows():
            #     row index in word index list
                if row_indx in entity_word_indx_list:
                    for root_tag in self.root_tags:
                        b_tag = 'B-' + root_tag
                        i_tag = 'I-' + root_tag
                        # row index in first of entty word index
                        if row_indx == entity_word_indx_list[0]:
                            # then add the b-tag prob in the list for averaging
                            avg_entity_prob_dict.setdefault(root_tag, []).append(row[b_tag])
                        else:
                            # for subsequent rows as i-tag probability
                            avg_entity_prob_dict.setdefault(root_tag, []).append(row[i_tag])
                    # need to retain probability of O for averaging
                    avg_entity_prob_dict.setdefault('O', []).append(row['O'])
            # if avg dict hve some residual while processing last row.
            if avg_entity_prob_dict:
                # for every root tag calculate seperately
                for root_tag in self.root_tags:
                    avg_list = avg_entity_prob_dict.get(root_tag, [])
                    if len(avg_list) > 0:
                        #if len(avg_list) > 1: ready_to_break = True
                        entity_avg = sum(avg_list)/len(avg_list)
                        # extend the original list by the number of probabilities being averaged
                        entity_prob_dict.setdefault(root_tag, []).extend([entity_avg] * len(avg_list))
                        # reset the list for doing average to emty list
                        avg_entity_prob_dict[root_tag] = []
                # similarly averaging will need to happen for O tag
                avg_list = avg_entity_prob_dict.get('O', [])
                if len(avg_list) > 0:
                    entity_avg = sum(avg_list)/len(avg_list)
                    # extend the original list by the number of probabilities being averaged
                    entity_prob_dict.setdefault('O', []).extend([entity_avg] * len(avg_list))
                    # reset the list for doing average to emty list
                    avg_entity_prob_dict['O'] = []
            # convert to dataframe
            entity_prob_df = pd.DataFrame(entity_prob_dict)
            entity_prob_df = entity_prob_df[entity_full_list]
            #print(entity_prob_df.columns)
            entity_prob_arr_list.append(entity_prob_df.to_numpy())

        # finally return numpy nd array
        entity_prob_mat = np.array(entity_prob_arr_list) 
        #print(entity_prob_mat)
        return entity_prob_mat
    
    def get_predict_function(self, word_index_list):
        def predict_func(flat_input):
            #print(flat_input)
            X, X_aux_input = self._unflatten_flat_input(flat_input)
            #print(X.shape)
            #print(X_aux_input.shape)
            p = self.model.predict([X,X_aux_input])
            # revise predicted probability vector for entity probabilities
            p_entity = self._get_entity_i_prediction(p, word_index_list)
            return p_entity[:,0,:]
        return predict_func   

    
    
class NER_LIMEExplainerGenerator(object):
    
    def __init__(self, model, word2idx, tag2idx, max_len, 
                 sent_getter_id_dict, sent_word_getter_id_dict, sentences, 
                 trained_preprocess_transform, num_word_next,num_word_prev,
                root_tags = ['PER','LOC','ORG','MISC']):
        self.model = model
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.idx2tag = {v: k for k,v in tag2idx.items()}
        self.max_len = max_len
        self.sent_getter_id_dict = sent_getter_id_dict
        self.sent_word_getter_id_dict = sent_word_getter_id_dict
        self.sentences = sentences
        self.trained_preprocess_transform = trained_preprocess_transform
        self.root_tags = root_tags
        self.num_word_next = num_word_next
        self.num_word_prev = num_word_prev
        
    def _preprocess(self, texts):
        #print(texts)
        X = [[self.word2idx.get(w, self.word2idx["<UNK>"]) for w in t.split()]
             for t in texts]
        X = pad_sequences(maxlen=self.max_len, sequences=X,
                          padding="post", value=self.word2idx["<PAD>"])
        
        X_sents_idx = [self.sent_getter_id_dict.get(text) for text in texts]
        X_sent_idx = [x for x in X_sents_idx if x is not None]
        original_sent_word_dict = self.sent_word_getter_id_dict[X_sent_idx[0]]
        
        X_sents = [[original_sent_word_dict.get((i,word),('','unk','unk','unk')) for i, 
                    word in enumerate(text.split())]for text in texts]
        
        X_aux =  [sent2features(X_sent, self.num_word_prev, self.num_word_next, self.max_len) for X_sent in X_sents]

        X_aux_input, _ = format_aux_input(X_aux, max_len = self.max_len, 
                                             preproc_transform = self.trained_preprocess_transform)
                                           #oh_encoder = trained_oh_encoder, standard_transform = trained_standard_transform)
        return X, X_aux_input
    
    def get_entity_i_prediction(self, validate_pred, entity_word_indx_list):
        pred_df_col_names = [self.idx2tag.get(id) for id in sorted(self.idx2tag.keys())]
        entity_prob_arr_list = []
        entity_full_list = self.root_tags.copy()
        entity_full_list.append('O')
        for val_sent_pred in validate_pred:
            val_sent_pred_df = pd.DataFrame(val_sent_pred, columns= pred_df_col_names)
            
            entity_prob_dict = {}
            avg_entity_prob_dict = {}
            # for each row:
            for row_indx, row in val_sent_pred_df.iterrows():
            #     row index in word index list
                if row_indx in entity_word_indx_list:
                    for root_tag in self.root_tags:
                        b_tag = 'B-' + root_tag
                        i_tag = 'I-' + root_tag
                        # row index in first of entty word index
                        if row_indx == entity_word_indx_list[0]:
                            # then add the b-tag prob in the list for averaging
                            avg_entity_prob_dict.setdefault(root_tag, []).append(row[b_tag])
                        else:
                            # for subsequent rows as i-tag probability
                            avg_entity_prob_dict.setdefault(root_tag, []).append(row[i_tag])
                    # need to retain probability of O for averaging
                    avg_entity_prob_dict.setdefault('O', []).append(row['O'])
            # if avg dict hve some residual while processing last row.
            if avg_entity_prob_dict:
                # for every root tag calculate seperately
                for root_tag in self.root_tags:
                    avg_list = avg_entity_prob_dict.get(root_tag, [])
                    if len(avg_list) > 0:
                        #if len(avg_list) > 1: ready_to_break = True
                        entity_avg = sum(avg_list)/len(avg_list)
                        # extend the original list by the number of probabilities being averaged
                        entity_prob_dict.setdefault(root_tag, []).extend([entity_avg] * len(avg_list))
                        # reset the list for doing average to emty list
                        avg_entity_prob_dict[root_tag] = []
                # similarly averaging will need to happen for O tag
                avg_list = avg_entity_prob_dict.get('O', [])
                if len(avg_list) > 0:
                    entity_avg = sum(avg_list)/len(avg_list)
                    # extend the original list by the number of probabilities being averaged
                    entity_prob_dict.setdefault('O', []).extend([entity_avg] * len(avg_list))
                    # reset the list for doing average to emty list
                    avg_entity_prob_dict['O'] = []
            # convert to dataframe
            entity_prob_df = pd.DataFrame(entity_prob_dict)
            entity_prob_df = entity_prob_df[entity_full_list]
            #print(entity_prob_df.columns)
            entity_prob_arr_list.append(entity_prob_df.to_numpy())

        # finally return numpy nd array
        entity_prob_mat = np.array(entity_prob_arr_list) 
        #print(entity_prob_mat)
        return entity_prob_mat
    
    def get_predict_function(self, word_index_list):
        def predict_func(texts):
            X, X_aux_input = self._preprocess(texts)
#             print(X.shape)
#             print(X_aux_input.shape)
            p = self.model.predict([X,X_aux_input])
            # revise prodicted probability vector for entity probabilities
            p_entity = self.get_entity_i_prediction(p, word_index_list)
            return p_entity[:,0,:]
        return predict_func
