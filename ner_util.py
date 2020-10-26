import pandas as pd
import numpy as np
import re
import string

# data pre_processing function
def pre_process_data_ner(iob_tagged_df):
    # get the ids for blank rows
    sent_idx = iob_tagged_df.index[iob_tagged_df.isnull().all(axis=1)].tolist()
    #sent_idx[0:5]

    # create sentence number list
    sentence_num_list = [-1] # only for first row
    e_last = 0
    i_last = 0
    for i, (s, e) in enumerate(zip(sent_idx[:-1],sent_idx[1:])):
        sentence_num_list.extend([i for _ in range(s,e)])
        e_last = e
        i_last = i

    # last few rows after last empty row
    sentence_num_list.extend([(i_last + 1) for _ in range(e_last, iob_tagged_df.shape[0])])
    #len(sentence_num_list),iob_tagged_df.shape

    # add sentence number to data frame
    iob_tagged_df['sentence_num'] = sentence_num_list
    #iob_tagged_df.head()

    # get the ids for -DOCSTART- rows
    doc_idx = iob_tagged_df.index[iob_tagged_df['word']=='-DOCSTART-'].tolist()
    doc_idx[0:5]

    # create document number list
    doc_num_list = [] 
    for i, (s, e) in enumerate(zip(doc_idx[:-1],doc_idx[1:])):
        doc_num_list.extend([i for _ in range(s,e)])

    # last few rows after last empty row
    doc_num_list.extend([(i + 1) for _ in range(e, iob_tagged_df.shape[0])])
    #len(doc_num_list),iob_tagged_df.shape

    # add document number to data frame
    iob_tagged_df['doc_num'] = doc_num_list
    #iob_tagged_df.tail()

    # delete all blank and doc start rows
    delete_rows = sent_idx + doc_idx
    iob_tagged_df_cleaned = iob_tagged_df.drop(iob_tagged_df.index[delete_rows])
    # iob_tagged_df_cleaned.tail()

    # print(iob_tagged_df_cleaned.isna().sum())
    # print(iob_tagged_df_cleaned.info())

    # remove rows with null values in any of the word column
    iob_tagged_df_cleaned['word'] = iob_tagged_df_cleaned['word'].replace(' ', np.nan)
    iob_tagged_df_cleaned = iob_tagged_df_cleaned.dropna(axis=0, subset=['word'])
    
    return iob_tagged_df_cleaned
    
    
# function to convert IOB1 to IOB2 format for NER and chunk tags
def iob_to_iob2(iob_tagged_df, convert_tag = 'tag', sentence_id = 'sentence_num'):
    prev_sentence = -1
    iob2_tag = []
    for _, row in iob_tagged_df.iterrows():
        cur_sentence = row[sentence_id]
        if cur_sentence != prev_sentence:
            prev_tag_ent = 'O'
            prev_chunk ='O'

        cur_tag = row[convert_tag]
        if cur_tag == 'O':
            iob2_tag.append('O')
            cur_tag_ent = 'O'
        else:
            cur_tag_ent = cur_tag.split('-')[1]
            if prev_tag_ent != cur_tag_ent:
                iob2_tag.append('B-'+ cur_tag_ent)
            else:
                iob2_tag.append(cur_tag)

        prev_tag_ent = cur_tag_ent
        prev_sentence = cur_sentence
    return iob2_tag
    
def reduce_sentence_length_by_word(df, max_word, sentence_id_col_name):
    sent_word_stat = df.groupby(sentence_id_col_name).size().reset_index()
    # keep sentences that less than or equal to (appproximately) 3rd quartile of number of words
    thirdq_list_sentence_id = list(sent_word_stat.loc[sent_word_stat[0] <= max_word,sentence_id_col_name])
    df_reduced = df.loc[df.sentence_num.isin(thirdq_list_sentence_id),:]
    return df_reduced
    
def format_aux_input(X_aux, max_len, preproc_transform = None,
                    num_col_key_words = ['word.len']):
    aux_dict_list = [aux_dict for dict_list in X_aux for aux_dict in dict_list ]
    stand_aux_df_all = pd.DataFrame(aux_dict_list)
#     print("first step")
    # segregate numeric and non_numeric columns
    num_col_key_words_regex = r'|'.join([re.escape(x) for x in num_col_key_words])
    num_cols_df = stand_aux_df_all.filter(regex=num_col_key_words_regex)
    num_cols_df.fillna(0, inplace = True)
    num_cols = list(num_cols_df.columns)

    # create non numeric colum df for different transformation
    not_num_cols = list(set.difference(set(stand_aux_df_all.columns),set(num_cols)))
#     not_num_cols_df = stand_aux_df_all[stand_aux_df_all.columns[~stand_aux_df_all.columns.isin(num_cols_df)]]
    not_num_cols_df = stand_aux_df_all[not_num_cols]
    not_num_cols_df.fillna('NA', inplace = True)
    not_num_cols_df = not_num_cols_df.astype(str)
#     stand_aux_df_all[not_num_cols] = stand_aux_df_all[not_num_cols].astype(str)
#     print("second step")
    
    # at the time of training
    if preproc_transform is None:

        oh_encoder = OneHotEncoder(handle_unknown='ignore')
        oh_encoder.fit(not_num_cols_df)
    
#         print("third step")
        # at the time of training
    
        standard_transform = StandardScaler()
        standard_transform.fit(num_cols_df)
        
        # transformed column names
        oh_encoded_columns = list(oh_encoder.get_feature_names(not_num_cols))
        transformed_feature_list = num_cols.copy()
        # concatenating categorical columns post numeric columns as that is order in which
        # the transformed metrices are concatenated later in this function
        transformed_feature_list.extend(oh_encoded_columns)
        
        # single object for fitted transformation
        preproc_transform = {"standard_transform":standard_transform,
                             "num_cols": num_cols,
                            "oh_encoder": oh_encoder,
                            "not_num_cols": not_num_cols,
                            'transformed_feature_list':transformed_feature_list}
        
    else: # need to make sure all the columns are as per traning
        standard_transform = preproc_transform['standard_transform']
        oh_encoder = preproc_transform['oh_encoder']
        not_num_cols_org = preproc_transform['not_num_cols']
        num_cols_org = preproc_transform['num_cols']
        
        # check whether all coulmns are present
        
        # numeric
        add_numeric = list(set.difference(set(num_cols_org),set(num_cols)))
        if len(add_numeric) > 0:
            for new_col in add_numeric:
                num_cols_df[new_col] = 0
            
        # categorical
        add_categorical = list(set.difference(set(not_num_cols_org),set(not_num_cols)))
        if len(add_categorical) > 0:
            for new_col in add_categorical:
                not_num_cols_df[new_col] = 'NA'
    
        
#     print("fourth step")
    stand_aux_array_list = []
    for i in range(0, stand_aux_df_all.shape[0], max_len):
        transformed_num_arr = standard_transform.transform(num_cols_df[i:i+max_len])
        transformed_not_num_arr = oh_encoder.transform(not_num_cols_df[i:i+max_len]).toarray()
        transformed_arr = np.column_stack((transformed_num_arr,transformed_not_num_arr))
#         transformed_arr = preproc_transform.transform(stand_aux_df_all)
        transformed_arr = np.nan_to_num(transformed_arr)
        stand_aux_array_list.append(transformed_arr)
    
#     print("fifth step")
    ## convert to nd array
    #print(len(stand_aux_array_list))
    aux_input = np.dstack(stand_aux_array_list)
    aux_input = np.rollaxis(aux_input,-1)
    #aux_input = np.nan_to_num(aux_input)
    return aux_input, preproc_transform
    
class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, c, t) for w, p, c, t in zip(s["word"].values.tolist(),
                                                           s["pos"].values.tolist(),
                                                            s["chunk"].values.tolist(),
                                                           s["tag"].values.tolist())]
        self.grouped = self.data.groupby("sentence_num").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

def pred2label(pred,idx2tag):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out

def get_word_format(text):
    text = re.sub(r'[A-Z]','A',text)
    text = re.sub(r'[a-z]','a',text)
    text = re.sub(r'[0-9]','0',text)
    text = re.sub(r'['+string.punctuation+']','_',text)
    
    return text

def get_word_format_summary(text):
    text = re.sub(r'[A-Z]+','A',text)
    text = re.sub(r'[a-z]+','a',text)
    text = re.sub(r'[0-9]+','0',text)
    text = re.sub(r'['+string.punctuation+']+','_',text)
    
    return text

def word2features(sent, i, num_word_prev, num_word_next):
    word = str(sent[i][0])
    postag = sent[i][1]
    chunktag = sent[i][2]
    features = {
        'word.len()':len(word),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'chunktag': chunktag,
        'chunktag[2:]': chunktag[2:],
        'has_number': bool(re.match(r'[0-9]+',word)),
        'has_upper': bool(re.match(r'[A-Z]+',word)),
        'has_symbol': bool(re.match(r'['+re.escape(string.punctuation)+']+', word)),
        'word_format': get_word_format(word),
        'word_format_summary': get_word_format_summary(word)
    }
    for j in range(num_word_prev,0,-1):
        if i > j-1:
            word1 = str(sent[i-j][0])
            postag1 = sent[i-j][1]
            chunktag1 = sent[i-j][2]
            features.update({
                '-'+str(j)+':word.len()': len(word1),
                '-'+str(j)+':word.istitle()': word1.istitle(),
                '-'+str(j)+':word.isupper()': word1.isupper(),
                '-'+str(j)+':word.isdigit()': word1.isdigit(),
                '-'+str(j)+':postag': postag1,
                '-'+str(j)+':postag[:2]': postag1[:2],
                '-'+str(j)+':chunktag': chunktag1,
                '-'+str(j)+':chunktag[2:]': chunktag1[2:],
                '-'+str(j)+':has_number': bool(re.match(r'[0-9]+',word1)),
                '-'+str(j)+':has_upper': bool(re.match(r'[A-Z]+',word1)),
                '-'+str(j)+':has_symbol': bool(re.match(r'['+re.escape(string.punctuation)+']+', word1)),
                #'-'+str(j)+':word_format': get_word_format(word1),
                '-'+str(j)+':word_format_summary': get_word_format_summary(word1)
            })
        else:
            features['BOS+' + str(i)] = True
    
    for j in range (1, num_word_next + 1):
        if i < len(sent)-j:
            word1 = str(sent[i+j][0])
            postag1 = sent[i+j][1]
            chunktag1 = sent[i+j][2]
            features.update({
                '+'+str(j)+':word.len()': len(word1),
                '+'+str(j)+':word.istitle()': word1.istitle(),
                '+'+str(j)+':word.isupper()': word1.isupper(),
                '+'+str(j)+':word.isdigit()': word1.isdigit(),
                '+'+str(j)+':postag': postag1,
                '+'+str(j)+':postag[:2]': postag1[:2],
                '+'+str(j)+':chunktag': chunktag1,
                '+'+str(j)+':chunktag[2:]': chunktag1[2:],
                '+'+str(j)+':has_number': bool(re.match(r'[0-9]+',word1)),
                '+'+str(j)+':has_upper': bool(re.match(r'[A-Z]+',word1)),
                '+'+str(j)+':has_symbol': bool(re.match(r'['+re.escape(string.punctuation)+']+', word1)),
                #'+'+str(j)+':word_format': get_word_format(word1),
                '+'+str(j)+':word_format_summary': get_word_format_summary(word1)
            })
        else:
            features['EOS-' + str(len(sent) - (i+1))] = True

    return features

def sent2features(sent,  num_word_prev = 2, num_word_next = 2, max_word = None):
    if max_word is None:
        max_word = len(sent)
    return [word2features(sent, i,  num_word_prev, num_word_next)\
            if i < len(sent) else {'word.len()': 0}\
            for i  in range(max_word)]

def sent2labels(sent):
    return [label for token, postag, label in sent]


import keras.backend as K
def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val