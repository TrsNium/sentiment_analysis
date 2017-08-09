import os
from urllib.request import urlopen
import re
import numpy as np

def mk_train_and_test_data(save_dir_path):
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)

    def read_content_save(url, save_dir_path, name):
        content = urlopen(url).read()
        with open(save_dir_path+name, "w") as fs:
            fs.write(content)
        print("success")

    train_data_url = "https://storage.googleapis.com/kaggle-competitions-data/inclass/255etraining.txt?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1502423006&Signature=lBHL5Z3NRdNyAUa%2FaKeWLn9alHhpDW3v4uxlfSuPMojTyiMSwmfVRthsupDnnsSmY1mLyCtUbSUpf%2FxH6nwdBlRgeCwpXVhLLKNDRzc%2B8ZRnWebaK%2FTHVj23Pp%2FON2f5QaEyApdeULrz2RnT07fl8gnTXcMQKug8CcnvKiJAAHCHKoNjV8T9q%2F8S5sYq7wgCuX9C3tLVlIxSP1ozQF2pVDbObCkzODEiF5pVX5Rp8yRHWgCOqoSNvAUKBp0EE8DoE7ncbTCBnFl7%2FFpjHi5QGya582A%2BRm1jaAzaMDrqVoKF4Y8llmIXt%2FyWOn5fUoZpd1uYp%2F8TURXt2hxCY1gxNQ%3D%3D"
    test_data_url = "https://storage.googleapis.com/kaggle-competitions-data/inclass/2558/testdata.txt?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1502422338&Signature=WD%2BoEzW8x%2BnGFl1nB0NCT8Ru5fV%2FtGcv9Gqp6188da6IFbSrKgOly%2FfBKzW%2FKl%2BZiF%2F76R2o%2BeUaWbQKtlf37nOr%2FgXjJMKEfua9UUA%2BfbqZ5P4ItIutUnoEWswFZYY31kSwGTD1ex7kof7nse6zWi2B2shJND8tKK1hyzRBwSnOtl8qQSi1LKGGJVL9XBc4ZlpHnH8m54bO985JWCt85yHNScxY%2BzbDoi2XszVHfHG%2Bue4RN82uXcTvycrLKTKnmELArbDMcZaQsJ7vhrfy2XQRxL8yJsrHmTlxKcGvU11GTAkCrfyNsbjiMVkefaSo1NtB2eDBATZICOKnMHFObQ%3D%3D"
    read_content_save(train_data_url, save_dir_path, "train.txt")
    read_content_save(test_data_url, save_dir_path, "test.txt")
    
def save_index(unique_word_list, save_path):
    #assert not type(unique_word_list) == "list" , "unique_word_list type is not list.please cast it"

    content = "\n".join(list(map(str, unique_word_list)))
    with open(save_path, "w") as fs:
        fs.write(content)

    print("success")

def read_index(data_path):
    with open(data_path, "r") as fs:
        lines = fs.readlines()
    
    return [line.split("\n")[0] for line in lines]
     
    
def remove_anti_pattern(sentence, patterns=[["\.+", " ."], ["\!+", " !"], ["\?+", " ?"], [",", " ,"], ["[\n\x99\x92\x80@�£ã’‘©µ…ªâ*&\]\“^[><_;:+#”$%'\"()=~|¥{}/\\\\]", ""],["\s{2,}", ""]]):
        for pattern in patterns:
            sentence = re.sub(pattern[0], pattern[1], sentence)
        return sentence    

def read_training_data(data_path):
    with open(data_path, "r") as fs:
        lines = fs.readlines()
    
    label = [line.split("	")[0] for line in lines]
    sentences = [remove_anti_pattern(line.split("	")[1].lower()) for line in lines]
    return label, sentences

def convert_sentence2index(sentences, index, time_step):
    r = []
    for sentence in sentences:
        words = remove_anti_pattern(sentence).split(" ")
        converted = [index.index(word) for word in words]
        while len(converted) != time_step and len(converted) <= time_step:
            converted.append(len(index))
        r.append(converted)
    return r

def convert_label(labels):
    r = []
    for label in labels:
        content = [0]*2
        content[int(label)] = 1
        r.append(content)
    return np.array(r)

def convert_sentence2word_idx(sentences, indexs, time_step, word_length):
    r = []
    for sentence in sentences:
        words = remove_anti_pattern(sentence).split(" ")
        t = []
        for word in words[:-1]:
            converted = [indexs.index(char) for char in word]
            while len(converted) != word_length and len(converted) <= word_length:
                converted.append(len(indexs))
            t.append(converted)
            
        while len(t) != time_step and len(t) <= time_step:
            t.insert(0, [len(indexs)+1]*word_length)
        
        r.append(t)
    return r
            

def mk_char_level_cnn_rnn_train_data(data_path, index_path, time_step, word_length=62):
    labels, sentences = read_training_data(data_path)
    if  not os.path.exists(index_path):
        chars = []
        for sentence in sentences:
            s_uniques = list(set(list(sentence)))
            for s_unique in s_uniques:
                if not s_unique in chars:
                    chars.append(s_unique)
        save_index(set(chars), index_path)
    
    indexs = read_index(index_path)
    labels = convert_label(labels)
    converted_sentences = convert_sentence2word_idx(sentences, indexs, time_step, word_length)
    return np.array(labels), np.array(converted_sentences)

def mk_train_data(data_path, index_path, time_step):
    labels, sentences = read_training_data(data_path)
    if  not os.path.exists(index_path):
        word = []
        for r_text in sentences:
            print(r_text)
            [word.append(word_) for word_ in r_text.split(' ')]
        save_index(set(word), index_path)
    
    indexs = read_index(index_path)
    labels = convert_label(labels)
    converted_sentences = convert_sentence2index(sentences, indexs, time_step)
    return np.array(labels), np.reshape(np.array(converted_sentences), (-1, time_steps, 1))
