# coding:utf8
import jieba
import gensim.models.word2vec as w2v

def segmentFile(input_file, outout_file):
    '''
    分词

    :return:
    '''
    with open('倚天屠龙记.Txt', encoding='utf-8') as fin:
        with open('倚天屠龙记_segmented.txt', mode='w', encoding='utf-8') as fout:
            line = fin.readline()
            while line:
                sarry = jieba.cut(line, cut_all=False)
                strout = ' '.join(sarry).replace('，','').replace('。', '').replace('？', '').replace('！', '').replace('”', '') \
                .replace('“', '').replace('：', '').replace('‘', '').replace('’', '').replace('-', '').replace('（', '') \
                .replace('）', '').replace('《', '').replace('》', '').replace('；', '').replace('.', '').replace('、', '') \
                .replace('…', '').replace(',', '').replace('?', '').replace('!', '')
                fout.write(strout + '\n')
                line = fin.readline()


def train(input_file, model_file):
    sentences = w2v.LineSentence(input_file)
    model = w2v.Word2Vec(sentences, size=20, window=5, min_count=5, workers=4)
    model.save(model_file)

def validate(model_file):
    model = w2v.Word2Vec.load(model_file)


