# coding:utf8
import jieba
import gensim.models.word2vec as w2v

def segmentFile(input_file, outout_file):
    '''
    分词

    :return:
    '''
    with open(input_file, encoding='utf-8') as fin:
        with open(outout_file, mode='w', encoding='utf-8') as fout:
            line = fin.readline()
            while line:
                sarry = jieba.cut(line, cut_all=False)
                strout = ' '.join(sarry).replace('，','').replace('。', '').replace('？', '').replace('！', '').replace('”', '') \
                .replace('“', '').replace('：', '').replace('‘', '').replace('’', '').replace('-', '').replace('（', '') \
                .replace('）', '').replace('《', '').replace('》', '').replace('；', '').replace('.', '').replace('、', '') \
                .replace('…', '').replace(',', '').replace('?', '').replace('!', '')
                fout.write(strout + '\n')
                line = fin.readline()
    print("segment done")


def train(input_file, model_file):
    sentences = w2v.LineSentence(input_file)
    model = w2v.Word2Vec(sentences, size=20, window=5, min_count=5, workers=4)
    model.save(model_file)
    print('train done')

def validate(model_file):
    model = w2v.Word2Vec.load(model_file)
    print(model.wv.similarity('赵敏', '赵敏'))
    print(model.wv.similarity('赵敏', '张无忌'))
    print(model.wv.similarity('屠龙刀', '倚天剑'))
    print(model.wv.similarity('成昆', '谢逊'))
    print(model.wv.similarity('成昆', '好大'))
    print(model.wv.get_vector('赵敏'))

    for k in model.wv.similar_by_word('张三丰', topn=15):
        print(k[0], k[1])


if __name__ == '__main__':
    input_file = '倚天屠龙记.Txt'
    output_file = '倚天屠龙记_segmented.txt'
    model_file = '倚天屠龙记_model.txt'
    # segmentFile(input_file, output_file)
    # train(output_file, model_file)
    validate(model_file)


