import gensim
from gensim.test.utils import  datapath,get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

# 转换函数，将Glove格式转换为word2vec格式

def transfer():
    gloveFile=datapath('/home/gpu2/bolvvv/glove.42B.300d.txt')
    word2vecFile=get_tmpfile('/home/gpu2/bolvvv/word2vec.42B.300d.txt')
    glove2word2vec(gloveFile,word2vecFile)

transfer()