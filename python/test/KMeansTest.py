import jieba
import re
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.externals import joblib

def jieba_tokenize(text):
    return jieba.lcut(text)

if __name__ == '__main__':
    tfidf_vectorizer =TfidfVectorizer(tokenizer=jieba_tokenize,lowercase=False)

    '''
    tokenizer: 指定分词函数
    lowercase: 在分词之前将所有的文本转换成小写，因为涉及到中文文本处理，
    所以最好是False
    '''

    text_list =["response","rest","resp","req","result",
                "components","numbytes","gzip_data","data",
                "last_byte_pos","numbytes","response_hash"]

    #需要进行聚类的文本集

    tfidf_matrix = tfidf_vectorizer.fit_transform(text_list)
    # print(tfidf_matrix)
    num_clusters = 5
    km_cluster = KMeans(n_clusters=num_clusters)

    #返回各自文本的所被分配到的类索引
    result = km_cluster.fit(tfidf_matrix)
    print(result)
    print (result.labels_)
    # print(km_cluster.inertia_)
    # for i in range(3):
    #     print("Cluster %d:" % i)
    #     for ind in order_centroids[i, :10]:
    #         print(' %s' % terms[ind])
    #     print()
    # print ("Predicting result: ", result)
def _init_raw_data():
    raw_training_data = _load_raw_data("F:\\IntelliJ IDEA\\PythonTest\\python\\test\Data\\same.txt")
    raw_testing_data = _load_raw_data("F:\\IntelliJ IDEA\\PythonTest\\python\\test\\Data\\diff.txt")
    print(raw_training_data)
    print(raw_testing_data)

def _load_raw_data(filepath):
    data = []
    with open(filepath, 'rb') as f:
        for line in f.readlines():
            infos = line.strip().split()
            if not re.match('^_+[\d_]*$', infos[0]):  # we ignore all variables only consists of char '_'
                data.append({'id': infos[0], 'stypes': infos[1].split('|'),
                             'dtypes': infos[2].split('|'), 'context': infos[3],
                             'location': infos[4]})
    return data