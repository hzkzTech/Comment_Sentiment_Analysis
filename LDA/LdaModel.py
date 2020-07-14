# _coding_='utf-8'
import json
import pandas as pd
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def loaddata_csv(filepath):
    '''
    load comments from .csv
    :param filepath: csvfile path
    :return: dataframe
    '''
    data=pd.read_csv(filepath, encoding = 'utf-8', header = None)
    data=data.drop(index=[0])
    data=data.drop(columns=[0])

    return data

def loaddata_json(filepath):
    '''
    load comments with tag from .json
    :param filepath: jsonfile path
    :return: dict,keys are the tags
    '''
    json_file=open(filepath,encoding='utf-8')
    data=json.load(json_file)

    return data

def LDA(data,components,htmlfile=None):
    '''
    LDA process
    :param data: list,a list of comments
    :param components: int,num of components
    :param htmlfile: str,path to save outHtmlFile
    :return: none
    '''

    # 关键词提取和向量转化
    tf_vectorizer = CountVectorizer(max_features=1000,
                                    max_df=0.5,
                                    min_df=10,
                                    encoding='utf-8'
                                    )
    tf = tf_vectorizer.fit_transform(data)

    lda = LatentDirichletAllocation(n_components=components,
                                    max_iter=50,
                                    learning_method='online',
                                    learning_offset=50,
                                    random_state=0,
                                    )
    lda.fit(tf)

    result = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
    if htmlfile:
        pyLDAvis.save_html(result,htmlfile)

if __name__ == '__main__':
    components=15

    filepath_pos='pos_comment_0.960_2.csv' #去掉介词，人名，地点，时间等
    filepath_neg='neg_comment_0.960_2.csv' #去掉介词，人名，地点，时间等
    pos=loaddata_csv(filepath_pos)
    neg=loaddata_csv(filepath_neg)

    LDA(pos[1],components,'{}_lda.html'.format(filepath_pos[:-4]))
    LDA(neg[1],components,'{}_lda.html'.format(filepath_neg[:-4]))

    #读取按标签分类文件
    tag_pos_file='tag_pos_comments.json'
    tag_neg_file='tag_neg_comments.json'
    tag_pos=loaddata_json(tag_pos_file)
    tag_neg=loaddata_json(tag_neg_file)

    for tag2 in tag_neg.keys():
        LDA(tag_neg[tag2],7,'.\\neg\\{}_neg_lda.html'.format(tag2))

    for tag in tag_pos.keys():
        LDA(tag_pos[tag],components,'.\\pos\\{}_pos_lda.html'.format(tag))

