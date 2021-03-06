# -*- coding:utf-8 -*-
# @Time： 2020/7/11 10:31 PM
# @Author: dyf-2316
# @FileName: getData.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description: entrance to the WebCrawler

from Logger import Logger
from WebCrawler.getData import get_search_url, get_product_data, get_comment_num, get_comment_data, get_product_id
from WebCrawler.saveData import save_to_mongo


mylogger = Logger("webCrawler").logger

if __name__ == '__main__':
    # keywords = input('输入商品关键字：')
    search_url = get_search_url('美的热水器')
    mylogger.info("获取商品ID")
    products = get_product_id(search_url)
    for i in range(len(products)):
        mylogger.info("获取第商品信息 -- id:{} [{}/{}]".format(products[i], (i+1), len(products)))
        product = get_product_data(products[i])
        if not product:
            continue
        mylogger.info("获取商品评论总数 -- id:{} [{}/{}]".format(products[i], (i+1), len(products)))
        comment_num = get_comment_num(products[i])
        for k in range(0, 8):
            for j in range(int(comment_num / 10)):
                mylogger.info("获取评论 -- id:{} [{}/{}] score:{} page:{}".format(products[i], (i+1), len(products), k, j+1))
                comment_data = get_comment_data(products[i], k, j, product)
                mylogger.info("数据存入数据库 -- id:{} [{}/{}] score:{} page:{}".format(products[i], (i+1), len(products), k, j+1))
                if not comment_data:
                    break
                save_to_mongo(comment_data)
                mylogger.info("已存入数据库 -- id:{} [{}/{}] score:{} page:{}".format(products[i], (i+1), len(products), k, j+1))
        mylogger.info("数据爬取完毕")

