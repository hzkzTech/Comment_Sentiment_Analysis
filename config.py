import logging
import os

DEFAULT_PRODUCT_NUM = 20

DEFAULT_COMMENT_NUM = 50000

SEARCH_URL = "https://search.jd.com/Search?keyword={}&qrst=1&suggest=1.his.0.0&stock=1&page=1&s=61&click=0"

PRODUCT_URL = "https://item.jd.com/{}.html"

COMMENT_URL = "https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId={" \
              "}&score={}&sortType=5&page={}&pageSize=10&isShadowSku=0&rid=0&fold=1 "

PRICE_URL = "https://p.3.cn/prices/mgets?skuIds=J_{}"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'
}

DATABASE_MONGO = {
    'host': 'localhost',
    'port': 27017
}

DEFAULT_LOG_LEVEL = logging.DEBUG

DEFAULT_LOG_FMT = '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s'

DEFAULT_LOG_DATEFMT = '%Y-%m-%d %H:%M:%S'

# 这里使用绝对路径，因为日志文件访问会在对象生成文件下进行相对路径访问，会出现路径错误的情况
DEFAULT_DATA_LOG_FILENAME = os.getcwd() + '/log/data.log'

DEFAULT_PROCESS_LOG_FILENAME = os.getcwd() + '/log/precess.log'

DEFAULT_ERROR_LOG_FILENAME = os.getcwd() + '/log/error.log'

