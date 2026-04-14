from chardet.universaldetector import UniversalDetector
import logging


def __get_file_encoding(file):
    """
    获取文本文件的编码类型
    :param file: file name
    :return string: 返回encoding或'utf8'
    """
    txt = open(file, "rb")
    detector, cnt = UniversalDetector(), 0
    for line in txt.readlines():
        cnt = cnt + 1
        detector.feed(line)
        if detector.done and cnt > 3:  # parse 3 lines at least
            break
    detector.close()
    txt.close()
    encoding = detector.result
    if encoding['encoding'] == 'GB2312':
        encoding['encoding'] = 'GBK'
    logging.info("_get_file_encoding: %s", encoding['encoding'])
    encoding = encoding['encoding'] if encoding else None
    return encoding or 'utf8'
