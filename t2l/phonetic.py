from pypinyin import lazy_pinyin
# https://modelpredict.com/language-identification-survey#benchmarked-libraries
# import pycld2 as cld2  # great speed, but inaccurate
import fasttext
import logging
import pykakasi  # ATTENTION: GPL licensed
import kroman
import re
import cyrtranslit

logger = logging.getLogger("phonetize")
kks = pykakasi.kakasi()
fasttext_model = fasttext.load_model('lid.176.ftz')


def phonetize(ch):
    """
    暂时只识别：zh,ja,ko,en
    :return: 返回数组[(word, phone), ...], 且转换后的phone只包含a-z'~字串
    [('拼', 'pin'), ('音', 'yin'), ('123', '123'), ('english !', 'english !')]
    ['igeoseun', 'je', 'geosi', 'anieyo.imyeongssiui', 'geosieyo.'];
    若无法识别语言，返回[]
    """
    ch = q2bs(ch)  # 全角转半角

    # 先识别语言，再分别转读音
    if ch.isascii():  # 纯ascii或空str，直接返回
        return [(ch, ch)]

    # help(cld2.detect)
    # reliable, _, details = cld2.detect(ch, isPlainText=True, returnVectors=False,
    #                                    hintLanguageHTTPHeaders='zh,ja,ko,en,ru')
    lang_result = detect_language(ch)

    # 暂时只取1种语言
    ret = []
    if lang_result == '__label__zh':
        # 对hhh h234 对冯绍峰撒发!fds dui地方
        ret = lazy_pinyin(ch)
        ret = convertPairsPinyin(ch, ret)
    elif lang_result == '__label__en':
        ret, phs, idx = [], ch.split(), 0
        for ii, ph in enumerate(phs):
            idx1  = ch.find(phs[ii + 1], idx+len(ph)) if ii < len(phs) - 1 else None
            ret.append((ch[idx:idx1], ph))
            idx = idx1
    elif lang_result == '__label__ja':
        # おはようごぎ.い!ます. thank you 123! ohayougogi. かな漢字交じり文.: 'ohayougogi.', 'i!', 'masu.', ' thank you 123!', 'kana', 'kanji', 'majiri', 'bun.'
        ret = kks.convert(ch)
        ret = convertPairsJp(ret)
    elif lang_result == '__label__ko':
        # 이것은 제 것이 아니에요.이명씨의 것이에요. thank-you 123: i-geos-eun je geos-i a-ni-e-yo.i-myeong-ssi-eui geos-i-e-yo. thank-you 123 geos je
        ret = kroman.parse(ch)
        ret = convertPairsKorean(ch, ret)
    elif lang_result == '__label__ru':
        # Моё судно на воздушной подушке полно угрей
        ret = cyrtranslit.to_latin(ch)
        ret = convertPairsRussian(ch, ret)
    else:
        logger.error('Unsupported language detect: %s, %s', lang_result, ch)
    return ret


def detect_language(txt):
    # 先正则匹配cjk
    if re.search('[\u30a0-\u30ff\u3040-\u309f]+', txt):
        return '__label__ja'
    if re.search('[\u4e00-\u9fa5]+', txt):
        return '__label__zh'
    if re.search('[\uac00-\ud7ff]+', txt):
        return '__label__ko'
    if re.search('[\u0400-\u04FF]+', txt):
        return '__label__ru'

    lang_result = fasttext_model.predict(
        txt, k=1)  # (('__label__en',), (0.95,))
    if len(lang_result[-1]) == 0 or lang_result[-1][0] < 0.5:
        logger.warning('unreliable language detect: %s, details=%s',
                    lang_result, txt)
    return lang_result[0][0] if len(lang_result[0]) > 0 else 'un'


def convertPairsJp(phones):
    """
    标点排除在phone外
    おはようごぎ.い!ます. thank you 123!かな漢字交じり文.: 'ohayougogi.', 'i!', 'masu.', ' thank you 123!', 'kana', 'kanji', 'majiri', 'bun.'
    :returns: [('おはようごぎ.', 'ohayougogi'), ('い!', 'i'), ('ます.', 'masu'), (' thank you 123!', ' thank you 123!'), ('かな', 'kana'), ('漢字', 'kanji'), ('交じり', 'majiri'), ('文.', 'bun')]
    """
    ret = []
    for item in phones:
        wd, ph = item['orig'], item['hepburn']
        if wd != ph and re.search("[^a-z'~]", ph):
            ph = re.sub("[^a-z'~]", '', ph)
        ret.append((wd, ph))
    return ret


def convertPairsKorean(words, phones):
    """
    convertPairsKorean('ilove이것은 제 것이 아니에요...이명씨의 것이에요. thank-you 123 geos je', 'ilovei-geos-eun je geos-i a-ni-e-yo...i-myeong-ssi-eui geos-i-e-yo. thank-you 123')
    :returns: [('ilove', 'ilove'), ('이', 'i'), ('것', 'geos'), ('은', 'eun'), (' 제', 'je'), (' 것', 'geos'), ('이', 'i'), (' 아', 'a'), ('니', 'ni'), ('에', 'e'), ('요', 'yo'), ('...', '...'), ('이', 'i'), ('명', 'myeong'), ('씨', 'ssi'), ('의', 'eui'), (' 것', 'geos'), ('이', 'i'), ('에', 'e'), ('요', 'yo'), ('. thank-you 123', '. thank-you 123')]
    """
    ret, idx0, wd = [], 0, ''
    for ii, ch in enumerate(words):
        # print(ii, ch, idx0, wd)
        idx1 = phones.find(ch, idx0)
        if idx1 < 0:  # korean
            if len(wd) > 0:
                if len(wd.strip()) <= 0:
                    ch = wd + ch
                else:
                    ret.append((wd, wd))
                wd = ''
            chn = words[ii + 1] if ii < len(words) - 1 else ''
            idx2 = phones.find(chn, idx0+1)
            idx1 = phones.find('-', idx0) if idx2 < 0 else idx2
            ret.append((ch, phones[idx0: None if chn == '' else idx1]))
            idx0 = idx1 + (1 if idx2 < 0 else 0)
        else:
            idx1 += 1
            wd += phones[idx0:idx1]
            idx0 = idx1
    if len(wd) > 0:
        ret.append((wd, wd))
    return ret


def convertPairsPinyin(words, phones):
    """
    convertPairsPinyin('hhh h234 对冯绍峰撒发!fds dui地方', ['hhh h234 ', 'dui', 'feng', 'shao', 'feng', 'sa', 'fa', '!fds dui', 'di', 'fang'])
    """
    ret, idx0 = [], 0
    for ph in phones:
        idx1 = idx0 + len(ph)
        if words[idx0:idx1] == ph:
            ret.append((ph, ph))
            idx0 = idx1
        else:  # pinyin
            ret.append((words[idx0], ph))
            idx0 += 1
    return ret


def convertPairsRussian(words, phones):
    """
    convertPairsRussian()
    """
    ret, phs, chs = [], phones.split(' '), words.split(' ')
    for ii, ph in enumerate(phs):
        if  ii < len(phs):
            ret.append((chs[ii], ph))

    return ret

def q2bs(unicode_str):
    """ 全角转半角
    q2bs("电影《2012》讲述了2012年12月21日的世界末日,主人公Jack以及世界各国人民挣扎求生的经历!。“f”·1‘’－【（）")
    """
    return u"".join(map(q2b, unicode_str))


def q2b(ch):
    """ 全角转半角"""
    od = ord(ch)
    if 12288 == od:
        return ' '
    elif 12289 == od:  # 、
        return ','
    elif 12290 == od:
        return '.'
    elif 12298 == od:
        return '<'
    elif 12299 == od:
        return '>'
    elif 12304 == od:
        return '['
    elif 12305 == od:
        return ']'
    elif 8212 == od:
        return '-'
    elif 183 == od:
        return '`'
    elif 8221 == od or 8220 == od:
        return '"'
    elif 8216 == od or 8217 == od:
        return "'"
    elif 65281 <= od <= 65374:
        return chr(od - 65248)
    return ch


if __name__ == '__main__':
    ch = '对hhh h234 对冯绍峰撒发!fds dui地方'
    ret = lazy_pinyin(ch)
    ret = convertPairsPinyin(ch, ret)
    print(ret)
    ch = '것은 제 것이 아니에요.이명씨의 것이에요. thank-you 123 geos je'
    ret = kroman.parse(ch)
    ret = convertPairsKorean(ch, ret)
    print(ret)
    ch = 'おはようごぎ.い!ます. thank you 123! ohayougogi. かな漢字交じり文.'
    ret = kks.convert(ch)
    ret = convertPairsJp(ret)
    print(ret)
    ch = 'Моё'
    ret = cyrtranslit.to_latin(ch)
    ret = convertPairsRussian(ch, ret)
    print(ret)
    
