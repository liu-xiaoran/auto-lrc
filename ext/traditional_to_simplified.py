from opencc import OpenCC

cc = OpenCC('t2s')

def t2s(traditional_text):
      # 't2s' 表示繁体到简体
    return cc.convert(traditional_text)


