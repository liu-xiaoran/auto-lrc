import re


def parse_time(time_str):

    match = re.match(r"(\d+):(\d+)(?:\.(\d+))?", time_str)

    # 获取分钟、秒和小数部分
    minutes, seconds, milliseconds = match.groups()

    seconds = int(minutes) * 60 + int(seconds)

    milliseconds = float('0.' + milliseconds)
    
    return seconds + milliseconds

def lrc_to_json(lrc_text):
    """
    将歌词文本转换为 JSON 格式

    Args:
        lrc_text: 歌词文本

    Returns:
        JSON 格式的歌词
    """
    # 返回的歌词对象
    lyricslrc, idx, timelist = [], 0, []

    # 去除[]时间    
    lrc_text = re.sub(r"\[(.*?)\]", "", lrc_text)

    # 匹配歌词行, 先按\n分行
    lines = re.split(r"\n", lrc_text)

    # 遍历每行
    for line in lines:
        # 每行歌词
        lineslrc = []

        if len(line.strip()) == 0:
            continue

        lrc_text = re.split(r"<(.*?)>", line)
        for words in lrc_text:
            print(words, end="\n")
            if len(words.strip()) > 0:
                if re.match(r"^\d{2}:\d{2}\.\d+$", words):
                    # 时间参数
                    timelist.append(parse_time(words))
                    continue
                else:
                    lineslrc.append({
                        'word': words, 
                        'idx': idx
                        })
                    idx += 1
                
            else:
                continue

        lyricslrc.append(lineslrc)

    for i in range(len(lyricslrc)):
        for j in range(len(lyricslrc[i])):
            idx = lyricslrc[i][j]['idx']
            if idx+1 >= len(timelist):
                lyricslrc[i][j]['start'] = timelist[idx]
                lyricslrc[i][j]['end'] = timelist[idx] + 1
            else:
                lyricslrc[i][j]['start'] = timelist[idx]
                lyricslrc[i][j]['end'] = timelist[idx+1]

    return lyricslrc


