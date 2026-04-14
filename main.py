#!/usr/bin/env python
import argparse
import ext.t2lutils as t2lutils
from t2l.t2l import process


def cli():
    parser = argparse.ArgumentParser(description='自动打轴工具')
    parser.add_argument('lrc_file', help='歌词文本')
    parser.add_argument('music_file', help='歌曲文件')
    # parser.add_argument('-g', '--gpu', help='是否使用GPU, 默认：0', default=0, required=False)
    parser.add_argument('-f', '--format', type=str, default='lrc',
                        help='输出文件格式，支持lrc（默认），srt', required=False)
    parser.add_argument('-l', '--line_only', type=int,
                        help='1:逐句/0:逐字, 默认：逐字', required=False)
    parser.add_argument('-v', '--vocalize', type=int, default=1,
                        help='1:分离人声/0:不分离, 默认：分离', required=False)
    parser.add_argument(
        '-m', '--model', help='demucs模型[mdx|mdx_extra|mdx_q|mdx_extra_q], 默认：mdx_extra', default='mdx_extra', required=False)
    parser.add_argument(
        '-i', '--idx', help='demucs模型idx, 默认：-1', default=-1, required=False)

    args = parser.parse_args()

    txt_lines = None
    encoding = t2lutils.__get_file_encoding(args.lrc_file)
    with open(args.lrc_file, 'r', encoding=encoding) as fp:
        txt_lines = fp.readlines()
    lrc = process(txt_lines, args.music_file, demucs_model=args.model, demucs_idx=int(
        args.idx), format=args.format, line_only=args.line_only, vocalize=(args.vocalize == 1))
    print(lrc)


if __name__ == '__main__':
    cli()
