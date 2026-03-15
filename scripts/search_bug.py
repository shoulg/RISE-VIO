#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path

def main():
    # 让用户输入文件路径
    file_path = input("请输入要检查的日志文件路径: ").strip()
    path = Path(file_path)
    if not path.exists():
        print(f"文件不存在: {file_path}")
        return

    # 匹配类似 "... 123 ms" 的记录
    pattern = re.compile(r'(\d+(?:\.\d+)?)\s*ms')

    over_lines = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for lineno, line in enumerate(f, start=1):
            match = pattern.search(line)
            if match:
                value = float(match.group(1))
                if value > 100:
                    over_lines.append((lineno, value, line.strip()))

    # 输出结果
    if over_lines:
        print(f"\n在文件 {path.name} 中发现 {len(over_lines)} 条超过 100ms 的记录：\n")
        for lineno, value, content in over_lines:
            print(f"第 {lineno} 行: {value} ms | {content}")
    else:
        print("未发现超过 100ms 的记录。")

if __name__ == "__main__":
    main()
