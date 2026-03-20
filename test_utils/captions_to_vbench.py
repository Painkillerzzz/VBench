#!/usr/bin/env python3
"""
将 captions.json（数组格式，每项含 video_path 与 caption）
整理成 VBench 所需的 {"video_path": prompt, ...} 格式，并保存为 JSON 文件。
可复用：支持通过命令行参数指定输入/输出路径。
"""

import argparse
import json
from pathlib import Path


def convert_captions_to_vbench(input_path: str, output_path: str) -> None:
    """
    读取 captions 数组 JSON，转换为 {video_path: prompt} 字典并写入文件。

    Args:
        input_path: 输入的 captions.json 路径（数组格式）
        output_path: 输出的 captions_vbench.json 路径（对象格式）
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入 JSON 应为对象数组，每项含 'video_path' 与 'caption'")

    result = {}
    for item in data:
        if not isinstance(item, dict) or "video_path" not in item or "caption" not in item:
            raise ValueError(
                f"每项需为含 'video_path' 与 'caption' 的对象，当前项: {item!r}"
            )
        result[item["video_path"]] = item["caption"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"已转换 {len(result)} 条记录，保存至: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 captions 数组 JSON 转为 VBench 格式的 {video_path: prompt} JSON"
    )
    parser.add_argument(
        "-i",
        "--input",
        default="/workspace/VBench/test_videos/captions.json",
        help="输入的 captions.json 路径（默认: test_videos/captions.json）",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="/workspace/VBench/test_videos/captions_vbench.json",
        help="输出的 captions_vbench.json 路径（默认: test_videos/captions_vbench.json）",
    )
    args = parser.parse_args()

    convert_captions_to_vbench(args.input, args.output)


if __name__ == "__main__":
    main()
