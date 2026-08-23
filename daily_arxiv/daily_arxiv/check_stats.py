#!/usr/bin/env python3
"""
检查Scrapy爬取统计信息的脚本 / Script to check Scrapy crawling statistics
用于获取去重检查的状态结果 / Used to get deduplication check status results

功能说明 / Features:
- 检查当日与昨日论文数据的重复情况 / Check duplication between today's and yesterday's paper data
- 删除重复论文条目，保留新内容 / Remove duplicate papers, keep new content
- 根据去重后的结果决定工作流是否继续 / Decide workflow continuation based on deduplication results
"""
import json
import sys
import os
import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

try:
    from .source_merge import history_keys
except ImportError:  # Script execution from the daily_arxiv directory.
    from source_merge import history_keys


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"


def resolve_run_date(run_date=None):
    value = run_date or os.environ.get("RUN_DATE")
    if value:
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"Invalid run date: {value}") from exc
    return datetime.now(timezone.utc).date()

def load_papers_data(file_path):
    """
    从jsonl文件中加载完整的论文数据
    Load complete paper data from jsonl file
    
    Args:
        file_path (str): JSONL文件路径 / JSONL file path
        
    Returns:
        list: 论文数据列表 / List of paper data
        set: 论文ID集合 / Set of paper IDs
    """
    if not os.path.exists(file_path):
        return [], set()
    
    papers = []
    ids = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    papers.append(data)
                    ids.update(history_keys(data))
        return papers, ids
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return [], set()

def save_papers_data(papers, file_path):
    """
    保存论文数据到jsonl文件
    Save paper data to jsonl file
    
    Args:
        papers (list): 论文数据列表 / List of paper data
        file_path (str): 文件路径 / File path
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for paper in papers:
                f.write(json.dumps(paper, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        print(f"Error saving {file_path}: {e}", file=sys.stderr)
        return False

def perform_deduplication(run_date=None, data_dir=None, history_dir=None):
    """
    执行多日去重：删除与历史多日重复的论文条目，保留新内容
    Perform deduplication over multiple past days
    
    Returns:
        str: 去重状态 / Deduplication status
             - "has_new_content": 有新内容 / Has new content
             - "no_new_content": 无新内容 / No new content  
             - "no_data": 无数据 / No data
             - "error": 处理错误 / Processing error
    """

    today_date = resolve_run_date(run_date)
    today = today_date.isoformat()
    data_root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    today_file = data_root / f"{today}.jsonl"
    history_days = 7  # 向前追溯几天的数据进行对比

    if not os.path.exists(today_file):
        print("今日数据文件不存在 / Today's data file does not exist", file=sys.stderr)
        return "no_data"

    try:
        today_papers, today_ids = load_papers_data(today_file)
        print(f"今日论文总数: {len(today_papers)} / Today's total papers: {len(today_papers)}", file=sys.stderr)

        if not today_papers:
            return "no_data"

        # history_dir is normally an extracted snapshot of origin/data.  It is
        # deliberately separate from the main worktree so checking history
        # cannot overwrite today's newly generated working file.
        if history_dir is None:
            history_root = data_root
            history_available = True
        else:
            history_root = Path(history_dir)
            history_available = history_root.is_dir()
            if not history_available:
                print(
                    f"WARN: data branch history directory is unavailable: {history_root}",
                    file=sys.stderr,
                )

        # 收集历史多日 canonical keys。除了 id，还包含 arXiv、DOI、bibcode
        # 和 title+first-author+year，确保跨源 canonical id 改变时仍能去重。
        history_ids = set()
        for i in range(1, history_days + 1):
            date_str = (today_date - timedelta(days=i)).isoformat()
            history_file = history_root / f"{date_str}.jsonl"
            _, past_ids = load_papers_data(history_file)
            history_ids.update(past_ids)

        print(
            f"历史{history_days}日去重库大小: {len(history_ids)} / "
            f"History {history_days} days deduplication library size: {len(history_ids)}",
            file=sys.stderr,
        )

        duplicate_indexes = [
            index
            for index, paper in enumerate(today_papers)
            if history_keys(paper).intersection(history_ids)
        ]

        if duplicate_indexes:
            print(
                f"发现 {len(duplicate_indexes)} 篇历史重复论文 / "
                f"Found {len(duplicate_indexes)} historical duplicate papers",
                file=sys.stderr,
            )
            duplicate_index_set = set(duplicate_indexes)
            new_papers = [
                paper
                for index, paper in enumerate(today_papers)
                if index not in duplicate_index_set
            ]

            print(f"去重后剩余论文数: {len(new_papers)} / Remaining papers after deduplication: {len(new_papers)}", file=sys.stderr)

            if new_papers:
                if save_papers_data(new_papers, today_file):
                    print(f"已更新今日文件，移除 {len(duplicate_indexes)} 篇重复论文 / Today's file updated, removed {len(duplicate_indexes)} duplicate papers", file=sys.stderr)
                    return "has_new_content"
                else:
                    print("保存去重后的数据失败 / Failed to save deduplicated data", file=sys.stderr)
                    return "error"
            else:
                try:
                    os.remove(today_file)
                    print("所有论文均为重复内容，已删除今日文件 / All papers are duplicate content, today's file deleted", file=sys.stderr)
                except Exception as e:
                    print(f"删除文件失败: {e} / Failed to delete file: {e}", file=sys.stderr)
                return "no_new_content"
        else:
            print("所有内容均为新内容 / All content is new", file=sys.stderr)
            return "has_new_content"

    except Exception as e:
        print(f"去重处理失败: {e} / Deduplication processing failed: {e}", file=sys.stderr)
        return "error"

def main():
    """
    检查去重状态并返回相应的退出码
    Check deduplication status and return corresponding exit code
    
    退出码含义 / Exit code meanings:
    0: 有新内容，继续处理 / Has new content, continue processing
    1: 无新内容，停止工作流 / No new content, stop workflow
    2: 处理错误 / Processing error
    """
    
    parser = argparse.ArgumentParser(description="Deduplicate one explicit UTC run date")
    parser.add_argument("--date", dest="run_date", help="UTC run date: YYYY-MM-DD")
    parser.add_argument("--data-dir", help="Working data directory")
    parser.add_argument(
        "--history-dir",
        help="Extracted data-branch data directory; never use the main worktree for this snapshot",
    )
    args = parser.parse_args()

    print("正在执行去重检查... / Performing intelligent deduplication check...", file=sys.stderr)
    
    # 执行去重处理 / Perform deduplication processing
    try:
        dedup_status = perform_deduplication(
            run_date=args.run_date,
            data_dir=args.data_dir,
            history_dir=args.history_dir,
        )
    except ValueError as exc:
        print(f"去重日期参数错误: {exc} / Invalid deduplication date", file=sys.stderr)
        dedup_status = "error"
    
    if dedup_status == "has_new_content":
        print("✅ 去重完成，发现新内容，继续工作流 / Deduplication completed, new content found, continue workflow", file=sys.stderr)
        sys.exit(0)
    elif dedup_status == "no_new_content":
        print("⏹️ 去重完成，无新内容，停止工作流 / Deduplication completed, no new content, stop workflow", file=sys.stderr)
        sys.exit(1)
    elif dedup_status == "no_data":
        print("⏹️ 今日无数据，停止工作流 / No data today, stop workflow", file=sys.stderr)
        sys.exit(1)
    elif dedup_status == "error":
        print("❌ 去重处理出错，停止工作流 / Deduplication processing error, stop workflow", file=sys.stderr)
        sys.exit(2)
    else:
        # 意外情况：未知状态 / Unexpected case: unknown status
        print("❌ 未知去重状态，停止工作流 / Unknown deduplication status, stop workflow", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
