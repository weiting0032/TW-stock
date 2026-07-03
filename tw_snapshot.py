"""
tw_snapshot.py -- 本機 DB 快照 <-> GitHub db-snapshot 孤兒分支橋接

架構：本機是資料生產者（每日盤後 sync_and_upload.py 上傳快照）；
Streamlit Cloud 容器磁碟是暫時性的且 TPEx 擋雲端 IP，故雲端啟動時
從公開 raw 網址下載快照當基底，再由 app 的 lazy sync 自行補 TWSE 最新日。

推送用 git plumbing（hash-object / commit-tree）產生單一無父 commit
force-push 到 db-snapshot 分支：不碰 main、不動工作區與 index、不累積歷史。
"""

import gzip
import os
import sqlite3
import subprocess
import tempfile
from datetime import date, datetime
from pathlib import Path

import requests

from tw_db import DB_PATH

REPO_DIR = Path(__file__).parent
SNAPSHOT_BRANCH = "db-snapshot"
SNAPSHOT_NAME = "tw_market.db.gz"
RAW_URL = (
    "https://raw.githubusercontent.com/weiting0032/TW-stock/"
    f"{SNAPSHOT_BRANCH}/{SNAPSHOT_NAME}"
)
# 下載後寫入的標記檔：有它代表「這份 DB 來自快照」（= 雲端消費端），
# 才允許之後的過期重抓；本機生產端沒有標記，永遠不會被覆蓋。
MARKER_PATH = str(DB_PATH) + ".from_snapshot"


# ── 生產端（本機）────────────────────────────────────────────────────────────

def make_snapshot() -> str:
    """VACUUM INTO 取得一致性快照（WAL 下安全）再 gzip，回傳 .gz 路徑。"""
    tmp_dir = tempfile.gettempdir()
    tmp_db = os.path.join(tmp_dir, "tw_market_snapshot.db")
    tmp_gz = os.path.join(tmp_dir, SNAPSHOT_NAME)
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute(f"VACUUM INTO '{tmp_db}'")
    finally:
        conn.close()
    with open(tmp_db, "rb") as fin, gzip.open(tmp_gz, "wb", compresslevel=6) as fout:
        while True:
            chunk = fin.read(1 << 20)
            if not chunk:
                break
            fout.write(chunk)
    os.remove(tmp_db)
    return tmp_gz


def push_snapshot(gz_path: str) -> str:
    """把 gz 檔做成孤兒 commit，force-push 到 db-snapshot 分支。"""
    env = os.environ.copy()
    tmp_index = os.path.join(tempfile.gettempdir(), "twstock_snapshot_index")
    env["GIT_INDEX_FILE"] = tmp_index
    if os.path.exists(tmp_index):
        os.remove(tmp_index)

    def git(*args) -> str:
        r = subprocess.run(
            ["git", *args], cwd=str(REPO_DIR), env=env,
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args[:2])} failed: {r.stderr.strip()}")
        return r.stdout.strip()

    blob = git("hash-object", "-w", gz_path)
    git("update-index", "--add", "--cacheinfo", f"100644,{blob},{SNAPSHOT_NAME}")
    tree = git("write-tree")
    commit = git(
        "commit-tree", tree, "-m",
        f"db snapshot {datetime.now():%Y-%m-%d %H:%M}",
    )
    git("push", "--force", "origin", f"{commit}:refs/heads/{SNAPSHOT_BRANCH}")
    os.remove(tmp_index)
    size_mb = os.path.getsize(gz_path) / 1048576
    return f"pushed {SNAPSHOT_NAME} ({size_mb:.1f} MB) -> {SNAPSHOT_BRANCH} ({commit[:8]})"


def make_and_push_snapshot() -> str:
    gz = make_snapshot()
    try:
        return push_snapshot(gz)
    finally:
        if os.path.exists(gz):
            os.remove(gz)


# ── 消費端（Streamlit Cloud）────────────────────────────────────────────────

def _db_state():
    """回傳 (inst 列數, 最新 trade_date)；讀不到視為空。"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        row = conn.execute(
            "SELECT COUNT(*), MAX(trade_date) FROM inst_flow"
        ).fetchone()
        conn.close()
        return (row[0] or 0), row[1]
    except Exception:
        return 0, None


def download_snapshot_if_needed(max_stale_days: int = 4):
    """
    DB 缺失/空 → 下載快照；DB 來自快照（有標記檔）且最新日落後超過
    max_stale_days 天 → 重新下載。本機生產端（無標記檔）永不覆蓋。
    回傳 'downloaded' / None（不需要）/ 錯誤訊息字串。絕不 raise。
    """
    try:
        n_rows, latest = _db_state()
        need = n_rows == 0
        if not need and os.path.exists(MARKER_PATH) and latest:
            age = (date.today() - date.fromisoformat(latest)).days
            need = age > max_stale_days
        if not need:
            return None

        r = requests.get(RAW_URL, timeout=180)
        if r.status_code != 200:
            return f"快照下載失敗 HTTP {r.status_code}"
        raw = gzip.decompress(r.content)

        DB_PATH.parent.mkdir(exist_ok=True)
        tmp = str(DB_PATH) + ".tmp"
        with open(tmp, "wb") as f:
            f.write(raw)
        # 完整性檢查後才替換
        chk = sqlite3.connect(tmp)
        n_new = chk.execute("SELECT COUNT(*) FROM inst_flow").fetchone()[0]
        chk.close()
        if n_new == 0:
            os.remove(tmp)
            return "快照內容為空，略過"
        for suf in ("", "-wal", "-shm"):
            try:
                os.remove(str(DB_PATH) + suf)
            except OSError:
                pass
        os.replace(tmp, str(DB_PATH))
        with open(MARKER_PATH, "w") as f:
            f.write(datetime.now().isoformat())
        return "downloaded"
    except Exception as e:
        return f"快照檢查失敗：{e}"
