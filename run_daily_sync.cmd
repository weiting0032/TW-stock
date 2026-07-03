@echo off
rem TW-stock 每日盤後同步+快照上傳（工作排程器入口）
cd /d "C:\Users\11106648\Desktop\claude project\Leisure\TW-stock"
python -X utf8 sync_and_upload.py >> "data\sync.log" 2>&1
