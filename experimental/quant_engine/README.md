# BBBQuantEngine 動態算牌實驗版 V2

本引擎已透過 quant_dynamic_adapter.js 接入主線。完整參數、未校準限制與回滾見根目錄 QUANT_DYNAMIC_RELEASE.md。

執行 node test_quant_dynamic.cjs 或 node experimental/quant_engine/test_engine.cjs。

新128D不注入既有Frozen Core；B/P/T扣牌與TC僅為啟發式估計，無勝率保證。邊注未校準則觀望。
