# 贵金属每日趋势自动报告

这是一个本地运行的 Python 程序，每天 08:00（Asia/Shanghai）从 Yahoo Finance 抓取贵金属、铜铝和美元/美债数据，计算技术指标，结合新闻关键词，生成中文 HTML + Markdown 趋势报告，并输出短线、中线、长线买卖/持有信号。

## 免责声明

本程序输出的所有买卖信号仅供技术研究，不构成投资建议。市场有风险，决策需独立判断。

## 环境要求

- Windows 10/11
- Python 3.12（推荐 3.12.10，最后一个提供 Windows 安装包的 3.12 版本）

## 安装

```powershell
cd "项目所在目录"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果系统没有 `python` 命令，使用安装路径，例如：

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
```

## 手动运行

联网生成当天报告：

```powershell
.\.venv\Scripts\python.exe -m pmreport --config config.yaml --verbose
```

使用本地已有数据、不联网：

```powershell
.\.venv\Scripts\python.exe -m pmreport --config config.yaml --date 2026-08-19 --no-fetch
```

报告输出到 `reports/<报告日期>/`。

## 定时任务

以当前 Windows 用户注册每天 08:00 的计划任务：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_task.ps1
```

日志写入 `logs\daily.log`。

## 目录结构

```text
pmreport/       程序包
config.yaml     标的、权重、阈值和路径配置
data/           SQLite 本地行情与新闻缓存
reports/        每日 HTML/Markdown 报告和图表
logs/           运行日志
scripts/        计划任务注册脚本
tests/          测试
```

## 配置

`config.yaml` 中可调整品种、三周期因子权重、信号阈值和风险覆盖规则。默认标的为：

- 贵金属：`GC=F`、`SI=F`、`PL=F`、`PA=F`
- 工业金属：`HG=F`、`ALI=F`
- 宏观：`DX-Y.NYB`、`^TNX`、`^TYX`

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## AlphaGPT 风格因子挖掘

本项目参考 AlphaGPT 的“特征 + 算子公式 + StackVM + 历史回测评分”思路，实现了本地因子引擎：

- 基础特征包括收益、成交量、买卖压力、动量、波动率、RSI、价格位置等 14 个因子。
- 算子包括加减乘除、取负、绝对值、符号、GATE、JUMP、DECAY、DELAY1、MAX3。
- 用公式搜索生成候选因子，并以历史 IC 和多空收益为 fitness 评分。
- 输出 `data/best_factors.json`，每日报告会读取该文件并显示“因子倾向”列。

重新挖掘因子：

```powershell
.\.venv\Scripts\python.exe -m pmreport.factor_miner --config config.yaml --db data\market.sqlite --output data\best_factors.json
```
