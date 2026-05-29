"""生成 v2 合同：基于 outputs/parsed_document.json 做精确文本替换。

模拟"合同 v1 → v2"的常见改动场景，便于 DiffAgent 演示对比能力：
1. 总金额 128 万 → 158 万（金额变更）
2. 付款比例 30/40/30 → 20/50/30（付款条件调整）
3. 交付期限 45 自然日 → 60 自然日（工期延长）
4. 乙方违约金累计上限 20% → 15%（违约责任放宽）
5. 增加一条新条款 1.7 知识产权（条款新增）
6. 删除原 1.8 违约责任 中的"甲方逾期付款"那一段（条款删除）

改完保存到 outputs/parsed_document_v2.json，DiffAgent 直接读它。
"""

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs" / "parsed_document.json"
DST = ROOT / "outputs" / "parsed_document_v2.json"

if not SRC.exists():
    raise SystemExit(
        f"未找到 {SRC}，请先跑 `python3 src/main.py` 生成基线 parsed_document.json"
    )

doc = json.loads(SRC.read_text(encoding="utf-8"))
blocks = doc["blocks"]

# ---- 替换规则：原文 → 新文（精确替换，仅命中给定 block） ----
PATCHES = [
    # 1. 总金额 1,280,000 → 1,580,000；大写 壹佰贰拾万 → 壹佰伍拾捌万
    {
        "block_index": 13,
        "find": "本合同总价为人民币 1,280,000.00 元，价格已包含软件授权、硬件设备、实施部署、培训、首年维保和项目管理费用。大写金额：人民币壹佰贰拾万元整。",
        "replace": "本合同总价为人民币 1,580,000.00 元，价格已包含软件授权、硬件设备、实施部署、培训、首年维保、项目管理费用以及第二年运维服务。大写金额：人民币壹佰伍拾捌万元整。",
    },
    # 2. 第一期 30% → 20%
    {
        "block_index": 29,
        "find": "甲方向乙方支付合同总价的 30%。",
        "replace": "甲方向乙方支付合同总价的 20%。",
    },
    # 3. 第二期 40% → 50%
    {
        "block_index": 30,
        "find": "甲方向乙方支付合同总价的 40%。",
        "replace": "甲方向乙方支付合同总价的 50%。",
    },
    # 4. 交付期限 45 → 60 自然日
    {
        "block_index": 26,
        "find": "乙方应自合同生效之日起 45 个自然日内完成",
        "replace": "乙方应自合同生效之日起 60 个自然日内完成",
    },
    # 5. 乙方违约金上限 20% → 15%
    {
        "block_index": 38,
        "find": "违约金累计最高不超过合同总价的 20%。",
        "replace": "违约金累计最高不超过合同总价的 15%。",
    },
]

# 应用替换
patched = 0
for p in PATCHES:
    b = blocks[p["block_index"]]
    if p["find"] not in b["content"]:
        print(f"[警告] block#{p['block_index']} 未匹配，跳过：{p['find'][:60]}…")
        continue
    b["content"] = b["content"].replace(p["find"], p["replace"])
    patched += 1
    print(f"[替换] block#{p['block_index']}：{p['find'][:50]}… → {p['replace'][:50]}…")

# 6. 删除"甲方逾期付款"段（block#40）
to_delete = []
for i, b in enumerate(blocks):
    if "甲方逾期付款的，每逾期一日按逾期金额的 0.02%" in b["content"]:
        to_delete.append(i)
for i in reversed(to_delete):
    print(f"[删除] block#{i}：{blocks[i]['content'][:60]}…")
    del blocks[i]

# 7. 在 1.6 付款方式（block#28）后插入新条款 1.7 知识产权
new_blocks = [
    {
        "block_id": "v2_added_b1",
        "block_type": "section_title",
        "section_path": "智能印章与合同审查平台建设项目合同 > 1.7 知识产权",
        "page": 4,
        "content": "1.7 知识产权",
        "metadata": {"level": 2},
        "confidence": 1.0,
        "needs_review": False,
    },
    {
        "block_id": "v2_added_b2",
        "block_type": "paragraph",
        "section_path": "智能印章与合同审查平台建设项目合同 > 1.7 知识产权",
        "page": 4,
        "content": "本项目交付的所有定制开发代码、文档与配置，知识产权归甲方所有。乙方对本项目所使用的通用模块、底层框架与第三方组件保留原始著作权。乙方在本项目交付前应向甲方明确披露所使用的开源软件清单及对应许可证要求。",
        "metadata": {},
        "confidence": 1.0,
        "needs_review": False,
    },
]

# 找到 1.8 违约责任 的位置（之前是 block#37，但删除了 1 个 block 后位置可能变）
insert_idx = None
for i, b in enumerate(blocks):
    if b["content"].strip() == "1.8 违约责任":
        insert_idx = i
        break
if insert_idx is None:
    # 找不到就插到原 block#37 位置
    insert_idx = 37
    print("[警告] 未找到 1.8 违约责任 标题，按原位置插入")
blocks[insert_idx:insert_idx] = new_blocks
print(f"[新增] 在 block#{insert_idx} 处插入 1.7 知识产权 章节（2 个新 block）")

# 保存
doc["filename"] = "AI知识库-综合测试文档_v2.pdf"
DST.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n生成 v2：{DST}（共 {len(blocks)} 个 block）")
print(f"改动汇总：")
print(f"  - 替换 {patched} 处")
print(f"  - 删除 {len(to_delete)} 个 block")
print(f"  - 新增 {len(new_blocks)} 个 block")
