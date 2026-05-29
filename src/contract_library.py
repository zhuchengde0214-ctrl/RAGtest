"""ContractLibrary：管理多份合同的长期记忆。

设计：
- 每份合同用 PDF 内容 MD5 前 12 位作为 contract_id
- 每份合同独立目录 outputs/contracts/<id>/ 存放 parsed_document.json / chunks.json
  / qa_results.json / review_results.json / metadata.json
- ChromaDB 用 collection name = "contract_<id>" 持久化向量
- 主索引 outputs/contracts_index.json 记录所有合同的元信息（文件名/别名/上传时间）

API：
    lib = ContractLibrary()
    info = lib.add_pdf("path/to/contract.pdf")          # 上传，返回 ContractInfo
    info = lib.get(contract_id)                          # 取一份
    items = lib.list()                                   # 全部，按 last_accessed 排序
    lib.rename(contract_id, "海岳合同 v1")
    lib.delete(contract_id)
    lib.touch(contract_id)                               # 更新 last_accessed
    paths = lib.paths(contract_id)                       # 该合同的所有文件路径
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# 默认根目录
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LIB_ROOT = ROOT / "outputs" / "contracts"
DEFAULT_INDEX_PATH = ROOT / "outputs" / "contracts_index.json"
DEFAULT_UPLOAD_DIR = ROOT / "data" / "uploads"
DEFAULT_CHROMA_ROOT = ROOT / "outputs" / "chroma_db"


@dataclass
class ContractInfo:
    id: str                          # contract_<hash12>
    pdf_path: str                    # 原始 PDF（永久保留）
    original_filename: str           # 用户上传时的原始文件名
    alias: str                       # 用户可改的别名（默认 = original_filename）
    role: str = "primary"            # primary | v2（仅用于 diff 场景的标记）
    uploaded_at: str = ""
    last_accessed: str = ""
    pages: int = 0
    chunks: int = 0
    status: str = "uploaded"         # uploaded | parsed | indexed | failed
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ContractInfo":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ContractLibrary:
    def __init__(
        self,
        lib_root: Path = DEFAULT_LIB_ROOT,
        index_path: Path = DEFAULT_INDEX_PATH,
        upload_dir: Path = DEFAULT_UPLOAD_DIR,
        chroma_root: Path = DEFAULT_CHROMA_ROOT,
    ):
        self.lib_root = Path(lib_root)
        self.index_path = Path(index_path)
        self.upload_dir = Path(upload_dir)
        self.chroma_root = Path(chroma_root)
        self.lib_root.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_root.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, ContractInfo] = self._load_index()

    # ------------------------------------------------------------------
    # 索引加载/保存
    # ------------------------------------------------------------------
    def _load_index(self) -> dict[str, ContractInfo]:
        if not self.index_path.exists():
            return {}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            return {k: ContractInfo.from_dict(v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"contracts_index.json 加载失败：{e}，按空库初始化")
            return {}

    def _save_index(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.to_dict() for k, v in self._index.items()}
        self.index_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # 路径辅助
    # ------------------------------------------------------------------
    def paths(self, contract_id: str) -> dict[str, Path]:
        base = self.lib_root / contract_id
        return {
            "base": base,
            "parsed_document": base / "parsed_document.json",
            "chunks": base / "chunks.json",
            "qa_results": base / "qa_results.json",
            "review_results": base / "review_results.json",
            "diff_results": base / "diff_results.json",
            "conversation": base / "conversation.jsonl",
            "metadata": base / "metadata.json",
            "ocr_cache": base / "ocr_cache",
            "chroma_collection_name": f"contract_{contract_id.replace('-', '_')}"[:60],
        }

    @staticmethod
    def _compute_id(pdf_bytes: bytes) -> str:
        h = hashlib.md5(pdf_bytes).hexdigest()
        return f"contract_{h[:12]}"

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add_pdf(
        self,
        pdf_source: str | bytes,
        original_filename: Optional[str] = None,
        alias: Optional[str] = None,
        role: str = "primary",
    ) -> ContractInfo:
        """上传一份 PDF。
        pdf_source: 路径字符串或 bytes
        返回 ContractInfo（如已存在同 hash 的合同，返回已有那份并仅更新 last_accessed）
        """
        if isinstance(pdf_source, bytes):
            pdf_bytes = pdf_source
            if not original_filename:
                raise ValueError("使用 bytes 上传时必须提供 original_filename")
        else:
            with open(pdf_source, "rb") as f:
                pdf_bytes = f.read()
            original_filename = original_filename or os.path.basename(pdf_source)

        contract_id = self._compute_id(pdf_bytes)
        now = datetime.now().isoformat(timespec="seconds")

        # 已存在 → 更新 last_accessed 直接返回
        if contract_id in self._index:
            info = self._index[contract_id]
            info.last_accessed = now
            self._save_index()
            logger.info(f"合同已存在：{contract_id}（{info.alias}）")
            return info

        # 新建：保存 PDF 到 data/uploads/
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_name = self._safe_filename(original_filename)
        pdf_dst = self.upload_dir / f"{ts}_{contract_id}_{safe_name}"
        with open(pdf_dst, "wb") as f:
            f.write(pdf_bytes)

        # 创建合同目录
        paths = self.paths(contract_id)
        paths["base"].mkdir(parents=True, exist_ok=True)

        info = ContractInfo(
            id=contract_id,
            pdf_path=str(pdf_dst),
            original_filename=original_filename,
            alias=alias or original_filename,
            role=role,
            uploaded_at=now,
            last_accessed=now,
            status="uploaded",
        )
        self._index[contract_id] = info
        self._save_index()
        self._save_metadata(info)
        logger.info(f"新合同注册：{contract_id}（{original_filename}）")
        return info

    def get(self, contract_id: str) -> Optional[ContractInfo]:
        return self._index.get(contract_id)

    def list(self) -> list[ContractInfo]:
        items = list(self._index.values())
        items.sort(key=lambda x: x.last_accessed or x.uploaded_at, reverse=True)
        return items

    def list_by_role(self, role: str) -> list[ContractInfo]:
        return [c for c in self.list() if c.role == role]

    def update(self, contract_id: str, **fields) -> Optional[ContractInfo]:
        info = self._index.get(contract_id)
        if info is None:
            return None
        for k, v in fields.items():
            if hasattr(info, k):
                setattr(info, k, v)
        self._save_index()
        self._save_metadata(info)
        return info

    def rename(self, contract_id: str, new_alias: str) -> Optional[ContractInfo]:
        return self.update(contract_id, alias=new_alias.strip() or contract_id)

    def touch(self, contract_id: str) -> None:
        info = self._index.get(contract_id)
        if info:
            info.last_accessed = datetime.now().isoformat(timespec="seconds")
            self._save_index()

    def delete(self, contract_id: str, drop_pdf: bool = False) -> bool:
        info = self._index.pop(contract_id, None)
        if info is None:
            return False
        # 清合同目录
        paths = self.paths(contract_id)
        if paths["base"].exists():
            shutil.rmtree(paths["base"], ignore_errors=True)
        # 清 chroma collection 持久化目录（PersistentClient 跨合同共享同一目录，
        # 仅删 collection 数据需要 client.delete_collection；此处保持简单：靠 retriever 重建时覆盖）
        if drop_pdf and info.pdf_path and os.path.exists(info.pdf_path):
            try:
                os.remove(info.pdf_path)
            except Exception:
                pass
        self._save_index()
        logger.info(f"已删除合同：{contract_id}")
        return True

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _save_metadata(self, info: ContractInfo) -> None:
        p = self.paths(info.id)["metadata"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(info.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _safe_filename(name: str) -> str:
        # 去掉路径分隔符，限制长度，保留中文
        name = os.path.basename(name)
        name = "".join(c for c in name if c not in '/\\:*?"<>|')
        return name[:120] or "contract.pdf"

    # ------------------------------------------------------------------
    # 状态查询便捷方法
    # ------------------------------------------------------------------
    def is_parsed(self, contract_id: str) -> bool:
        return self.paths(contract_id)["parsed_document"].exists()

    def is_indexed(self, contract_id: str) -> bool:
        return self.paths(contract_id)["chunks"].exists()
