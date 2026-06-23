"""
Golden Set 构建器
从知识库采样文档，用 LLM 生成问答对，输出待人工 review 的草稿。
"""
import json
import asyncio
from pathlib import Path
from typing import List, Dict
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config


class GoldenSetBuilder:
    def __init__(self, num_questions: int = 2):
        self.num_questions = num_questions

    def sample_chunks_from_milvus(self, n: int = 50) -> List[Dict]:
        """从 Milvus 采样 n 条 chunk"""
        client = get_milvus_client()
        if not client:
            logger.error("Milvus 客户端不可用")
            return []

        try:
            results = client.query(
                collection_name=milvus_config.chunks_collection,
                filter="",
                output_fields=["chunk_id", "content", "file_title", "item_name"],
                limit=n
            )
            return results
        except Exception as e:
            logger.error(f"采样 chunks 失败: {e}")
            return []

    async def generate_qa_for_chunk(self, chunk: Dict, idx: int) -> List[Dict]:
        """为单个 chunk 生成 QA 对"""
        content = chunk.get("content", "")[:2000]
        title = chunk.get("file_title", "")
        item_name = chunk.get("item_name", "")

        prompt = load_prompt("golden_set_generate",
                             num_questions=self.num_questions,
                             context=content,
                             title=title,
                             item_name=item_name)

        try:
            llm = get_llm_client(json_mode=True)
            response = llm.invoke(prompt)
            raw = response.content.strip()

            # 清理 markdown 代码块
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]

            qa_list = json.loads(raw.strip())
            results = []
            for i, qa in enumerate(qa_list):
                results.append({
                    "case_id": f"GOLD-{idx:03d}-{i}",
                    "question": qa.get("question", ""),
                    "ground_truth": qa.get("ground_truth", ""),
                    "query_type": qa.get("query_type", "factual"),
                    "difficulty": qa.get("difficulty", "medium"),
                    "context": content,
                    "item_names": [item_name] if item_name else [],
                    "source_chunk_id": chunk.get("chunk_id", ""),
                })
            return results

        except Exception as e:
            logger.warning(f"生成 QA 失败 (chunk {idx}): {e}")
            return []

    async def build(self, target_size: int = 50, output_path: str = "data/golden_set_draft.jsonl") -> List[Dict]:
        """构建 Golden Set 草稿"""
        chunks = self.sample_chunks_from_milvus(target_size)
        if not chunks:
            logger.error("无可用的 chunks")
            return []

        all_cases = []
        for i, chunk in enumerate(chunks):
            cases = await self.generate_qa_for_chunk(chunk, i)
            all_cases.extend(cases)
            logger.info(f"进度: {i + 1}/{len(chunks)}, 累计 {len(all_cases)} 条")

        # 保存草稿
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            for case in all_cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")

        logger.info(f"Golden Set 草稿已保存: {output} ({len(all_cases)} 条)")
        return all_cases


async def main():
    from dotenv import load_dotenv
    load_dotenv()

    builder = GoldenSetBuilder(num_questions=2)
    cases = await builder.build(target_size=30)
    print(f"生成 {len(cases)} 条 QA 对")


if __name__ == "__main__":
    asyncio.run(main())