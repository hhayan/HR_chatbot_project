# backend/test.py
"""
임베딩 모델 비교 평가 테스트
Step 1: Embedding Quality (numpy cosine similarity)
Step 2: Retrieval 성능 (Pinecone vector search)

모델 4종:
  - text-embedding-3-small  (baseline, OpenAI)
  - text-embedding-3-large  (high quality, OpenAI)
  - solar-embedding-1-large (한국어 특화, Upstage)
  - qwen3-embedding:0.6b    (instruction retrieval, Ollama)
"""

import json
import time
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
import re

# .env에서 환경변수 로드
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from db import _load_and_split_docs, EXISTING_HR_DOCS
from llm import get_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent / "data"
DATASET_PATH = DATA_DIR / "test_dataset.json"

# Pinecone 인덱스 — 모델별 1개씩, native dim 보존
INDEX_OPENAI_SMALL = "hr-test-openai-small"   # dim=1536
INDEX_OPENAI_LARGE = "hr-test-openai-large"   # dim=3072
INDEX_SOLAR        = "hr-test-solar"          # dim=4096
INDEX_QWEN         = "hr-test-qwen"           # dim=1024

# 모델 설정
MODELS = {
    "openai_small": {
        "type": "openai",
        "model": "text-embedding-3-small",
        "index": INDEX_OPENAI_SMALL,
        "dim": 1536,
    },
    "openai_large": {
        "type": "openai",
        "model": "text-embedding-3-large",
        "index": INDEX_OPENAI_LARGE,
        "dim": 3072,
    },
    "solar_large": {
        "type": "upstage",
        "model": "solar-embedding-1-large",
        "index": INDEX_SOLAR,
        "dim": 4096,
    },
    "qwen_0.6b": {
        "type": "ollama",
        "model": "qwen3-embedding:0.6b",
        "index": INDEX_QWEN,
        "dim": 1024,
    },
}
# ─────────────────────────────────────────────
# step1. Embedding Model test
# ─────────────────────────────────────────────
class EmbeddingFactory:
    """임베딩 모델 생성 팩토리"""

    @staticmethod
    def create(model_type: str, **kwargs):
        if model_type == "openai":
            from langchain_openai import OpenAIEmbeddings
            model = kwargs.get("model", "text-embedding-3-small")
            return OpenAIEmbeddings(model=model)

        elif model_type == "upstage":
            from langchain_upstage import UpstageEmbeddings
            model = kwargs.get("model", "solar-embedding-1-large")
            return UpstageEmbeddings(model=model)

        elif model_type == "ollama":
            from langchain_ollama import OllamaEmbeddings
            model = kwargs.get("model", "qwen3-embedding:0.6b")
            return OllamaEmbeddings(model=model)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

# ─────────────────────────────────────────────
# Step 1: Embedding Quality (numpy)
# ─────────────────────────────────────────────
def step1_embedding_quality(
    chunks: List[Document],
    dataset: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    각 모델별로 cosine similarity gap + embedding latency 측정 (numpy 로컬)

    임베딩 모델별 dimension
  - text-embedding-3-small: 1536
  - text-embedding-3-large: 3072  
  - solar-embedding-1-large: 4096
  - qwen3-embedding:0.6b: 1024    
    """
    results = {}

    for model_name, model_cfg in MODELS.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"[Step 1] {model_name} — embedding quality 평가")
        logger.info(f"{'='*50}")

        embedder = EmbeddingFactory.create(model_cfg["type"], model=model_cfg["model"])

        # ── 청크 임베딩 ──
        chunk_texts = [c.page_content for c in chunks]
        logger.info(f"  청크 {len(chunk_texts)}개 임베딩 생성 중...")
        chunk_vecs = np.array(embedder.embed_documents(chunk_texts))
        logger.info(f"  ✅ 임베딩 완료 — dim={chunk_vecs.shape[1]}")

        # ── 쿼리별 cosine similarity ──
        relevant_cosines = []
        irrelevant_cosines = []
        embed_latencies = []

        for item in dataset:
            query = item["query"]
            relevant_section = item["relevant_section"]

            # 임베딩 latency 측정
            t0 = time.time()
            query_vec = np.array(embedder.embed_query(query))
            embed_latencies.append((time.time() - t0) * 1000)  # ms

            # 각 청크와의 cosine similarity
            for i, chunk in enumerate(chunks):
                sim = cosine_similarity(query_vec, chunk_vecs[i])
                if is_relevant(chunk, relevant_section):
                    relevant_cosines.append(sim)
                else:
                    irrelevant_cosines.append(sim)

        avg_relevant = float(np.mean(relevant_cosines)) if relevant_cosines else 0.0
        avg_irrelevant = float(np.mean(irrelevant_cosines)) if irrelevant_cosines else 0.0
        gap = avg_relevant - avg_irrelevant
        avg_latency = float(np.mean(embed_latencies))

        results[model_name] = {
            "avg_relevant_cosine": round(avg_relevant, 4),
            "avg_irrelevant_cosine": round(avg_irrelevant, 4),
            "cosine_gap": round(gap, 4),
            "embedding_latency_ms": round(avg_latency, 2),
        }

        logger.info(f"  relevant cosine:   {avg_relevant:.4f}")
        logger.info(f"  irrelevant cosine: {avg_irrelevant:.4f}")
        logger.info(f"  gap:               {gap:.4f}")
        logger.info(f"  avg latency:       {avg_latency:.2f} ms")

    return results

# ─────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────
def load_test_dataset() -> List[Dict[str, Any]]:
    """골든셋 JSON 로드"""
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """두 벡터의 코사인 유사도"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def is_relevant(doc: Document, relevant_section: str) -> bool:
    """검색된 문서가 정답 섹션에 해당하는지 판단"""
    meta = doc.metadata
    content = doc.page_content

    # 메타데이터의 sub_category / main_category에서 매칭
    section_lower = relevant_section.lower()
    for key in ["sub_category", "main_category", "doc_title"]:
        val = meta.get(key, "")
        if val and section_lower in val.lower():
            return True

    # page_content에 섹션 키워드 포함 여부 (fallback)
    if relevant_section in content:
        return True

    return False

# ─────────────────────────────────────────────
# Pinecone 모델별 인덱스 생성
# ─────────────────────────────────────────────
def ensure_pinecone_index(pc: Pinecone, name: str, dimension: int):
    """인덱스가 없으면 생성"""
    existing = [idx.name for idx in pc.list_indexes()]
    if name in existing:
        logger.info(f"Pinecone 인덱스 '{name}' 이미 존재.")
        return

    logger.info(f"Pinecone 인덱스 '{name}' 생성 (dim={dimension})...")
    pc.create_index(
        name=name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # 준비 대기
    for _ in range(30):
        status = pc.describe_index(name).status
        if status and status.get("ready"):
            logger.info(f"인덱스 '{name}' 준비 완료.")
            return
        time.sleep(2)
    logger.warning(f"인덱스 '{name}' 시간 내 준비 안됨.")


def clear_index(pc: Pinecone, index_name: str):
    """인덱스 내 모든 벡터 삭제"""
    index = pc.Index(index_name)
    try:
        index.delete(delete_all=True)
        logger.info(f"인덱스 '{index_name}' 클리어 완료.")
    except Exception as e:
        logger.warning(f"인덱스 클리어 실패 (비어있을 수 있음): {e}")


# ─────────────────────────────────────────────
# Step 2: Retrieval 성능 (Pinecone)
# ─────────────────────────────────────────────
def step2_retrieval_eval(
    chunks: List[Document],
    dataset: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Pinecone 검색으로 Hit@K, MRR, retrieval latency 측정
    모델별 인덱스 4개 (native dim 보존)
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # 모델별 인덱스 생성
    for model_cfg in MODELS.values():
        ensure_pinecone_index(pc, model_cfg["index"], model_cfg["dim"])

    results = {}

    for model_name, model_cfg in MODELS.items():
        logger.info(f"[Step 2] {model_name} — retrieval 성능 평가")

        embedder = EmbeddingFactory.create(model_cfg["type"], model=model_cfg["model"])
        index_name = model_cfg["index"]

        # ── 인덱스 클리어 & 업로드 ──
        clear_index(pc, index_name)
        time.sleep(1)

        index = pc.Index(index_name)

        # 청크 임베딩 생성 (native dim 그대로)
        chunk_texts = [c.page_content for c in chunks]
        logger.info(f"  청크 {len(chunk_texts)}개 임베딩 생성 중...")
        vecs = embedder.embed_documents(chunk_texts)
        logger.info(f"  ✅ 임베딩 완료 — dim={len(vecs[0])}")

        # Pinecone 업로드
        vectors_to_upsert = []
        for i, (vec, chunk) in enumerate(zip(vecs, chunks)):
            vectors_to_upsert.append({
                "id": f"{model_name}_{i}",
                "values": vec,
                "metadata": {
                    "text": chunk.page_content[:1000],
                    "source": chunk.metadata.get("source", ""),
                    "main_category": chunk.metadata.get("main_category", ""),
                    "sub_category": chunk.metadata.get("sub_category", ""),
                },
            })

        # 배치 업로드
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch)
        logger.info(f"  {len(vectors_to_upsert)}개 벡터 Pinecone 업로드 완료.")

        # 업로드 반영 대기
        time.sleep(3)

        # ── 쿼리 검색 & 평가 ──
        hit_at_1 = 0
        hit_at_3 = 0
        hit_at_5 = 0
        reciprocal_ranks = []
        retrieval_latencies = []
        embedding_latencies = []

        for item in dataset:
            query = item["query"]
            relevant_section = item["relevant_section"]

            # 쿼리 임베딩 (native dim 그대로)
            t0 = time.time()
            qvec = embedder.embed_query(query)
            t_embed = (time.time() - t0) * 1000
            embedding_latencies.append(t_embed)

            # Pinecone 검색
            t0 = time.time()
            search_results = index.query(
                vector=qvec,
                top_k=3,
                include_metadata=True,
            )
            t_retrieve = (time.time() - t0) * 1000
            retrieval_latencies.append(t_retrieve)

            # 결과 평가
            matches = search_results.get("matches", [])
            found_rank = None

            for rank, match in enumerate(matches, 1):
                meta = match.get("metadata", {})
                # 정답 섹션 매칭
                section_lower = relevant_section.lower()
                is_match = False
                for key in ["sub_category", "main_category"]:
                    val = meta.get(key, "")
                    if val and section_lower in val.lower():
                        is_match = True
                        break
                if not is_match:
                    text = meta.get("text", "")
                    if relevant_section in text:
                        is_match = True

                if is_match and found_rank is None:
                    found_rank = rank

            if found_rank is not None:
                if found_rank <= 1:
                    hit_at_1 += 1
                if found_rank <= 3:
                    hit_at_3 += 1
                if found_rank <= 5:
                    hit_at_5 += 1
                reciprocal_ranks.append(1.0 / found_rank)
            else:
                reciprocal_ranks.append(0.0)

        n = len(dataset)
        avg_embed_lat = float(np.mean(embedding_latencies))
        avg_retrieve_lat = float(np.mean(retrieval_latencies))

        results[model_name] = {
            "hit_at_1": round(hit_at_1 / n, 4),
            "hit_at_3": round(hit_at_3 / n, 4),
            "hit_at_5": round(hit_at_5 / n, 4),
            "mrr": round(float(np.mean(reciprocal_ranks)), 4),
            "embedding_latency_ms": round(avg_embed_lat, 2),
            "retrieval_latency_ms": round(avg_retrieve_lat, 2),
            "total_latency_ms": round(avg_embed_lat + avg_retrieve_lat, 2),
        }

        logger.info(f"  Hit@1: {hit_at_1/n:.4f}  Hit@3: {hit_at_3/n:.4f}  Hit@5: {hit_at_5/n:.4f}")
        logger.info(f"  MRR:   {float(np.mean(reciprocal_ranks)):.4f}")
        logger.info(f"  Latency — embed: {avg_embed_lat:.2f}ms  retrieve: {avg_retrieve_lat:.2f}ms")

    return results

# ─────────────────────────────────────────────
# Step 3: Reranker 성능 (text-embedding-3-small 전용)
# ─────────────────────────────────────────────
def step3_reranker_eval(
    chunks: List[Document],
    dataset: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Pinecone Top-20 검색 후 LLM Reranker를 사용해 Top-5 재정렬.
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    model_name = "openai_small"
    model_cfg = MODELS[model_name]
    
    logger.info(f"\n{'='*50}")
    logger.info(f"[Step 3] {model_name} + LLM Reranker 성능 평가")
    logger.info(f"{'='*50}")

    embedder = EmbeddingFactory.create(model_cfg["type"], model=model_cfg["model"])
    index = pc.Index(model_cfg["index"])
    llm = get_llm("gen")

    hit_at_1, hit_at_3, hit_at_5 = 0, 0, 0
    reciprocal_ranks = []
    
    embedding_latencies = []
    retrieval_latencies = []
    reranker_latencies = []

    # OOD 제외한 HR 규정 질문만 평가
    eval_dataset = [item for item in dataset if item.get("ground_truth") != "OOD"]
    n = len(eval_dataset)

    for item in eval_dataset:
        query = item["query"]
        relevant_section = item["relevant_section"]

        # 1. 쿼리 임베딩
        t0 = time.time()
        qvec = embedder.embed_query(query)
        embedding_latencies.append((time.time() - t0) * 1000)

        # 2. Pinecone 1차 검색 (top_k=20)
        t0 = time.time()
        search_results = index.query(
            vector=qvec,
            top_k=20,
            include_metadata=True,
        )
        retrieval_latencies.append((time.time() - t0) * 1000)

        matches = search_results.get("matches", [])
        retrieved_docs = [Document(page_content=m.get("metadata", {}).get("text", ""), metadata=m.get("metadata", {})) for m in matches]

        # 3. Reranker (LLM) 기반 재정렬
        t0 = time.time()
        scored: List[Tuple[Document, float]] = []
        for doc in retrieved_docs:
            prompt = f"질문: \"{query}\"\n문서 내용: \"{doc.page_content}\"\n0~1 사이 숫자로 관련도만 출력:"
            txt = (llm.invoke(prompt).content or "").strip()
            cleaned = txt.replace(",", ".")
            match_score = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
            score = float(match_score.group()) if match_score else 0.0
            score = max(0.0, min(1.0, score))
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in scored[:5]]
        reranker_latencies.append((time.time() - t0) * 1000)

        # 4. 평가
        found_rank = None
        for rank, doc in enumerate(top_docs, 1):
            meta = doc.metadata
            section_lower = relevant_section.lower()
            is_match = False
            for key in ["sub_category", "main_category"]:
                val = meta.get(key, "")
                if val and section_lower in val.lower():
                    is_match = True
                    break
            if not is_match and relevant_section in meta.get("text", ""):
                is_match = True

            if is_match and found_rank is None:
                found_rank = rank
                
        if found_rank is not None:
            if found_rank <= 1: hit_at_1 += 1
            if found_rank <= 3: hit_at_3 += 1
            if found_rank <= 5: hit_at_5 += 1
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

    avg_embed_lat = float(np.mean(embedding_latencies))
    avg_retrieve_lat = float(np.mean(retrieval_latencies))
    avg_reranker_lat = float(np.mean(reranker_latencies))

    results = {
        model_name: {
            "hit_at_1": round(hit_at_1 / n, 4),
            "hit_at_3": round(hit_at_3 / n, 4),
            "hit_at_5": round(hit_at_5 / n, 4),
            "mrr": round(float(np.mean(reciprocal_ranks)), 4),
            "embedding_latency_ms": round(avg_embed_lat, 2),
            "retrieval_latency_ms": round(avg_retrieve_lat, 2),
            "reranker_latency_ms": round(avg_reranker_lat, 2),
        }
    }

    logger.info(f"  Hit@1: {hit_at_1/n:.4f}  Hit@3: {hit_at_3/n:.4f}  Hit@5: {hit_at_5/n:.4f}")
    logger.info(f"  MRR:   {float(np.mean(reciprocal_ranks)):.4f}")
    logger.info(f"  Latency — embed: {avg_embed_lat:.2f}ms  retrieve: {avg_retrieve_lat:.2f}ms  rerank: {avg_reranker_lat:.2f}ms")
    
    return results

# ─────────────────────────────────────────────
# Step 4: 검증 루프 (Verification Loop) 및 LLM Latency
# ─────────────────────────────────────────────
def step4_verification_eval(
    chunks: List[Document],
    dataset: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """
    Top-5 청크 기반 LLM 답변 생성 후 Groundedness 검증 및 OOD 기각 성능 평가.
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    model_name = "openai_small"
    model_cfg = MODELS[model_name]
    
    logger.info(f"\n{'='*50}")
    logger.info(f"[Step 4] {model_name} + Verification Loop 성능 평가")
    logger.info(f"{'='*50}")

    embedder = EmbeddingFactory.create(model_cfg["type"], model=model_cfg["model"])
    index = pc.Index(model_cfg["index"])
    llm = get_llm("gen")

    in_domain_dataset = [item for item in dataset if item.get("ground_truth") != "OOD"]
    ood_dataset = [item for item in dataset if item.get("ground_truth") == "OOD"]

    hallucination_count = 0
    llm_latencies = []

    # 1. In-domain 평가 (Hallucination Rate)
    for item in in_domain_dataset:
        query = item["query"]
        qvec = embedder.embed_query(query)
        search_results = index.query(vector=qvec, top_k=5, include_metadata=True)
        retrieved_docs = [Document(page_content=m.get("metadata", {}).get("text", ""), metadata=m.get("metadata", {})) for m in search_results.get("matches", [])]
        
        t0 = time.time()
        context = ""
        for i, doc in enumerate(retrieved_docs, start=1):
            context += f"[{i}] ({doc.metadata.get('source', 'unknown')})\n{doc.page_content}\n\n"
            
        gen_prompt = f"당신은 가이다 플레이 스튜디오 HR 챗봇입니다.\n아래 문서에만 근거하여 답하세요.\n\n#질문\n{query}\n\n#문서\n{context}"
        final_answer = llm.invoke(gen_prompt).content.strip()
        
        ver_context = "".join([f"- {doc.page_content}\n" for doc in retrieved_docs])
        ver_prompt = f"아래 '답변'이 '문서' 내용과 일치하면 '일치함', 아니면 '불일치함'을 출력하세요.\n\n#문서\n{ver_context}\n\n#답변\n\"{final_answer}\""
        verdict = llm.invoke(ver_prompt).content.strip()
        
        llm_latencies.append((time.time() - t0) * 1000)
        
        if "일치함" not in verdict:
            hallucination_count += 1

    # 2. OOD 평가 (Rejection Accuracy)
    rejection_count = 0
    for item in ood_dataset:
        query = item["query"]
        qvec = embedder.embed_query(query)
        search_results = index.query(vector=qvec, top_k=5, include_metadata=True)
        retrieved_docs = [Document(page_content=m.get("metadata", {}).get("text", ""), metadata=m.get("metadata", {})) for m in search_results.get("matches", [])]
        
        context = ""
        for i, doc in enumerate(retrieved_docs, start=1):
            context += f"[{i}] ({doc.metadata.get('source', 'unknown')})\n{doc.page_content}\n\n"
            
        gen_prompt = f"당신은 가이다 플레이 스튜디오 HR 챗봇입니다.\n아래 문서에만 근거하여 답하고 문서에 없으면 '문서에 근거가 없어 답변드리기 어렵습니다.'라고 하세요.\n\n#질문\n{query}\n\n#문서\n{context}"
        final_answer = llm.invoke(gen_prompt).content.strip()
        
        if "답변드리기 어렵습니다" in final_answer or "근거가 없" in final_answer:
            rejection_count += 1

    n_in = len(in_domain_dataset)
    n_ood = len(ood_dataset)
    hallucination_rate = hallucination_count / n_in if n_in > 0 else 0.0
    rejection_accuracy = rejection_count / n_ood if n_ood > 0 else 0.0
    avg_llm_lat = float(np.mean(llm_latencies)) if llm_latencies else 0.0

    results = {
        model_name: {
            "hallucination_rate": round(hallucination_rate, 4),
            "rejection_accuracy": round(rejection_accuracy, 4),
            "llm_latency_ms": round(avg_llm_lat, 2)
        }
    }

    logger.info(f"  Hallucination Rate: {hallucination_rate:.4f} ({hallucination_count}/{n_in})")
    if n_ood > 0:
        logger.info(f"  Rejection Accuracy: {rejection_accuracy:.4f} ({rejection_count}/{n_ood})")
    logger.info(f"  LLM Latency: {avg_llm_lat:.2f}ms")
    
    return results

# ─────────────────────────────────────────────
# 결과 출력
# ─────────────────────────────────────────────
MODEL_LABELS = {
    "openai_small": "text-embedding-3-small",
    "openai_large": "text-embedding-3-large",
    "solar_large": "solar-embedding-1-large",
    "qwen_0.6b": "qwen3-embedding:0.6b",
}


def print_step1_results(step1: Dict):
    """Step 1 결과 테이블 출력"""
    models = list(MODELS.keys())

    print("\n")
    print("  Step 1: Embedding Quality (numpy cosine similarity)")
    header = f"{'Model':<30} {'Avg Rel':>8} {'Avg Irr':>8} {'Gap':>8} {'Latency':>10}"
    print(header)
    print("-" * 70)
    for m in models:
        s = step1[m]
        label = MODEL_LABELS[m]
        print(f"{label:<30} {s['avg_relevant_cosine']:>8.4f} {s['avg_irrelevant_cosine']:>8.4f} "
              f"{s['cosine_gap']:>8.4f} {s['embedding_latency_ms']:>8.2f}ms")
    print("=" * 70)

def print_step2_results(step2: Dict):
    """Step 2 결과 테이블 출력"""
    models = list(MODELS.keys())

    print("\n")
    print("=" * 70)
    print("  Step 2: Retrieval Performance (Pinecone)")
    print("=" * 70)
    header = f"{'Model':<30} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} {'MRR':>7}"
    print(header)
    print("-" * 70)
    for m in models:
        s = step2[m]
        label = MODEL_LABELS[m]
        print(f"{label:<30} {s['hit_at_1']:>7.4f} {s['hit_at_3']:>7.4f} "
              f"{s['hit_at_5']:>7.4f} {s['mrr']:>7.4f}")
    print("=" * 70)


def print_step3_results(step3: Dict):
    """Step 3 결과 테이블 출력"""
    print("\n")
    print("  Step 3: Reranker Performance (text-embedding-3-small + LLM)")
    header = f"{'Model':<30} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} {'MRR':>7}"
    print(header)
    print("-" * 70)
    for m, s in step3.items():
        label = MODEL_LABELS.get(m, m) + " + LLM"
        print(f"{label:<30} {s['hit_at_1']:>7.4f} {s['hit_at_3']:>7.4f} "
              f"{s['hit_at_5']:>7.4f} {s['mrr']:>7.4f}")
    print("=" * 70)


def print_step4_results(step4: Dict):
    """Step 4 결과 테이블 출력"""
    print("\n")
    print("  Step 4: Verification Loop (text-embedding-3-small + LLM)")
    header = f"{'Model':<30} {'Hallucination Rate':>20} {'Rejection Accuracy':>20}"
    print(header)
    print("-" * 70)
    for m, s in step4.items():
        label = MODEL_LABELS.get(m, m)
        print(f"{label:<30} {s['hallucination_rate']:>20.4f} {s['rejection_accuracy']:>20.4f}")
    print("=" * 70)


def print_latency_results(step1: Dict, step2: Dict, step3: Dict = None, step4: Dict = None):
    """Latency 통합 테이블 출력 (초 단위 s 변환)"""
    models = list(MODELS.keys())

    print("\n")
    print("=" * 70)
    print("  Latency (s/query)")
    print("=" * 70)
    header = f"{'Model':<25} {'Embed':>8} {'Retrieve':>10} {'Reranker':>10} {'LLM':>6} {'Total':>8}"
    print(header)
    print("-" * 70)
    for m in models:
        label = MODEL_LABELS[m]
        embed_s = (step1[m]["embedding_latency_ms"] if step1 else 0.0) / 1000.0
        retrieve_s = (step2[m]["retrieval_latency_ms"] if step2 else 0.0) / 1000.0
        
        rerank_s = 0.0
        if step3 and m in step3:
            rerank_s = step3[m].get("reranker_latency_ms", 0.0) / 1000.0
            
        llm_s = 0.0
        if step4 and m in step4:
            llm_s = step4[m].get("llm_latency_ms", 0.0) / 1000.0
            
        total_s = embed_s + retrieve_s + rerank_s + llm_s
        
        rerank_str = f"{rerank_s:>9.2f}s" if rerank_s > 0 else f"{'-':>10}"
        llm_str = f"{llm_s:>5.2f}s" if llm_s > 0 else f"{'-':>6}"
        
        print(f"{label:<25} {embed_s:>7.2f}s {retrieve_s:>9.2f}s {rerank_str} {llm_str} {total_s:>7.2f}s")
    print("=" * 70)

# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode not in ("step1", "step2", "step3", "step4", "all"):
        print("사용법: python test.py [step1|step2|step3|step4|all]")
        print("  step1 — 임베딩 모델 비교 (numpy cosine similarity)")
        print("  step2 — 전체 RAG 검색 성능 (Pinecone)")
        print("  step3 — Reranker 성능 및 시간 측정")
        print("  step4 — 검증 루프 (Hallucination 및 Rejection 성능)")
        print("  all   — Step 1, 2, 3, 4 전체 실행")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"RAG 파이프라인 성능 평가 테스트 시작 (mode={mode})")
    logger.info("=" * 60)

    # 공통: 데이터 로드 & 문서 청킹
    dataset = load_test_dataset()
    logger.info(f"테스트 데이터셋 로드: {len(dataset)}개 Q&A (OOD 포함)")

    chunks = _load_and_split_docs(EXISTING_HR_DOCS)
    logger.info(f"문서 청킹 완료: {len(chunks)}개 청크")

    step1_results = None
    step2_results = None
    step3_results = None
    step4_results = None

    # Step 1 — Embedding Quality
    if mode in ("step1", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("Step 1: Embedding Quality 평가 시작 (numpy)")
        logger.info("=" * 60)
        step1_results = step1_embedding_quality(chunks, dataset)
        print_step1_results(step1_results)

    # Step 2 — Retrieval 성능
    if mode in ("step2", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("Step 2: Retrieval 성능 평가 시작 (Pinecone)")
        logger.info("=" * 60)
        step2_results = step2_retrieval_eval(chunks, dataset)
        print_step2_results(step2_results)

    # Step 3 — Reranker 성능
    if mode in ("step3", "all"):
        step3_results = step3_reranker_eval(chunks, dataset)
        print_step3_results(step3_results)

    # Step 4 — Verification Loop성능
    if mode in ("step4", "all"):
        step4_results = step4_verification_eval(chunks, dataset)
        print_step4_results(step4_results)

    # Latency 통합
    if mode == "all" and step1_results and step2_results and step3_results and step4_results:
        print_latency_results(step1_results, step2_results, step3_results, step4_results)

    print("\n" + "=" * 70)
    print("  평가 완료")
    print("=" * 70)

    '''
      Step 1: Embedding Quality (numpy cosine similarity)
Model                           Avg Rel  Avg Irr      Gap    Latency
----------------------------------------------------------------------
text-embedding-3-small           0.4898   0.2270   0.2628   182.96ms
text-embedding-3-large           0.4907   0.2389   0.2518   147.74ms
solar-embedding-1-large          0.4897   0.2437   0.2460   535.54ms
qwen3-embedding:0.6b             0.5273   0.2507   0.2766    52.17ms

  Step 2: Retrieval Performance (Pinecone)
======================================================================
Model                            Hit@1   Hit@3   Hit@5     MRR
----------------------------------------------------------------------
text-embedding-3-small          0.9333  1.0000  1.0000  0.9556
text-embedding-3-large          0.8333  1.0000  1.0000  0.9000
solar-embedding-1-large         0.9333  1.0000  1.0000  0.9556
qwen3-embedding:0.6b            0.9667  1.0000  1.0000  0.9778

  Latency (ms/query)
======================================================================
Model                           Embedding  Retrieval      Total
----------------------------------------------------------------------
text-embedding-3-small           182.96ms   225.42ms   408.38ms
text-embedding-3-large           147.74ms   271.13ms   418.87ms
solar-embedding-1-large          535.54ms   241.06ms   776.60ms
qwen3-embedding:0.6b              52.17ms   208.51ms   260.68ms

Latency — embed: 454.02ms  retrieve: 271.64ms  rerank: 11129.29ms

Step 3: Reranker Performance (text-embedding-3-small + LLM)
Model                            Hit@1   Hit@3   Hit@5     MRR
----------------------------------------------------------------------
text-embedding-3-small + LLM    1.0000  1.0000  1.0000  1.0000

Step 4: Verification Loop (text-embedding-3-small + LLM)
Model                            Hallucination Rate   Rejection Accuracy
----------------------------------------------------------------------
text-embedding-3-small                       0.0000               1.0000
'''




