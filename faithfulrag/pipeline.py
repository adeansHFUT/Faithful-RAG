import asyncio
from typing import Dict, List, Optional
from datasets import Dataset
from tqdm import tqdm
from .evaluate import (
    exact_match_score, 
    acc_score, 
    f1_score, 
    metric_max_over_ground_truths
)
from .modules import (
    FactMiningModule,
    ContextualAlignmentModule,
    SelfThinkModule
)
from .util import FormatConverter
# ===== 新增：通用 JSONL 缓存工具 =====
import os, json
from contextlib import contextmanager

try:
    import fcntl
    _HAS_FCNTL = True
except Exception:
    _HAS_FCNTL = False

@contextmanager
def _locked_append(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, "a+", encoding="utf-8")
    try:
        if _HAS_FCNTL:
            fcntl.flock(f, fcntl.LOCK_EX)
        yield f
    finally:
        if _HAS_FCNTL:
            fcntl.flock(f, fcntl.LOCK_UN)
        f.close()

def _load_jsonl_as_map(path: str, key: str = "id"):
    m = {}
    if not path or not os.path.exists(path):
        return m
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                k = obj.get(key)
                if k is not None:
                    m[k] = obj
            except json.JSONDecodeError:
                continue
    return m

def _append_jsonl(path: str, obj: dict):
    with _locked_append(path) as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

class FaithfulRAG:
    def __init__(
        self,
        backend_type: str,
        model_name: str,
        similarity_model: str,
        mining_sampling_params: Optional[Dict] = None,
        generation_sampling_params: Optional[Dict] = None,
        *,
        # ===== 新增：阶段性缓存文件，可选 =====
        knowledges_cache_path: Optional[str] = None,
        contexts_cache_path: Optional[str] = None,
        self_facts_cache_path: Optional[str] = None,
        chunks_cache_path: Optional[str] = None,
        predictions_cache_path: Optional[str] = None,
        **backend_config
    ):
        """
        Initialize the FaithfulRAG pipeline
        
        Args:
            backend_type: Type of LLM backend (openai, hf, llamafactory)
            model_name: Name of the model to use
            similarity_model: Sentence similarity model name
            mining_sampling_params: Parameters for fact mining generation
            generation_sampling_params: Parameters for answer generation
            backend_config: Backend-specific configuration parameters
        """
        self.backend_type = backend_type
        self.model_name = model_name
        self.similarity_model = similarity_model
        self.knowledges_cache_path = knowledges_cache_path
        self.contexts_cache_path = contexts_cache_path
        self.self_facts_cache_path = self_facts_cache_path
        self.chunks_cache_path = chunks_cache_path
        self.predictions_cache_path = predictions_cache_path
        
        # Set default sampling parameters if not provided
        self.mining_sampling_params = mining_sampling_params or {
            'max_tokens': 1000,
            'top_p': 1.0,
            'temperature': 0.0
        }
        
        self.generation_sampling_params = generation_sampling_params or {
            'max_tokens': 1000,
            'top_p': 1.0,
            'temperature': 0.0
        }
        
        # Initialize modules
        self.fact_mining_module = FactMiningModule(
            backend_type=backend_type,
            model_name=model_name,
            **backend_config
        )
        
        self.contextual_alignment_module = ContextualAlignmentModule(
            similarity_model=similarity_model
        )
        
        self.self_think_module = SelfThinkModule(
            backend_type=backend_type,
            model_name=model_name,
            **backend_config
        )
    
    async def get_self_facts(
        self, 
        dataset: Dataset, 
        fact_mining_type: str = "default",
        **mining_params
    ) -> List[Dict]:
        """
        Generate self-consistent facts for the dataset
        
        Args:
            dataset: Input dataset
            fact_mining_type: Type of fact mining ("default")
            mining_params: Override parameters for fact mining
            
        Returns:
            List of self-consistent facts dictionaries
        """
        # Use provided parameters or defaults
        params = {**self.mining_sampling_params, **mining_params}
        
        if fact_mining_type == "default":
                # ---------- 阶段 1：候选知识（knowledges） ----------
            if self.knowledges_cache_path:
                k_cache = _load_jsonl_as_map(self.knowledges_cache_path, key="id")
                missing_ds_k = [ex for ex in dataset if ex['id'] not in k_cache]
                if missing_ds_k:
                    new_ks = await self.fact_mining_module.generate_knowledges(missing_ds_k, **params)
                    for obj in new_ks:
                        _append_jsonl(self.knowledges_cache_path, obj)
                        k_cache[obj["id"]] = obj
                knowledges = [k_cache[ex['id']] for ex in dataset if ex['id'] in k_cache]
            else:
                knowledges = await self.fact_mining_module.generate_knowledges(dataset, **params)

            # ---------- 阶段 2：自我背景文（self_context） ----------
            if self.contexts_cache_path:
                c_cache = _load_jsonl_as_map(self.contexts_cache_path, key="id")
                missing_ids_c = [ex['id'] for ex in dataset if ex['id'] not in c_cache]
                if missing_ids_c:
                    # 子集的 dataset & 对应的子集 knowledges，顺序与 dataset 保持一致
                    sub_ds = [ex for ex in dataset if ex['id'] in missing_ids_c]
                    k_map = {k['id']: k for k in knowledges}
                    sub_k = [k_map[ex['id']] for ex in sub_ds]
                    new_ctxs = await self.fact_mining_module.generate_self_context(sub_ds, knowledges=sub_k, **params)
                    for obj in new_ctxs:
                        _append_jsonl(self.contexts_cache_path, obj)
                        c_cache[obj["id"]] = obj
                self_context = [c_cache[ex['id']] for ex in dataset if ex['id'] in c_cache]
            else:
                self_context = await self.fact_mining_module.generate_self_context(dataset, knowledges=knowledges, **params)

            # ---------- 阶段 3：抽取自洽事实（self_facts） ----------
            if self.self_facts_cache_path:
                f_cache = _load_jsonl_as_map(self.self_facts_cache_path, key="id")
                # extract_facts 的输入是 contexts 列表；我们按 dataset 顺序检查缺失
                missing_ids_f = [ctx['id'] for ctx in self_context if ctx['id'] not in f_cache]
                if missing_ids_f:
                    sub_ctxs = [ctx for ctx in self_context if ctx['id'] in missing_ids_f]
                    new_facts = await self.fact_mining_module.extract_facts(sub_ctxs, **params)
                    for obj in new_facts:
                        _append_jsonl(self.self_facts_cache_path, obj)
                        f_cache[obj["id"]] = obj
                self_facts = [f_cache[ex['id']] for ex in dataset if ex['id'] in f_cache]
            else:
                self_facts = await self.fact_mining_module.extract_facts(self_context, **params)

            return self_facts
            
        else:
            raise ValueError(f"Unsupported fact mining type: {fact_mining_type}")
    
    def get_topk_chunks(
        self, 
        dataset: Dataset, 
        self_facts: List[Dict],
        sent_topk: int = 5,
        chunk_topk: int = 5,
        chunk_size: int = 20
    ) -> List[Dict]:
        """
        Retrieve top-k contextual chunks for each fact
        
        Args:
            dataset: Input dataset
            self_facts: Self-consistent facts
            sent_topk: Number of top sentences to retrieve
            chunk_topk: Number of top chunks to return
            chunk_size: Size of context chunks
            
        Returns:
            List of dictionaries with top-k chunks
        """
        if not self.chunks_cache_path:
            contextual_chunks = self.contextual_alignment_module.get_contextual_chunks(
                self_facts, dataset, sent_topk, chunk_size
            )
            return self.contextual_alignment_module.get_topk_contextual_chunks(contextual_chunks, chunk_topk)

        cache = _load_jsonl_as_map(self.chunks_cache_path, key="id")
        have = set(cache.keys())
        need_ids = [ex['id'] for ex in dataset if ex['id'] not in have]

        if need_ids:
            # 子集：与 ContextualAlignmentModule 约定顺序一致（zip）
            sub_ds = [ex for ex in dataset if ex['id'] in need_ids]
            f_map = {f['id']: f for f in self_facts}
            sub_f = [f_map[ex['id']] for ex in sub_ds]

            contextual_chunks = self.contextual_alignment_module.get_contextual_chunks(
                sub_f, sub_ds, sent_topk, chunk_size
            )
            topk = self.contextual_alignment_module.get_topk_contextual_chunks(
                contextual_chunks, chunk_topk
            )
            for item in topk:
                _append_jsonl(self.chunks_cache_path, item)
                cache[item["id"]] = item

        return [cache[ex['id']] for ex in dataset if ex['id'] in cache]
    
    async def get_predictions(
        self,
        dataset: Dataset, 
        facts: List[Dict],
        generation_type: str = "normal_cot",
        **generation_params
    ) -> Dict[str, str]:
        """
        Generate predictions for the dataset
        
        Args:
            dataset: Input dataset
            facts: Factual knowledge with top-k chunks
            generation_type: Type of generation ("normal_cot", "scheduled_cot", "wo_cot")
            generation_params: Override parameters for generation
            
        Returns:
            Dictionary of predictions keyed by item ID
        """
        # Use provided parameters or defaults
        params = {**self.generation_sampling_params, **generation_params}
        
         # 选择对应的预测函数（保持原有三种）
        if generation_type == "normal_cot":
            params['response_format'] = {"type": "json_object"}
            runner = self.self_think_module.predict_answer_normal_cot
        elif generation_type == "scheduled_cot":
            params['response_format'] = {"type": "json_object"}
            runner = self.self_think_module.predict_answer_scheduled_cot
        elif generation_type == "wo_cot":
            runner = self.self_think_module.predict_answer_wo_cot
        else:
            raise ValueError(f"Unsupported generation type: {generation_type}")

        # 不启用缓存：保持原行为
        if not self.predictions_cache_path:
            return await runner(dataset, facts, **params)

        # 启用缓存：只对缺失 id 生成
        cache = _load_jsonl_as_map(self.predictions_cache_path, key="id")
        have = set(cache.keys())
        need_ds = [ex for ex in dataset if ex['id'] not in have]

        if need_ds:
            # 需要把 facts 也对齐成与 need_ds 同序
            f_map = {f['id']: f for f in facts}
            need_facts = [f_map[ex['id']] for ex in need_ds]
            new_pred = await runner(need_ds, need_facts, **params)  # 返回 {id: text}
            for ex in need_ds:
                rec = {"id": ex['id'], "prediction": new_pred.get(ex['id'], "")}
                _append_jsonl(self.predictions_cache_path, rec)
                cache[ex['id']] = rec

        # 聚合成 {id: prediction}
        return {ex['id']: cache[ex['id']]["prediction"] for ex in dataset if ex['id'] in cache}
    
    def evaluate(
        self, 
        dataset: Dataset, 
        predictions: Dict[str, str],
        cot_format: bool = False,
        detailed_output: bool = False
    ) -> Dict:
        """
        Evaluate predictions against ground truth
        
        Args:
            dataset: Input dataset with ground truth
            predictions: Generated predictions
            detailed_output: Whether to include per-item details
            
        Returns:
            Evaluation results dictionary
        """
        prediction_details = []
        total_em = total_acc = total_f1 = 0
        num_items = 0
        
        for item in tqdm(dataset, desc="Evaluating"):
            prediction = predictions.get(item['id'], "")
            # if prediction is in JSON format, extract the 'answer' field
            if cot_format:
                prediction = FormatConverter.extract_answer(prediction)
            ground_truth = item['answer']
            
            # Calculate metrics
            em_score = metric_max_over_ground_truths(
                exact_match_score, prediction, ground_truth)
            acc_score_val = metric_max_over_ground_truths(
                acc_score, prediction, ground_truth)
            f1_score_val = metric_max_over_ground_truths(
                f1_score, prediction, ground_truth)
            
            # Accumulate totals
            total_em += em_score
            total_acc += acc_score_val
            total_f1 += f1_score_val
            num_items += 1
            
            # Store details if requested
            if detailed_output:
                prediction_details.append({
                    "id": item['id'],
                    "question": item['question'],
                    "answer": ground_truth,
                    "prediction": prediction,
                    "exact_match": em_score,
                    "acc": acc_score_val,
                    "f1": f1_score_val
                })
        
        # Calculate averages
        avg_em = 100.0 * total_em / num_items if num_items > 0 else 0
        avg_acc = 100.0 * total_acc / num_items if num_items > 0 else 0
        avg_f1 = 100.0 * total_f1 / num_items if num_items > 0 else 0
        
        # Prepare result
        result = {
            "num_items": num_items,
            "exact_match": avg_em,
            "acc": avg_acc,
            "f1": avg_f1
        }
        
        if detailed_output:
            result["details"] = prediction_details
            
        return result