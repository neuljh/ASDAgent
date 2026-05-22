import os
import random
import logging
import json
import copy
import math

from pydantic import BaseModel, ValidationError

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

from asdagent.child_profile.compute_current_child_info import _age_to_years
from asdagent.tinyaba.json_schema.act import DoctorResponse
from asdagent.tinyaba.json_schema.child import RoleContent
from asdagent.tinyaba.json_schema.correct import DoctorFullResponse
from asdagent.tinyaba.json_schema.observe import ResponseData
from asdagent.tinyaba.json_schema.think import ThoughtProcess
from asdagent.tinyaba.prompts.act_full_assistant import ACT_FULL_ASSISTANT_SYS_PROMPT_TEMPLATE_FEW_SHOT, ACT_FULL_ASSISTANT_USER_PROMPT_TEMPLATE_FEW_SHOT
from asdagent.tinyaba.prompts.act_half_assistant import ACT_HALF_ASSISTANT_SYS_PROMPT_TEMPLATE_FEW_SHOT, ACT_HALF_ASSISTANT_USER_PROMPT_TEMPLATE_FEW_SHOT
from asdagent.tinyaba.prompts.act_instruct import ACT_INSTRUCT_SYS_PROMPT_TEMPLATE_FEW_SHOT, ACT_INSTRUCT_USER_PROMPT_TEMPLATE_FEW_SHOT
from asdagent.tinyaba.prompts.act_other import ACT_OTHER_SYS_PROMPT_TEMPLATE_FEW_SHOT, ACT_OTHER_USER_PROMPT_TEMPLATE_FEW_SHOT
from asdagent.tinyaba.prompts.act_reinforce import ACT_REINFORCE_SYS_PROMPT_TEMPLATE_FEW_SHOT, ACT_REINFORCE_USER_PROMPT_TEMPLATE_FEW_SHOT
from asdagent.tinyaba.prompts.child_act_irrelevent import CHILD_ACT_IRRELEVENT_SYS_PROMPT_TEMPLATE_FEW_SHOT, CHILD_ACT_IRRELEVENT_USER_PROMPT_TEMPLATE_FEW_SHOT
from asdagent.tinyaba.prompts.child_act_relevent import CHILD_ACT_RELEVENT_SYS_PROMPT_TEMPLATE_FEW_SHOT, CHILD_ACT_RELEVENT_USER_PROMPT_TEMPLATE_FEW_SHOT
from asdagent.tinyaba.prompts.child_act_repeat import CHILD_ACT_REPETITIVE_SYS_PROMPT_TEMPLATE_FEW_SHOT, CHILD_ACT_REPETITIVE_USER_PROMPT_TEMPLATE_FEW_SHOT
from asdagent.tinyaba.prompts.correct import STRATEGY_EXTRACTION_SYS_PROMPT, STRATEGY_EXTRACTION_USER_PROMPT
from asdagent.tinyaba.prompts.observe import OBSERVE_SYS_PROMPT_TEMPLATE_FEW_SHOT, OBSERVE_USER_PROMPT_TEMPLATE_FEW_SHOT
from asdagent.tinyaba.prompts.opening import DOCTOR_OPENING_PROMPT
from asdagent.tinyaba.prompts.think import *
from asdagent.utils.llm_api import generate_text_by_llm_api_via_openai, record_api_call

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.getenv("ASDAGENT_DATA_ROOT", REPO_ROOT / "data"))
OUTPUT_ROOT = Path(os.getenv("ASDAGENT_OUTPUT_ROOT", REPO_ROOT / "outputs"))
PROFILE_ROOT = Path(os.getenv("ASDAGENT_PROFILE_ROOT", DATA_ROOT))

# Configure logging to print to stdout with timestamps if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )



class LLMBackend:
    """Abstract interface for LLM calls."""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class MockLLM(LLMBackend):
    """Lightweight mock that returns simple JSON echoes for testing."""

    def generate(self, prompt: str) -> str:
        payload = {
            "role": "llm",
            "content": f"Simulated response for: {prompt[:80]}",
            "meta": {"note": "mocked"},
        }
        return json.dumps(payload, ensure_ascii=False)

def _count_turns_from_lines(lines: List[str]) -> int:
    """
    一轮 = 医生行动(可多条) + 儿童行动(单条)，以儿童说话次数为准。
    若未检测到儿童发言，回退为行数//2，至少1。
    """
    turns = 0
    for line in lines:
        role_token = line.strip().split("：", 1)[0] if "：" in line else line.strip().split(":", 1)[0]
        if "儿童" in role_token:
            turns += 1
    if turns == 0:
        turns = max(1, len(lines) // 2)
    return turns


def _load_dialogue_text(file_name: str) -> str:
    path = DATA_ROOT / "processed" / "all" / file_name
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def _compute_embedding(model, tokenizer, text: str):
    import torch

    with torch.no_grad():
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**tokens)
        emb = outputs.last_hidden_state.mean(dim=1)
    return emb.squeeze(0)


def _cosine_similarity(a, b) -> float:
    import torch

    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    return float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def _sample_turns_from_stats(stats_path: str, fallback: int, min_turns: int = 5, max_turns: int = 50) -> int:
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        turns_list = data.get("turns_list") or data.get("global_stats", {}).get("turns_list", [])
        turns_list = [t for t in turns_list if isinstance(t, (int, float)) and t > 0]
        if not turns_list:
            return fallback
        log_vals = [math.log(t) for t in turns_list if t > 0]
        if not log_vals:
            return fallback
        mu = sum(log_vals) / len(log_vals)
        sigma_sq = sum((v - mu) ** 2 for v in log_vals) / len(log_vals)
        sigma = math.sqrt(sigma_sq) if sigma_sq > 0 else 0.1
        logging.info(f'mu: {mu}, sigma: {sigma}')
        sample = random.lognormvariate(mu, sigma)
        return int(max(min_turns, min(max_turns, round(sample))))
    except Exception:
        return fallback


class LLMClient:
    """LLM facade that prefers the real API but falls back to a mock."""

    def __init__(
        self,
        model_name: str = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        backend: Optional[LLMBackend] = None,
    ) -> None:
        load_dotenv()
        self.model_name = model_name or os.getenv("ASDAGENT_MODEL") or os.getenv("MODEL_GPT4O_MINI_NAME") or "gpt-4o-mini"
        self.base_url = base_url or os.getenv("ASDAGENT_BASE_URL") or os.getenv("GPT_BASE_URL") or "https://api.openai.com/v1"
        self.api_key = api_key or os.getenv("ASDAGENT_API_KEY") or os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.backend = backend or MockLLM()

    def chat(self, messages: List[Dict[str, str]]) -> str:
        # logging.info(self.api_key)
        # logging.info(self.base_url)
        # logging.info(generate_text_by_llm_api_via_openai)
        if self.api_key and self.base_url and generate_text_by_llm_api_via_openai:
            response = generate_text_by_llm_api_via_openai(
                messages,
                model_name=self.model_name,
                base_url=self.base_url,
                api_key=self.api_key,
            )
            if record_api_call:
                record_api_call(response)
            return response.choices[0].message.content
        fallback_prompt = json.dumps(messages, ensure_ascii=False)
        return self.backend.generate(fallback_prompt)
        # if self.api_key and self.base_url and generate_text_by_llm_api_via_openai:
        #     try:
        #         response = generate_text_by_llm_api_via_openai(
        #             messages,
        #             model_name=self.model_name,
        #             base_url=self.base_url,
        #             api_key=self.api_key,
        #         )
        #         return response.choices[0].message.content
        #     except Exception:
        #         pass
        # fallback_prompt = json.dumps(messages, ensure_ascii=False)
        # return self.backend.generate(fallback_prompt)

    def chat_json(self, messages: List[Dict[str, str]]) -> Any:
        raw = self.chat(messages)
        return Validator.parse_json_from_text(raw)

    def chat_json_via_json_schema(
        self,
        messages: List[Dict[str, str]],
        schema_model: type[BaseModel],
        max_tries: int = 5,
    ) -> Any:
        """Call chat, parse JSON, and validate against a Pydantic schema. Retry up to max_tries."""
        if not self.api_key:
            fallback = self._fallback_for_schema(schema_model)
            if fallback is not None:
                return fallback

        last_error: Optional[Exception] = None
        for _ in range(max_tries):
            raw = self.chat(messages)
            logging.info(f"LLM Raw Output: {raw}")
            try:
                parsed = Validator.parse_json_from_text(raw)
                model = schema_model.parse_obj(parsed)
                return model.dict()
            except (ValidationError, ValueError, TypeError) as exc:
                last_error = exc
                continue
        raise ValueError(f"LLM output failed schema validation after {max_tries} attempts: {last_error}")

    @staticmethod
    def _fallback_for_schema(schema_model: type[BaseModel]) -> Optional[Dict[str, Any]]:
        """Return schema-valid placeholders for local smoke tests without an API key."""
        schema_name = getattr(schema_model, "__name__", "")
        fallbacks: Dict[str, Dict[str, Any]] = {
            "ResponseData": {
                "response_type": "相关的回答",
                "reason": "Mock fallback without an API key.",
                "semantic": {"sign": True, "reason": "Mock semantic match."},
                "quality_assessment": "Mock response.",
                "emotional_state": {"stress": "Low", "engagement": "Medium"},
                "detailed_observation": "Mock observation for local smoke testing.",
            },
            "ThoughtProcess": {
                "step": 1,
                "step_evidence": "Mock fallback without an API key.",
                "chain_of_thought": ["Use a minimal safe action for smoke testing."],
                "strategy": "指令",
                "reason": "Mock strategy selection.",
                "take_action": False,
            },
            "DoctorResponse": {
                "role": "医生",
                "content": "我们继续试一试。",
                "strategy": "指令",
            },
            "RoleContent": {
                "role": "儿童",
                "content": "好的。",
                "type": "相关的回答",
                "detail": None,
            },
            "DoctorFullResponse": {
                "full_response": "我们继续试一试。",
                "segments": [{"content": "我们继续试一试。", "strategy": "指令"}],
            },
        }
        return fallbacks.get(schema_name)


@dataclass
class DialogueTurn:
    role: str
    content: str
    type: Optional[str] = None
    observe_type: Optional[str] = None
    strategy: Optional[str] = None
    reason: Optional[str] = None
    take_action: Optional[bool] = None
    detail: Optional[str] = None
    segments: Optional[list] = None


def logging_dialogue(turns: List[DialogueTurn]) -> None:
    for turn in turns:
        logging.info(f"{turn['role']} ({turn['observe_type'] or turn['strategy']}): {turn['content']}")

class MemoryStream:
    def __init__(self) -> None:
        self.dialogue_history: List[Dict[str, Any]] = []
        self.observations: List[str] = []
        self.strategies: Dict[List[Dict[str, Any]]] = {}
        self.interruptions: List[Dict[str, Any]] = []

    def add_dialogue(self, entry: DialogueTurn) -> None:
        self.dialogue_history.append(
            {
                "role": entry.role,
                "content": entry.content,
                "type": entry.type,
                "strategy": entry.strategy,
                # "reason": entry.reason,
                "take_action": entry.take_action,
                "detail": entry.detail,
                "observe_type": entry.observe_type,
                "segments": entry.segments
            }
        )

    def add_observation(self, observation: str) -> None:
        self.observations.append(observation)

    def add_strategy(
        self, strategy: str, content: str, turn_number: int
    ) -> None:
        if turn_number not in self.strategies.keys():
            self.strategies[turn_number] = []
        self.strategies[turn_number].append(
            {
                "strategy": strategy,
                "content": content,
            }
        )
        

    def export(self) -> Dict[str, Any]:
        return {
            "dialogue_history": self.dialogue_history,
            "observations": self.observations,
            "interruptions": self.interruptions,
        }


class Validator:
    @staticmethod
    def parse_json_from_text(text: str) -> Any:
        """Extract JSON even if wrapped in markdown fences."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.lstrip("`")
            cleaned = cleaned.rstrip("`")
            parts = cleaned.split("\n", 1)
            if len(parts) == 2:
                cleaned = parts[1]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON: {exc}") from exc

    @staticmethod
    def logic_check(child_type: str, doctor_strategy: str) -> bool:
        matrix = {
            "相关的回答": "强化",
            "不相关的回答": "指令",
            "重复": "指令",
            "无响应": "辅助",
        }
        expected = matrix.get(child_type)
        return expected == doctor_strategy


@dataclass
class ABADocument:
    id: str
    tags: Sequence[str]
    content: str


class ABAKnowledgeBase:
    def __init__(self, documents: Optional[Sequence[ABADocument]] = None) -> None:
        self.documents: List[ABADocument] = list(documents) if documents else self._default_docs()

    @staticmethod
    def _default_docs() -> List[ABADocument]:
        return [
            ABADocument(
                id="rule_001",
                tags=["no response", "无响应", "assistance", "dtt"],
                content=(
                    "When a child is unresponsive, the 'Prompt Hierarchy' strategy should "
                    "be used. Use verbal assistance (continue using language assistance, or say the first part of the answer)."
                ),
            ),
            ABADocument(
                id="rule_002",
                tags=["incorrect response", "irrelevant", "不相关的回答", "重复", "correction"],
                content=(
                    "When a child gives an incorrect or irrelevant answer, the therapist should immediately: 1. Accept the child's response "
                    "if it is semantic; 2. Reissue the instruction; 3. Provide assistance immediately to ensure the next response is correct "
                    "(error-free learning)."
                ),
            ),
            ABADocument(
                id="rule_003",
                tags=["correct response", "相关的回答", "reinforcement"],
                content=(
                    "When a child responds correctly on their own, they should receive the strongest social reinforcement and concrete material "
                    "reinforcement (if applicable). Reinforcement must be provided after the behavior has occurred."
                ),
            ),
        ]

    def retrieve(self, query_context: str) -> ABADocument:
        query = query_context.lower()
        scored: List[tuple[int, ABADocument]] = []
        for doc in self.documents:
            score = sum(1 for tag in doc.tags if tag.lower() in query)
            scored.append((score, doc))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[0][1]


class DoctorState(Enum):
    IDLE = auto()
    DEMAND_ISSUED = auto()
    WAITING_RESPONSE = auto()
    CONSEQUENCE = auto()


class DoctorAgent:
    def __init__(
            self, 
            knowledge_base: ABAKnowledgeBase, 
            llm: LLMClient,
            child_profile: Optional[Dict[str, Any]] = None,
        ) -> None:
        self.state = DoctorState.IDLE
        self.knowledge_base = knowledge_base
        self.llm = llm
        self.memory = MemoryStream()
        if child_profile is None:
            child_profile = {
                "name": "小明",
                "age": "5岁",
                "gender": "男",
                "verbal_level": "2岁",
            }
        self.child_profile = child_profile  

    def generate_opening(self, topic: str, mode: str, child_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据主题和儿童画像，动态生成开场白。
        """
        # 1. 准备填充数据
        prompt_input = {
            "topic": topic,
            "mode": mode,
            "name": child_profile.get("name", "小朋友"),
            "gender": child_profile.get("gender", '男'),
            "verbal_level": child_profile.get("verbal_level", "3岁")
        }

        # 2. 构造消息
        messages = [
            {
                "role": "system",
                "content": DOCTOR_OPENING_PROMPT.format(**prompt_input)
            },
            {
                "role": "user",
                "content": f"请围绕此主题{topic}生成本节课的开场白。"
            }
        ]

        # # 3. 调用 LLM
        # try:
        #     response = self.llm.chat_json_via_json_schema(
        #         messages=messages,
        #         schema_model=DoctorOpening,  
        #     )
        # except Exception as e:
        #     # 降级容错：如果 LLM 失败，回退到硬编码
        #     print(f"Opening Generation Error: {e}")
        #     response = {
        #         "role": "医生",
        #         "content": f"你好{child_profile.get('name')}，今天我们来学习{topic}。准备好了吗？",
        #         "strategy": "指令",
        #         "reason": "Fallback opening",
        #     }
        response = {
                "role": "医生",
                "content": f"你好{child_profile.get('name')}，今天我们来学习{topic}。准备好了吗？",
                "strategy": "指令",
                "reason": f"围绕主题‘{topic}’，生成一个简单的开场白。",
        }

        # 4. 确保字段完整
        response["role"] = "医生"
        response.setdefault("strategy", "指令") # 开场通常都是发起指令
        
        return response

    @staticmethod
    def get_prompt_by_strategy(strategy: str) -> str:
        if strategy == "指令":
            return ACT_INSTRUCT_USER_PROMPT_TEMPLATE_FEW_SHOT, ACT_INSTRUCT_SYS_PROMPT_TEMPLATE_FEW_SHOT
        elif strategy == "强化":
            return ACT_REINFORCE_USER_PROMPT_TEMPLATE_FEW_SHOT, ACT_REINFORCE_SYS_PROMPT_TEMPLATE_FEW_SHOT
        elif strategy == "全辅助":
            return ACT_FULL_ASSISTANT_USER_PROMPT_TEMPLATE_FEW_SHOT, ACT_FULL_ASSISTANT_SYS_PROMPT_TEMPLATE_FEW_SHOT
        elif strategy == "半辅助":
            return ACT_HALF_ASSISTANT_USER_PROMPT_TEMPLATE_FEW_SHOT, ACT_HALF_ASSISTANT_SYS_PROMPT_TEMPLATE_FEW_SHOT
        elif strategy == "其他":
            return ACT_OTHER_USER_PROMPT_TEMPLATE_FEW_SHOT, ACT_OTHER_SYS_PROMPT_TEMPLATE_FEW_SHOT
        return None, None

    def observe(
            self, 
            topic: str,
            child_response: Dict[str, Any], 
            history: Optional[List[Dict[str, Any]]] = None,
            doctor_last_action: Optional[Dict[str, Any]] = None,
            max_histoty_len: Optional[int] = 15,
        ) -> Dict[str, Any]:
            """
            使用 LLM 感知儿童的行为，分析其背后的动机和状态。
            """
            recent_history = history[-max_histoty_len:] if history else []
            
            # 1. 准备上下文
            last_content = "（对话刚开始，无前置指令）"
            if doctor_last_action and "content" in doctor_last_action and "strategy" in doctor_last_action:
                last_content = doctor_last_action["content"]
                last_strategy = doctor_last_action["strategy"]
                
            child_content = child_response.get("content", "[No Response]")
            child_type = child_response.get("type", "Unknown")

            # 2. 构造 Prompt
            # info = json.dumps(
            #             {
            #                 "doctor_behavior":{
            #                     "strategy": last_strategy,
            #                     "content": last_content
            #                 },
            #                 "child_response":{
            #                     "content": child_content
            #                 }
            #             }
            #             , ensure_ascii=False
            #         )
            prompt = [
                {
                    "role": "system",
                    "content": OBSERVE_SYS_PROMPT_TEMPLATE_FEW_SHOT
                },
                {
                    "role": "user",
                    "content": OBSERVE_USER_PROMPT_TEMPLATE_FEW_SHOT.format(
                        topic=topic,
                        history=recent_history,
                        child_response=child_content
                    )
                }
            ]

            # 3. 调用 LLM
            try:
                observation = self.llm.chat_json_via_json_schema(
                    messages=prompt,
                    schema_model=ResponseData,  
                )
            except Exception as e:
                # 降级容错：如果 LLM 挂了，回退到简单的规则判断
                print(f"Observation LLM Error: {e}")
                observation = {
                    "quality_assessment": "LLM Error",
                    "function_hypothesis": "Unknown",
                    "emotional_state": {"stress": "Unknown", "engagement": "Unknown"},
                    "detailed_observation": f"Fallback: Child type is {child_type}"
                }
                
            # 4. 存储到记忆流 (增加更丰富的描述)
            log_text = (
                f"Observation: {observation.get('detailed_observation')} "
                f"[Stress: {observation.get('emotional_state', {}).get('stress')}]"
            )
            self.memory.add_observation(log_text)
            
            # 5. 返回给 Think 环节使用
            # 我们把原始的 response 也并进去，方便后续使用
            return {**observation, "raw_response": child_response}

    def think(
        self,
        topic: str,
        child_response: Dict[str, Any],
        observation: Dict[str, Any],
        current_phase: str,
        history: Optional[List[Dict[str, Any]]] = None,
        prior_action: Optional[Dict[str, Any]] = None,
        turn_number: Optional[int] = None,
        max_histoty_len: Optional[int] = 15,
    ) -> Dict[str, Any]:
        child_type = child_response.get("type", "")
        # retrieved = self.knowledge_base.retrieve(child_type)
        strategy = self._select_strategy(child_type)
        recent_history = history[-max_histoty_len:] if history else []
        actions_just_taken = self.memory.strategies[turn_number] if turn_number in self.memory.strategies.keys() else []
        # consecutive_non_relevant = self._count_recent_non_relevant(recent_history)

        # if strategy == "指令" and child_type != "相关的回答" and consecutive_non_relevant >= 3:
        #     strategy = "辅助"
        #     observation = {**observation, "escalated_assist": True}
        info = json.dumps(
                    {
                        "child_info": {
                            "name": self.child_profile.get("name", "小明"),
                            "gender": self.child_profile.get("gender", "男"),
                            "age": self.child_profile.get("age", "4岁"),
                            "verbal_level": self.child_profile.get("verbal_level", "2岁")
                        },
                        "topic": topic,
                        "child_response": {
                            # "semantic": observation.get("semantic", {}).get("sign", True),
                            "response_type": observation.get("response_type", ""),
                            "content": child_response.get("content", ""),
                            # "response_type": child_type,
                        },
                        "observation": observation.get("detailed_observation", ""),
                        "actions_just_taken": actions_just_taken,
                        "history_tail": recent_history,
                        # "consecutive_non_relevant": consecutive_non_relevant,
                        # "retrieved_rule": {"id": retrieved.id, "content": retrieved.content},
                        "current_phase": current_phase,
                        # "expected_strategy": strategy,
                        # "prior_action": prior_action,
                    },
                    ensure_ascii=False,
                ),
        prompt = [
            {
                "role": "system",
                "content": (
                    THINK_SYS_PROMPT_TEMPLATE_FEW_SHOT
                ),
            },
            # {{
            # "topic": "当前的教学主题",
            # "child_last_response": {
            # "response_type": "儿童在上一轮的反应类型",
            # "content": "儿童在上一轮的具体反应内容"
            # },
            # "observation": "在感知阶段，根据儿童的反应类型和回答内容分析的儿童当前的状态，以及猜测儿童这样回复的原因。",

            # "actions_just_taken": [
            # {
            # "strategy": "在等待儿童回复前，已经采取的策略1",
            # },
            # {
            # "strategy": "在等待儿童回复前，已经采取的策略2",
            # }
            # ],

            # "history_tail": "儿童和医生的历史对话记录",
            # "retrieved_rule": {
            # "id": "检索到的规则id", 
            # "content": "检索到的规则内容"
            # },
            # "current_phase": "是DTT还是NET教学阶段"
            # }}
            {
                "role": "user",
                "content": THINK_USER_PROMPT_TEMPLATE_FEW_SHOT.format(info=info)
            },
        ]
        thought = self.llm.chat_json_via_json_schema(
            messages=prompt,
            schema_model=ThoughtProcess,  
        )
        # try:
        #     thought = self.llm.chat_json(prompt)
        # except Exception:
        #     thought = {
        #         "observation": f"Child responded with type '{child_type}'",
        #         "chain_of_thought": [
        #             f"Child type = {child_type}",
        #             f"Retrieved {retrieved.id}",
        #             f"Strategy set to {strategy} by rule matrix",
        #         ],
        #         "strategy": strategy,
        #         "reason": f"Based on rule {retrieved.id} for {child_type} during {current_phase}",
        #         "take_action": True,
        #         "next_action": "",
        #     }
        # if "observation" not in thought:
        #     thought["observation"] = f"Child responded with type '{child_type}'"
        # if "chain_of_thought" not in thought:
        #     thought["chain_of_thought"] = [f"Applying {retrieved.id} for {child_type}"]
        # if "reason" not in thought:
        #     thought["reason"] = f"Based on rule {retrieved.id} for {child_type} during {current_phase}"
        # thought.setdefault("take_action", True)
        # thought.setdefault("next_action", "")
        # if not Validator.logic_check(child_type, thought.get("strategy", "")):
        #     thought["strategy"] = strategy
        #     thought["reason"] = f"Adjusted to rule matrix using {retrieved.id}"
        thought["role"] = "医生"
        # thought["retrieved_rule"] = {"id": retrieved.id, "content": retrieved.content}
        # self.state = DoctorState.CONSEQUENCE
        actions_just_taken_strategy = [strategy["strategy"] for strategy in actions_just_taken]
        if thought["strategy"] in actions_just_taken_strategy:
            thought["take_action"] = False
            thought["reason"] = f"Already took action with strategy '{thought['strategy']}', waiting for child response."
        elif len(actions_just_taken_strategy) > 0 and actions_just_taken_strategy[-1] == "指令":
            thought["take_action"] = False
            thought["reason"] = f"Giving child time to respond to instruction."
        elif (thought["strategy"] == "半辅助" or thought["strategy"] == "全辅助") and ("全辅助" in actions_just_taken_strategy or "半辅助" in actions_just_taken_strategy):
            thought["take_action"] = False
            thought["reason"] = f"Already provided assistance, waiting for child response."
        
        return thought

    def _select_strategy(self, child_type: str) -> str:
        if child_type == "相关的回答":
            return "强化"
        if child_type in {"不相关的回答", "重复"}:
            return "指令"
        return "辅助"

    def _count_recent_non_relevant(self, history: List[Dict[str, Any]]) -> int:
        count = 0
        for entry in reversed(history):
            if entry.get("role") != "儿童":
                continue
            etype = entry.get("type")
            if etype == "相关的回答":
                break
            count += 1
        return count

    def correct(
            self,
            act_response: Dict[str, Any],
            history: Optional[List[Dict[str, Any]]] = None,
            max_history_len: Optional[int] = 15,
            turn_number: Optional[int] = None,
            max_tries: int = 3,
        ) -> Dict[str, Any]:
        """
        Use label_strategy prompts to split doctor content into strategy-tagged segments,
        ensure lossless reconstruction, and keep only the segments that match the chosen strategy.
        """
        history = history[-max_history_len:] if history else []
        doctor_content = act_response.get("content", "")
        strategy = act_response.get("strategy")
        if not doctor_content or not strategy:
            return act_response

        history_text = json.dumps(history or [], ensure_ascii=False, indent=2)
        messages = [
            {"role": "system", "content": STRATEGY_EXTRACTION_SYS_PROMPT},
            {
                "role": "user",
                "content": STRATEGY_EXTRACTION_USER_PROMPT.format(
                    history=history_text,
                    doctor_response=doctor_content,
                ),
            },
        ]

        parsed = None
        last_error: Optional[Exception] = None
        for _ in range(max_tries):
            try:
                parsed = self.llm.chat_json_via_json_schema(
                    messages=messages,
                    schema_model=DoctorFullResponse,
                    max_tries=1,
                )
            except Exception as exc:
                last_error = exc
                continue

            segments = parsed.get("segments", []) if isinstance(parsed, dict) else []
            reconstructed = "".join(seg.get("content", "") for seg in segments)
            if reconstructed == doctor_content:
                break
            parsed = None
            last_error = ValueError("Segment contents do not reconstruct the doctor response.")

        if parsed is None:
            logging.warning(f"Correct step failed after {max_tries} tries: {last_error}")
            return act_response

        segments = parsed.get("segments", [])

        segments = normalize_segments(segments)
        matched_segments = [seg for seg in segments if seg.get("strategy") == strategy]
        if matched_segments:
            act_response["content"] = "".join(seg.get("content", "") for seg in matched_segments)
        act_response["segments"] = segments

        if turn_number is not None:
            self.memory.add_strategy(
                strategy=strategy,
                content=act_response.get("content", doctor_content),
                turn_number=turn_number,
            )
        return act_response

    @staticmethod
    def extract_segments(
            history: Optional[List[Dict[str, Any]]],
            doctor_content: str,
            llm: Optional[LLMClient] = None,
            max_tries: int = 3,
            # model_name = Optional[str],
            # base_url = Optional[str],
            # api_key = Optional[str],
        ) -> Optional[Dict[str, Any]]:
        """
        Split doctor content into strategy-tagged segments using the label_strategy prompts.
        Ensures segments reconstruct the original content and validates against the schema.
        """
        if not doctor_content:
            return None

        llm_client = llm or LLMClient()
        history_text = json.dumps(history or [], ensure_ascii=False, indent=2)
        messages = [
            {"role": "system", "content": STRATEGY_EXTRACTION_SYS_PROMPT},
            {
                "role": "user",
                "content": STRATEGY_EXTRACTION_USER_PROMPT.format(
                    history=history_text,
                    doctor_response=doctor_content,
                ),
            },
        ]

        last_error: Optional[Exception] = None
        for _ in range(max_tries):
            try:
                parsed = llm_client.chat_json_via_json_schema(
                    messages=messages,
                    schema_model=DoctorFullResponse,
                    max_tries=1,
                )
            except Exception as exc:
                last_error = exc
                continue

            segments = parsed.get("segments", []) if isinstance(parsed, dict) else []
            reconstructed = "".join(seg.get("content", "") for seg in segments)
            if reconstructed == doctor_content:
                return parsed
            last_error = ValueError("Segment contents do not reconstruct the doctor response.")

        logging.warning(f"extract_segments failed after {max_tries} tries: {last_error}")
        return None

    def act(
            self, 
            thought: Dict[str, Any],
            topic: str,
            turn_number: Optional[int] = None,
            doctor_style: Optional[str] = "",
            history: Optional[List[Dict[str, Any]]] = None,
            max_history_len: Optional[int] = 15,
        ) -> Dict[str, Any]:
        strategy = thought.get("strategy", "")
        ACT_USER_PROMPT_TEMPLATE_FEW_SHOT, ACT_SYS_PROMPT_TEMPLATE_FEW_SHOT = self.get_prompt_by_strategy(strategy)
        recent_history = history[-max_history_len:] if history else []
        # logging.info(f"Doctor doctor_style: {doctor_style}")

        info = json.dumps(
                    {
                        "strategy": strategy,
                        "reason": thought.get("reason", ""),
                        "chain_of_thought": thought.get("chain_of_thought", []),
                        # "next_action": thought.get("next_action", ""),
                        "take_action": thought.get("take_action", True),
                        # "style": doctor_style,
                    },
                    ensure_ascii=False,
                ),
        child_info =  {
            "name": self.child_profile.get("name", "小明"),
            "gender": self.child_profile.get("gender", "男"),
            "age": self.child_profile.get("age", "4岁"),
            "verbal_level": self.child_profile.get("verbal_level", "2岁")
        }
        prompt = [
            {
                "role": "system",
                "content": (
                    ACT_SYS_PROMPT_TEMPLATE_FEW_SHOT.format(style=doctor_style)
                ),
            },
            {
                "role": "user",
                "content": ACT_USER_PROMPT_TEMPLATE_FEW_SHOT.format(
                    info=info,
                    history_tail=recent_history,
                    topic=topic,
                    child_info=child_info,
                )
            },
        ]
        response = self.llm.chat_json_via_json_schema(
            messages=prompt,
            schema_model=DoctorResponse,  
        )
        # try:
        #     response = self.llm.chat_json(prompt)
        # except Exception:
        #     response = None
        # if (
        #     not isinstance(response, dict)
        #     or "content" not in response
        #     or response.get("role") != "医生"
        #     or "Simulated response" in str(response.get("content", ""))
        # ):
        #     response = {
        #         "content": {
        #             "强化": "做得很好，我们继续！",
        #             "指令": "让我们再试一次，听好指令哦。",
        #             "辅助": "我来帮助你，先说出答案的第一部分。",
        #         }.get(strategy, "让我们继续练习。"),
        #         "reason": thought.get("reason", ""),
        #         "next_action": thought.get("next_action", "继续观察并给予提示"),
        #         "take_action": thought.get("take_action", True),
        #     }
        response["role"] = "医生"
        response["reason"] = response.get("reason", thought.get("reason", ""))
        response["strategy"] = strategy
        response.setdefault("take_action", thought.get("take_action", True))
        # response.setdefault("next_action", thought.get("next_action", ""))
        # if response.get("take_action") and response.get("next_action"):
        if response.get("take_action"):
            response["content"] = f"{response.get('content')}"
        return self.correct(
            act_response=response,
            history=recent_history,
            turn_number=turn_number,
        )


class ChildAgent:
    def __init__(
            self, 
            profile: Dict[str, Any], 
            llm: LLMClient, 
            # using_default_llm: bool = True
            global_stats: Optional[Dict[str, Any]] = None,
            alpha: float = 0.5,
        ) -> None:
        self.profile = profile
        self.internal_state = {"stress_level": 40, "engagement": 60}
        self.response_types = ["相关的回答", "不相关的回答", "重复", "无响应"]
        self.llm = llm
        self.alpha = max(0.0, min(alpha, 1.0))
        self.personal_stats = profile.get("child_stats", {})
        self.global_stats = global_stats or {}
        self.turn_strategies: Dict[int, List[str]] = {}
        local_bigram = profile.get("bigram") or {}
        global_payload = self.global_stats.get("global_stats", self.global_stats)
        global_bigram = global_payload.get("bigram", {})
        self.bigram = self._blend_bigram(local_bigram, global_bigram)
        # if using_default_llm:
        #     self.llm.model_name = "gpt4o-mini-ca"

    def _update_state(self, doctor_strategy: str) -> None:
        if doctor_strategy == "指令":
            self.internal_state["stress_level"] = min(100, self.internal_state["stress_level"] + 0.08)
            self.internal_state["engagement"] = max(0, self.internal_state["engagement"] - 0.06)
        elif doctor_strategy == "强化":
            self.internal_state["stress_level"] = max(0, self.internal_state["stress_level"] - 0.10)
            self.internal_state["engagement"] = min(100, self.internal_state["engagement"] + 0.12)
        elif doctor_strategy == "辅助":
            self.internal_state["stress_level"] = max(0, self.internal_state["stress_level"] - 0.04)
            self.internal_state["engagement"] = min(100, self.internal_state["engagement"] + 0.06)

    def _response_distribution(self, turn: int) -> List[float]:
        base = [0.50, 0.25, 0.10, 0.15]  # fallback
        strategies = self.turn_strategies.get(turn, [])
        key_seq = "，".join(strategies) if strategies else ""
        last = strategies[-1] if strategies else ""

        def get_probs(src: Dict[str, Dict[str, float]], key: str) -> Dict[str, float]:
            if not isinstance(src, dict):
                return {}
            return src.get(key, {})

        # personal and global distributions
        seq_personal = get_probs(self.personal_stats.get("sequential_probs", {}), key_seq)
        global_payload = self.global_stats.get("global_stats", self.global_stats)
        seq_global = get_probs(global_payload.get("stats", {}).get("sequential_probs", {}), key_seq)
        last_personal = get_probs(self.personal_stats.get("last_probs", {}), last)
        last_global = get_probs(global_payload.get("stats", {}).get("last_probs", {}), last)

        def blend(pers: Dict[str, float], glob: Dict[str, float]) -> Dict[str, float]:
            merged = {}
            keys = set(pers.keys()) | set(glob.keys())
            for k in keys:
                merged[k] = (1 - self.alpha) * pers.get(k, 0.0) + self.alpha * glob.get(k, 0.0)
            return merged

        dist_map = {}
        if seq_personal or seq_global:
            dist_map = blend(seq_personal, seq_global)
        elif last_personal or last_global:
            dist_map = blend(last_personal, last_global)

        def to_weights(d: Dict[str, float]) -> List[float]:
            if not d:
                return base
            vals = [
                d.get("相关的回答", 0.0),
                d.get("不相关的回答", 0.0),
                d.get("重复", 0.0),
                d.get("无响应", 0.0),
            ]
            s = sum(vals)
            if s <= 0:
                return base
            return [v / s for v in vals]

        return to_weights(dist_map)

    def record_strategy(self, turn: int, strategy: str) -> None:
        if turn not in self.turn_strategies:
            self.turn_strategies[turn] = []
        self.turn_strategies[turn].append(strategy)

    def _blend_bigram(self, local: Dict[str, float], global_b: Dict[str, float]) -> Dict[str, float]:
        # logging.info(local)

        # keys = {"child_after_prob", "doctor_after_prob", "child_after", "doctor_after"}
        keys = {"child_after_prob", "doctor_after_prob"}
        blended = {}
        for k in keys:
            lv = local.get(k, 0.0)
            gv = global_b.get(k, 0.0)
            blended[k] = (1 - self.alpha) * lv + self.alpha * gv
        # exp_vals = {k: math.exp(v) for k, v in blended.items()}
        # total = sum(exp_vals.values())
        # for k in blended:
        #     blended[k] = exp_vals[k] / total
        # logging.info(blended)
        return blended

    def _generate_content(self, response_type: str, topic: str) -> str:
        if response_type == "相关的回答":
            return f"这是一个{random.choice(['苹果', '香蕉', '橙子'])}。"
        if response_type == "不相关的回答":
            return "我想玩玩具车。"
        if response_type == "重复":
            return "苹果，苹果，苹果。"
        return "[No Response]"

    @staticmethod
    def get_prompt_by_response_type(response_type: str) -> str:
        # CHILD_ACT_RELEVENT_USER_PROMPT_FEW_SHOT, CHILD_ACT_IRRELEVENT_SYS_PROMPT_FEW_SHOT
        if response_type == "相关的回答":
            return CHILD_ACT_RELEVENT_USER_PROMPT_TEMPLATE_FEW_SHOT, CHILD_ACT_RELEVENT_SYS_PROMPT_TEMPLATE_FEW_SHOT
        elif response_type == "不相关的回答":
            return CHILD_ACT_IRRELEVENT_USER_PROMPT_TEMPLATE_FEW_SHOT, CHILD_ACT_IRRELEVENT_SYS_PROMPT_TEMPLATE_FEW_SHOT
        elif response_type == "重复":
            return CHILD_ACT_REPETITIVE_USER_PROMPT_TEMPLATE_FEW_SHOT, CHILD_ACT_REPETITIVE_SYS_PROMPT_TEMPLATE_FEW_SHOT
        return None, None

    def act(self, doctor_input: Dict[str, Any], topic: str) -> Dict[str, Any]:
        strategy = doctor_input.get("strategy", "指令")
        self._update_state(strategy)
        # record current doctor strategy externally via World

        weights = self._response_distribution(doctor_input.get("turn_number", 0))
        response_type = random.choices(self.response_types, weights=weights, k=1)[0]

        if response_type == "无响应":
            child_response = {"role": "儿童", "type": "无响应", "content": "[儿童无响应]", "detail": None}
        else:
            CHILD_ACT_USER_PROMPT_TEMPLATE_FEW_SHOT, CHILD_ACT_SYS_PROMPT_TEMPLATE_FEW_SHOT = self.get_prompt_by_response_type(response_type)

            info = json.dumps(
                        {
                            "doctor_input": doctor_input.get("content", ""),
                            "target_response_type": response_type,
                            # "state": self.internal_state,
                            # "profile": self.profile,
                        },
                        ensure_ascii=False,
                    )
            prompt = [
                {
                    "role": "system",
                    "content": (
                        CHILD_ACT_SYS_PROMPT_TEMPLATE_FEW_SHOT.format(
                            age=self.profile.get("age", "5岁"),
                            name=self.profile.get("name", "小朋友"),
                            gender=self.profile.get("gender", "男"),
                            verbal_level=self.profile.get("verbal_level", "2岁"),
                            dialogue_history=self.profile.get("dialogue_history", ""),
                        )
                        # CHILD_ACT_PROMPT_ONE_SHOT.format(
                        #     age=self.profile.get("age", 4),
                        #     name=self.profile.get("name", "小朋友"),
                        #     verbal_level=self.profile.get("verbal_level", "2岁"),
                        #     dialogue_history=self.profile.get("dialogue_history", ""),
                        #     stress=self.internal_state.get("stress_level", 40),
                        #     engagement=self.internal_state.get("engagement", 60),
                        #     target_response_type=response_type,
                        # )
                    ),
                },
                {
                    "role": "user",
                    "content": CHILD_ACT_USER_PROMPT_TEMPLATE_FEW_SHOT.format(info=info),
                },
            ]
            try:
                child_response = self.llm.chat_json_via_json_schema(
                    messages=prompt,
                    schema_model=RoleContent,  
                )
            except Exception:
                child_response = {}
        if not isinstance(child_response, dict):
            child_response = {}
        if child_response.get("type") not in self.response_types:
            # weights = self._response_distribution()
            # response_type = random.choices(self.response_types, weights=weights, k=1)[0]
            child_response["type"] = response_type
            child_response["content"] = self._generate_content(response_type, topic)
        if "content" not in child_response:
            response_type = child_response.get("type", "无响应")
            child_response["content"] = self._generate_content(response_type, topic)
        child_response["role"] = "儿童"
        return child_response

def normalize_segments(segments: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Merge misplaced strategies and enforce allowed patterns."""
    if not segments:
        return []
    merged: List[Dict[str, Any]] = []
    i = 0
    while i < len(segments):
        seg = dict(segments[i])
        strategy = seg.get("strategy", "")
        content = seg.get("content", "")

        # If '其他' or '强化' not in first position, merge into previous
        if strategy in {"其他", "强化"} and merged:
            merged[-1]["content"] = merged[-1].get("content", "") + content
            i += 1
            continue

        # If '指令' not in last position, merge into next
        if strategy == "指令" and i < len(segments) - 1:
            nxt = dict(segments[i + 1])
            nxt["content"] = content + nxt.get("content", "")
            segments[i + 1] = nxt
            i += 1
            continue

        merged.append(seg)
        i += 1

    # Merge consecutive identical strategies
    compact: List[Dict[str, Any]] = []
    for seg in merged:
        if compact and compact[-1].get("strategy") == seg.get("strategy"):
            compact[-1]["content"] = compact[-1].get("content", "") + seg.get("content", "")
        else:
            compact.append(seg)

    # allowed = {
    #     ("其他", "指令"),
    #     ("强化", "指令"),
    #     ("其他", "半辅助"),
    #     ("其他", "半辅助", "指令"),
    #     ("其他", "全辅助", "指令"),
    #     ("其他", "全辅助"),
    # }
    # seq = tuple(seg.get("strategy", "") for seg in compact)
    # print('After checking:')
    # print(compact)
    # if seq not in allowed:
    #     return None
    return compact

class World:
    def __init__(
            self, 
            topic: str,
            mode: str, 
            child_profile: Dict[str, Any],
            doctor_style: str = "",
            doctor_name: str = None,
            alpha: float = 0.7,
            global_stats_path: Optional[str] = None,
            output_dir: Optional[str] = None,
        ) -> None:
        self.topic = topic
        self.mode = mode
        self.memory = MemoryStream()
        self.knowledge_base = ABAKnowledgeBase()
        self.doctor_llm_client = LLMClient()
        self.child_llm_client = LLMClient(
    
        )
        self.doctor = DoctorAgent(self.knowledge_base, self.doctor_llm_client, child_profile)
        global_stats = {}
        global_stats_file = Path(global_stats_path or os.getenv("ASDAGENT_GLOBAL_STATS_PATH", PROFILE_ROOT / "child_global_stats.sample.json"))
        if global_stats_file.exists():
            try:
                with global_stats_file.open("r", encoding="utf-8") as gf:
                    global_stats = json.load(gf)
            except Exception:
                global_stats = {}
        self.child = ChildAgent(child_profile, self.child_llm_client, global_stats=global_stats, alpha=alpha)
        self.log: List[Dict[str, Any]] = []
        self.doctor_style = doctor_style
        self.output_dir = Path(output_dir or os.getenv("ASDAGENT_OUTPUT_DIR", OUTPUT_ROOT / "json_files"))

    def run_session(self, turns: int) -> Dict[str, Any]:
        logging.info(f"Starting session on topic '{self.topic}' in mode '{self.mode}' with {turns} turns.")
        logging.info(f"-------------Doctor Action Start-------------")
        doctor_action = self.doctor.generate_opening(
            topic=self.topic,
            mode=self.mode,
            child_profile=self.child.profile
        )
        logging.info(f"Doctor strategy: {doctor_action['strategy']}")
        logging.info(f"Doctor content: {doctor_action['content']}")
        logging.info(f"-------------Doctor Action End-------------")
        self.memory.add_dialogue(DialogueTurn(**doctor_action))
        for turn in range(1, turns + 1):
            # 初始化当轮的策略记录，并记录起始医生策略
            self.child.turn_strategies[turn] = []
            if doctor_action.get("strategy"):
                self.child.record_strategy(turn, doctor_action.get("strategy"))
            doctor_action["turn_number"] = turn
            logging.info(f"-------------Child Action Start(Turn {turn})-------------")
            child_response = self.child.act(doctor_action, topic=self.topic)
            if child_response["content"] == "":
                child_response["content"] = '[儿童无响应]'
            logging.info(f"Child response type: {child_response['type']}")
            logging.info(f"Child content: {child_response['content']}")
            logging.info(f"-------------Child Action End(Turn {turn})-------------")
            self.memory.add_observation(
                f"Turn {turn}: stress={self.child.internal_state['stress_level']} engagement={self.child.internal_state['engagement']}"
            )
            logging.info(f"-------------Doctor Observe Start(Turn {turn})-------------")
            perception = self.doctor.observe(
                topic=self.topic,
                history=self.memory.dialogue_history,
                child_response=child_response,
                doctor_last_action=doctor_action
            )
            child_response.update(
                {"observe_type": perception.get('response_type', None) or child_response['type']}
            )
            self.memory.add_dialogue(DialogueTurn(**child_response))
            logging_dialogue(self.memory.dialogue_history)
            # logging.info(f"Observation details: {perception.get('detailed_observation', '')}")
            logging.info(f"Observation details: {perception}")
            logging.info(f"-------------Doctor Observe End(Turn {turn})-------------")
            prior_action: Optional[Dict[str, Any]] = None
            action_count = 0
            while True:
                logging.info(f"-------------Doctor Think Start(Turn {turn}, Action Count {action_count})-------------")
                thought = self.doctor.think(
                    topic=self.topic,
                    child_response=child_response,
                    observation=perception,
                    current_phase=self.mode,
                    history=self.memory.dialogue_history,
                    prior_action=prior_action,
                    turn_number=turn,
                )
                logging_dialogue(self.memory.dialogue_history)
                logging.info(f"Doctor thought actions_just_taken: {json.dumps(self.doctor.memory.strategies.get(turn, []), ensure_ascii=False)}")
                logging.info(f"Doctor thought strategy: {thought.get('strategy', '')}")
                logging.info(f"Doctor thought take action: {thought.get('take_action', '')}")
                logging.info(f"Doctor thought step: {thought.get('step', '')}, action length: {len(self.doctor.memory.strategies.get(turn, []))}")
                logging.info(f"Doctor thought step_evidence: {thought.get('step_evidence', '')}")
                logging.info(f"Doctor thought CoT: {thought.get('chain_of_thought', '')}")
                logging.info(f"-------------Doctor Think End(Turn {turn}, Action Count {action_count})-------------")
                self.log.append({"turn": turn, "observation": perception, "thought": thought})
                if not thought.get("take_action", True):
                    break
                logging.info(f"-------------Doctor Action Start(Turn {turn}, Action Count {action_count})-------------")
                doctor_action = self.doctor.act(
                    thought,
                    topic=self.topic,
                    history=self.memory.dialogue_history,
                    turn_number=turn,
                    doctor_style=self.doctor_style,  # 可以传入特定的医生对话风格
                )
                logging.info(f"Doctor action strategy: {doctor_action.get('strategy', '')}")
                logging.info(f"Doctor action content: {doctor_action.get('content', '')}")
                logging.info(f"-------------Doctor Action End(Turn {turn}, Action Count {action_count})-------------")
                self.memory.add_dialogue(DialogueTurn(**doctor_action))
                doctor_action["turn_number"] = turn
                if doctor_action.get("strategy"):
                    self.child.record_strategy(turn, doctor_action.get("strategy"))
                prior_action = doctor_action
                # bigram 判断儿童是否打断医生（仅第一次医生行动后）
                # if action_count == 0:
                bigram = self.child.bigram or {}
                child_after_prob = bigram.get("child_after_prob", 0.0)
                logging.info(f"-------------Child Disturb Process(Turn {turn})-------------")
                prob = random.random()
                logging.info(f'Disturb prob: {prob}; child_after_prob: {child_after_prob}')
                if prob >= child_after_prob:
                    logging.info(f'Child DO NOT Disturb Doctor...')
                elif prob < child_after_prob:
                    logging.info(f'Child Disturb Doctor !!!')
                    self.memory.interruptions.append({"turn": turn, "by": "child"})
                    break
                action_count += 1
                if action_count >= 3:
                    break
                perception = {**perception, "last_action": doctor_action}
        session_data = {
            "topic": self.topic,
            "mode": self.mode,
            "profile": self.child.profile,
            "memory": self.memory.export(),
            "thoughts": self.log,
        }
        self._persist_session(session_data)
        return session_data

    def _persist_session(self, session_data: Dict[str, Any]) -> None:
        child_profile = getattr(self.child, "profile", {}) if hasattr(self, "child") else {}
        child_name = child_profile.get("full_name", "child")
        child_type = child_profile.get("type", "unknown")
        base_dir = self.output_dir
        safe_topic = str(self.topic).replace("/", "_")
        path = base_dir / self.doctor.llm.model_name / child_type / child_name / f"{safe_topic}.jsonl"
        os.makedirs(path.parent, exist_ok=True)
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(session_data, ensure_ascii=False))
            file.write("\n")
        logging.info(f"Session saved to {path}")
        self.last_saved_path = str(path)


def _load_child_type_map(path: str | Path) -> Dict[str, str]:
    path = Path(path)
    mapping: Dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.warning(f"Child type mapping file not found: {path}")
        return mapping
    except json.JSONDecodeError:
        logging.warning(f"Failed to parse child type mapping file: {path}")
        return mapping

    for type_key, info in data.items():
        if not isinstance(info, dict):
            continue
        name = info.get("姓名") or info.get("name") or info.get("小名")
        child_type = info.get("类型") or type_key
        if name:
            mapping[name] = child_type
        nickname = info.get("小名")
        if nickname:
            mapping[nickname] = child_type
    return mapping


def _build_child_profile(
    child_name: str,
    type_map: Dict[str, str],
    profile_path: str | Path | None = None,
) -> Dict[str, Any]:
    profile_path = Path(profile_path or os.getenv("ASDAGENT_PROFILE_PATH", PROFILE_ROOT / "child_profiles.sample.jsonl"))
    matched: Optional[Dict[str, Any]] = None
    with profile_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("姓名") == child_name or entry.get("小名") == child_name:
                matched = entry
                break
    if not matched:
        raise ValueError(f"Child '{child_name}' not found in {profile_path}")

    profile: Dict[str, Any] = {
        "name": matched.get("小名") or matched.get("姓名") or child_name,
        "gender": matched.get("性别", "男"),
        "age": _age_to_years(matched.get("年龄") or matched.get("现在的年龄") or ""),
        "verbal_level": _age_to_years(
            matched.get("语言生长发育年龄") or matched.get("现在的语言生长发育年龄") or ""
        ),
        "dialogue_history": "",
        "file_infos": matched.get("file_infos", []),
        "type": type_map.get(child_name, "unknown"),
        "full_name": matched.get("姓名") or child_name
    }
    profile["child_stats"] = {
        "last_probs": matched.get("last_probs", {}),
        "sequential_probs": matched.get("sequential_probs", {}),
    }
    # profile["bigram"] = {
    #     "child_after_prob": matched.get("child_after_prob", {}),
    #     "doctor_after_prob": matched.get("doctor_after_prob", {}),
    # }
    profile["bigram"] = matched.get("bigram", {})
    history_file = None
    if profile["file_infos"]:
        history_file = profile["file_infos"][0].get("file_name")
    if history_file:
        profile["dialogue_history"] = _load_dialogue_text(history_file)
    return profile


def _load_topics(topics_path: str | Path) -> List[str]:
    topics_path = Path(topics_path)
    try:
        with topics_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.warning(f"Topics file not found: {topics_path}")
        return []
    except json.JSONDecodeError:
        logging.warning(f"Failed to parse topics file: {topics_path}")
        return []
    topics = data.get("topics", [])
    return [t for t in topics if isinstance(t, str) and t.strip()]


def _load_child_names(profile_path: str | Path) -> List[str]:
    profile_path = Path(profile_path)
    names: List[str] = []
    try:
        with profile_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                nm = entry.get("小名") or entry.get("姓名")
                if nm:
                    names.append(nm)
    except FileNotFoundError:
        logging.warning(f"Child profile file not found: {profile_path}")
    return names


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run TinyABA synthesized dialogue generator.")
    parser.add_argument("--child_name", help="Target child name for dialogue synthesis (overrides random pick).")
    parser.add_argument("--doctor_name", default="S1")
    parser.add_argument("--alpha", type=float, default=0.3, help="Weight for global child stats (0-1).")
    parser.add_argument(
        "--topics-path",
        default=str(DATA_ROOT / "topics.sample.json"),
        help="Path to topics.json",
    )
    parser.add_argument(
        "--profile-path",
        default=str(PROFILE_ROOT / "child_profiles.sample.jsonl"),
        help="Path to anonymized child profile statistics JSONL.",
    )
    parser.add_argument(
        "--child-type-path",
        default=str(PROFILE_ROOT / "child_type_map.sample.json"),
        help="Path to child type/category mapping JSON.",
    )
    parser.add_argument(
        "--global-stats-path",
        default=str(PROFILE_ROOT / "child_global_stats.sample.json"),
        help="Path to global behavior statistics JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_ROOT / "json_files"),
        help="Directory for generated JSONL sessions.",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=None,
        help="Number of child turns. If omitted, sample from global statistics.",
    )

    args = parser.parse_args()

    child_type_map = _load_child_type_map(args.child_type_path)

    available_children = _load_child_names(args.profile_path)
    if not available_children and not args.child_name:
        raise ValueError("No child names available and no child_name provided.")

    logging.info(f'Loading topics from {args.topics_path}')
    topics = _load_topics(args.topics_path)
    if not topics:
        raise ValueError(f"No topics found in {args.topics_path}")

    fallback_turns = 4
    turns = args.turns or _sample_turns_from_stats(args.global_stats_path, fallback_turns)
    logging.info(f'Turn: {turns}')
    # turns = 4
    # turns = fallback_turns

    for topic in topics:
        # pick child
        child_name = args.child_name or random.choice(available_children)
        child_profile = _build_child_profile(child_name, child_type_map, profile_path=args.profile_path)
        child_profile["alpha"] = args.alpha
        doctor_dialogue_style = child_profile.get("dialogue_history", "")

        world = World(
            topic=topic,
            mode="DTT",
            child_profile=copy.deepcopy(child_profile),
            doctor_style=doctor_dialogue_style,
            doctor_name=args.doctor_name,
            alpha=args.alpha,
            global_stats_path=args.global_stats_path,
            output_dir=args.output_dir,
        )
        logging.info(f"Doctor Using model: {world.doctor.llm.model_name}")
        logging.info(f"Child Using model: {world.child.llm.model_name}")
        session = world.run_session(turns=turns)
        logging.info(
            json.dumps(
                {"topic": topic, "saved_path": getattr(world, "last_saved_path", "")},
                ensure_ascii=False,
            )
        )

if __name__ == "__main__":
    # python -m asdagent.tinyaba.main
    main()
