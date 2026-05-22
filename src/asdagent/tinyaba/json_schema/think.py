from pydantic import BaseModel, Field
from typing import List, Literal

class ThoughtProcess(BaseModel):
    step: int = Field(..., description="思考步骤编号")
    step_evidence: str = Field(..., description="支持该步骤的证据或理由")
    chain_of_thought: List[str] = Field(..., description="步骤链，描述决策过程")
    strategy: Literal["指令", "强化", "半辅助", "其他", "全辅助", ""] = Field(..., description="选择的策略,如指令、强化、半辅助、全辅助、其他或空字符串表示无策略")
    reason: str = Field(..., description="选择策略的理由，引用ABA原则")
    take_action: bool = Field(..., description="是否采取行动，true表示需要执行下一步动作，false表示等待儿童反应或不采取行动")