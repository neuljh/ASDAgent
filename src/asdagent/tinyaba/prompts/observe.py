

OBSERVE_USER_PROMPT_TEMPLATE_FEW_SHOT = """
## 示例

### 示例对话1

对话主题：宇航员与太空

医生：你好小明，今天我们来学习宇航员与太空。准备好了吗？（指令）
儿童：飞船。（不相关的回答）
医生：哦，飞船是宇航员用来去太空的工具，真不错！（其他）
医生：你知道宇航员在太空中会做些什么吗？（指令）
儿童：飞行。（相关的回答）
医生：哇！你说得太对了，宇航员在太空中会飞行，真棒！（强化）
医生：那你觉得宇航员在太空中还可以做些什么呢？比如他们会做一些实验吗？（指令）
儿童：做实验。（相关的回答）
医生：没错，宇航员在太空中确实会做实验，真棒！（强化）
医生：那宇航员在太空中还可以做些什么呢？除了实验，他们会观察什么呢？（指令）
儿童：观察星星。（相关的回答）
医生：很好，宇航员在太空中会观察星星，真棒！（强化）
医生：宇航员除了观察星星，还可以观察到宇宙中的其他东西，比如行星，你觉得他们还会观察到什么呢？（指令）
儿童：还有月亮。（相关的回答）
医生：哇！你说得太对了，还有月亮！（强化）
医生：那你觉得宇航员在太空中还可以观察到哪些有趣的东西呢？（指令）


**儿童回复1**: 星星和月亮。
**医生观察1**:
    输出：
    {{
        "response_type": "不相关的回答" ,
        "reason": "在历史对话中，儿童已经回答了“星星和月亮”，这一轮的回答“星星和月亮”与之前医生给出相同指令“那你觉得宇航员在太空中还可以观察到哪些有趣的东西呢？”下儿童的回答内容“观察星星”和“还有月亮”相同，因此属于不相关的回答。",
        "quality_assessment": "在之前的指令中，儿童的回答与指令内容无关，表现出不相关的回答行为。",
        "semantic": {{
            "sign": true,
            "reason": "儿童的回答“星星和月亮”是对医生指令的具体回应，具有明确的语义内容。"
        }},
        "emotional_state": {{
            "stress": "Low", 
            "engagement": "High"
        }},
        "detailed_observation": "在历史对话中，儿童已经回答了“星星和月亮”，这一轮的回答与之前的内容重复，因此属于不相关的回答。儿童可能对当前任务感到厌倦或缺乏兴趣，建议调整教学策略以提高儿童的参与度。"
    }}

**儿童回复2**: 我要吃糖！
**医生观察2**:
    输出：
    {{
        "response_type": "不相关的回答" ,
        "reason": "儿童的回答“我要吃糖！”与医生的指令内容“那你觉得宇航员在太空中还可以观察到哪些有趣的东西呢？”无关，表现出不相关的回答行为。",
        "quality_assessment": "在之前的指令中，儿童的回答与指令内容无关，表现出不相关的回答行为。",
        "semantic": {{
            "sign": true,
            "reason": "儿童的回答“我要吃糖！”是对医生指令的具体回应，具有明确的语义内容。"
        }},
        "emotional_state": {{
            "stress": "Medium", 
            "engagement": "Low"
        }},
        "detailed_observation": "儿童无视了医生指令，转而发起对强化物（糖）的索取。这表明儿童当下的动机在糖果上，而非任务上。可能存在对任务的逃避或注意力分散。"
    }}

**儿童回复3**: 我不知道。
**医生观察3**:
    输出：
    {{
        "response_type": "相关的回答" ,
        "reason": "儿童的回答“我不知道”未能提供与医生指令“那你觉得宇航员在太空中还可以观察到哪些有趣的东西呢？”相关的信息，但是贴合了医生的问题意图，儿童回答不知道是正常，因此被视为相关的回答。",
        "quality_assessment": "儿童的回答虽然未能提供具体信息，但符合医生的问题意图，表现出一定的相关性。",
        "semantic": {{
            "sign": true,
            "reason": "儿童的回答“我不知道”是对医生指令的具体回应，具有明确的语义内容。"
        }},
        "emotional_state": {{
            "stress": "Low", 
            "engagement": "Low"
        }},
        "detailed_observation": "儿童未能提供具体信息，但符合医生的问题意图，表现出一定的相关性。儿童可能对任务感到困惑或缺乏信心，建议提供更多支持和引导以帮助儿童理解任务要求。"
    }}

**儿童回复4**: 太阳。
**医生观察4**:
    输出：
    {{
        "response_type": "相关的回答" ,
        "reason": "儿童的回答“太阳”与医生的指令内容“那你觉得宇航员在太空中还可以观察到哪些有趣的东西呢？”相关，表现出相关的回答行为。",
        "quality_assessment": "儿童的回答与指令内容相关，表现出良好的理解和回应能力。",
        "semantic": {{
            "sign": true,
            "reason": "儿童的回答“太阳”是对医生指令的具体回应，具有明确的语义内容。"
        }},
        "emotional_state": {{
            "stress": "Low", 
            "engagement": "High"
        }},
        "detailed_observation": "儿童的回答与指令内容相关，表现出良好的理解和回应能力。儿童显示出对任务的兴趣和积极参与，建议继续使用类似的教学策略以维持儿童的高参与度。"
    }}

**儿童回复5**: [儿童无响应]。
**医生观察5**:
    输出：
    {{
        "response_type": "无响应" ,
        "reason": "儿童对医生指令“那你觉得宇航员在太空中还可以观察到哪些有趣的东西呢？”未作出任何回应，表现出无响应行为。",
        "quality_assessment": "儿童未能回应医生的指令，表现出无响应行为。",
        "semantic": {{
            "sign": false,
            "reason": "儿童未作出任何回应，因此没有语义内容。"
        }},
        "emotional_state": {{
            "stress": "High", 
            "engagement": "Low"
        }},
        "detailed_observation": "儿童未作出任何回应，表现出无响应行为。可能存在对任务的逃避或注意力分散，建议调整教学策略以提高儿童的参与度。"
    }}

**儿童回复7**: 西红柿。
**医生观察7**:
    输出：
    {{
        "response_type": "不相关的回答" ,
        "reason": "对话主题为宇航员与太空，儿童的回答“西红柿”与医生的指令内容“那你觉得宇航员在太空中还可以观察到哪些有趣的东西呢？”无关，表现出不相关的回答行为。",
        "quality_assessment": "儿童的回答与指令内容无关，表现出不相关的回答行为。",
        "semantic": {{
            "sign": true,
            "reason": "儿童的回答“西红柿”是对医生指令的具体回应，具有明确的语义内容。"
        }},
        "emotional_state": {{
            "stress": "Medium", 
            "engagement": "Low"
        }},
        "detailed_observation": "儿童的回答“西红柿”与医生的指令内容无关，表现出不相关的回答行为。儿童可能对当前任务感到厌倦或缺乏兴趣，建议调整教学策略以提高儿童的参与度。"  
    }}

**儿童回复8**: 哦，我我我...星。
**医生观察8**:
    输出：
    {{
        "response_type": "不相关的回答" ,
        "reason": "儿童的回答“哦，我我我...星”与医生的指令内容“那你觉得宇航员在太空中还可以观察到哪些有趣的东西呢？”无关，表现出不相关的回答行为。",
        "quality_assessment": "儿童的回答与指令内容无关，表现出不相关的回答行为。",
        "semantic": {{
            "sign": false, 
            "reason": "儿童的回答“哦，我我我...星”没有明确的语义内容，无法与医生指令建立有效关联。"
        }},
        "emotional_state": {{
            "stress": "Medium", 
            "engagement": "Low"
        }},
        "detailed_observation": "儿童的回答“哦，我我我...星”与医生的指令内容无关并且无语义，表现出不相关的回答行为。儿童可能对当前任务感到厌倦或缺乏兴趣，建议调整教学策略以提高儿童的参与度。"  
    }}

## 输入对话主题：
{topic}

## 输入历史对话：
{history}

## 儿童回复：
{child_response}
"""

OBSERVE_SYS_PROMPT_TEMPLATE_FEW_SHOT = """
## 角色设定
你是一名专业的ABA行为治疗师。你的任务是进行**实时行为观察（Behavioral Observation）**。
你需要分析“儿童的反应”与“医生刚才的指令”之间的关系，并推断儿童的内部状态。

## 核心分类准则 (Classifications)
**请仔细区分“重复”与“相关的回答”：**
1. **相关的回答 (Related)**: 儿童理解了指令，并给出了符合语境的回复。
   * **重要提示**: 如果儿童重复医生提到的**核心名词**来表示确认、关注或回答，这属于【相关的回答】。
   * 例：医生“我们要学宇航员”，儿童“宇航员”。(判为：相关的回答，表示确认主题)
2. **重复 (Repeated)**: 这种行为通常指**回声语言 (Echolalia)**。即儿童机械地、无意识地复述医生的整句或末尾词组，通常带有疑问语调或完全不理解含义。
   * 例：医生“我们要学宇航员”，儿童“要学宇航员”。(判为：重复)
3. **不相关的回答 (Unrelated)**: 答非所问。

## 分析维度
请从以下三个维度进行分析：

1. **响应质量 (Response Quality)**:
   - 儿童的回答和医生上一句说的话有什么关系？
   - 以此来决定儿童的回答类型是什么？（不相关的回答/相关的回答/无响应/重复）

2. **行为功能假设 (Function Hypothesis)**:
   - 儿童为什么会有这个反应？
   - 常见功能：获取实物/关注 (Access)、逃避/回避任务 (Escape)、自我刺激 (Sensory)。
   - 例如：儿童说“我要找妈妈”通常是“逃避”功能。

3. **状态推断 (State Inference)**:
   - **Stress (压力)**: Low (平静) / Medium (焦虑) / High (崩溃边缘)。
   - **Engagement (投入度)**: High (专注) / Medium (分心) / Low (游离)。


## 输出格式 (JSON)
请输出严格的 JSON:
```json
{{
    "response_type": "不相关的回答" | "相关的回答" | "无响应" | "重复",
    "reason": "说明你的判断依据，需要引用医生上个回复和儿童的回复详细分析",
    "semantic": {{
        "sign": true | false,
        "reason": "儿童的回答内容是否有语义的理由"
    }},
    "quality_assessment": "对儿童回复的简短评价",
    "emotional_state": {{
        "stress": "Low" | "Medium" | "High",
        "engagement": "High" | "Medium" | "Low"
    }},
    "detailed_observation": "综合描述，例如：'儿童表现出逃避倾向，可能是因为指令过难导致压力升高。'"
}}
"""

if __name__ == "__main__":
    from pydantic import BaseModel

    class DoctorBehavior(BaseModel):
        strategy: str
        content: str

    class ChildResponse(BaseModel):
        content: str

    class ObservationInput(BaseModel):
        doctor_behavior: DoctorBehavior
        child_response: ChildResponse

    info = ObservationInput(
        doctor_behavior=DoctorBehavior(
            strategy="指令",
            content="把积木放在桌子上面。"
        ),
        child_response=ChildResponse(
            content="我要吃糖！"
        )
    )

    user_prompt = OBSERVE_USER_PROMPT_TEMPLATE_FEW_SHOT.format(info=info.json())
    print("=== User Prompt ===")
    print(user_prompt)

    sys_prompt = OBSERVE_SYS_PROMPT_TEMPLATE_FEW_SHOT
    print("=== System Prompt ===")
    print(sys_prompt)