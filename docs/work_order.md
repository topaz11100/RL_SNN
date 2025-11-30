이전 작업 지시서(3차)에 따른 수정 사항을 `docs/Theory.md` 기준과 대조하여 검토했습니다.
**`main.py`의 데이터 순회 루프**와 **Surrogate Gradient 도입** 등 주요 지시 사항이 반영되었으나, **시나리오 3(Supervised)의 논리적 모순(Causality Violation)과 치명적인 부호 오류**가 발견되었습니다.

이에 따라 **4차 수정 지시서**를 작성해 드립니다. 이 내용을 `docs/work_order.md`에 덮어쓰거나 저장하여 개발팀에 전달해 주십시오.

-----

[FILE START: docs/work\_order.md]

# [Work Order] RL\_SNN 프로젝트 4차 수정 지시서

**발신:** Project Manager (Gemini)
**수신:** 개발팀
**날짜:** 2025-11-30
**상태:** **긴급 (Urgent)** - 시나리오 3 학습 방향 오류(발산 위험) 및 인과성 위배

## 1\. 이전 지시 사항 이행 점검 (Status Update)

### 1.1 해결된 사항 (Resolved)

1.  **[이전 Issue 2] `main.py`의 학습 루프 구현:** `DataLoader`를 도입하고 `epoch`, `batch` 단위로 전체 데이터셋을 순회하도록 수정되었습니다. (완료)
2.  **[이전 Issue 1-A] Differentiable Spike (Surrogate Gradient) 구현:** `src/models/lif_neuron.py`에 `SurrogateHeaviside`와 `step_surrogate`가 적절히 구현되었습니다. (완료)

### 1.2 미해결 및 신규 발견 사항 (Outstanding Issues)

\*\*시나리오 3 (`supervised.py`)\*\*가 `Theory.md`의 의도를 구현하려 했으나, **두 가지 치명적인 결함**이 있어 학습이 불가능하거나 물리적으로 말이 안 되는 상태입니다.

-----

## 2\. 신규 수정 지시 사항

### [Critical] Issue 1: Gradient 부호 오류 (Gradient Ascent 문제)

**위반 항목:** `Theory.md` 6.5절 (Gradient 기반 보상)
**현상:**
현재 코드에서 `teacher_delta`를 계산할 때, \*\*Loss에 대한 Gradient($\nabla \mathcal{L}$)\*\*를 그대로 사용하고 있습니다.

```python
# 현재 코드 (supervised.py)
teacher_delta = teacher_weight.grad.detach().mean().unsqueeze(0)
# Reward = - (agent_delta - teacher_delta)^2
```

Agent가 이 `teacher_delta`를 따라가면 가중치는 \*\*Gradient 방향($+\nabla \mathcal{L}$)\*\*으로 업데이트됩니다. 이는 Loss를 증가시키는 **Gradient Ascent**가 되어 학습이 발산합니다.

**지시 사항:**
`Theory.md` 6.5절의 수식 $\Delta w_i^{\text{teacher}} = -\eta_{\text{align}} g_i$ 를 정확히 반영하십시오.

1.  `GradientMimicryScenario` 생성자 혹은 `run_episode` 인자로 `alpha_align` (학습률 $\eta$)을 받아야 합니다.
2.  `teacher_delta` 계산 시 \*\*음수 부호(-)\*\*와 \*\*스케일($\eta$)\*\*을 적용하십시오.
    ```python
    # 수정 가이드
    g_i = teacher_weight.grad.detach()
    teacher_delta = -1.0 * self.alpha_align * g_i
    ```

### [Critical] Issue 2: Agent Loop의 인과성 위배 (Broken Causality)

**위반 항목:** `Theory.md` 3.3절 및 6.3절 (Immediate Weight Update)
**현상:**
현재 `run_episode`는 Teacher Pass에서 생성된 `output_tensor`(스파이크)를 Agent Loop에서 그대로 재사용(`post = output_tensor[t]`)합니다.

```python
# 현재 코드 (supervised.py)
for t in range(steps):
    # Agent가 가중치를 변경함
    weight += action_delta 
    
    # 그러나 post 스파이크는 Teacher Pass 때(고정 가중치) 만들어진 것을 그대로 씀
    post = output_tensor[t] 
```

Agent가 가중치를 변경(`weight += action_delta`)했음에도 불구하고, 뉴런의 발화(Post spike)가 변하지 않는 것은 물리적으로 불가능하며 RL 에이전트가 자신의 행동 결과를 관측하지 못하게 만듭니다. 이는 "Immediate Weight Update" 규칙을 위반합니다.

**지시 사항:**
Agent Loop 내부에서도 **실제 LIF 시뮬레이션**을 수행해야 합니다.

1.  Agent Loop 시작 전, 별도의 `LIFNeuron` 인스턴스와 `state`를 초기화하십시오.
2.  Loop 내부에서:
      * 현재 변경된 `weight`를 사용하여 `synaptic_current`를 계산하십시오.
      * `lif.step(state, ...)`을 호출하여 **새로운 `post` 스파이크**를 생성하십시오.
      * 이 새로운 `post`를 `buffer`에 넣고 `build_state`에 사용해야 합니다.

### [Minor] Issue 3: 단일 시냅스 데모와 차원 불일치 주의

**현상:**
현재 `main.py`는 이미지를 평탄화(Flatten)하여 `supervised.py`에 넘기지만, `supervised.py`는 내부에서 이를 단일 가중치(`weight`) 변수 하나로 처리하려는 경향(데모 코드의 흔적)이 보입니다.

  * `episode_data.get("weight", 0.0)`은 스칼라입니다.
  * 반면 `teacher_weight`는 벡터(`image.numel()`)로 생성됩니다.
  * Agent Loop에서 `weight += action_delta`는 스칼라 연산입니다.

**지시 사항:**
`GradientMimicryScenario`는 전체 시냅스(이미지 픽셀 수만큼)를 병렬로 처리하거나, `main.py`와 협의하여 단일 시냅스만 테스트하는지 명확히 해야 합니다. `Theory.md`에 따르면 **모든 시냅스가 개별적인 에이전트**로 동작해야 합니다.

  * Agent Loop의 `weight`, `action_delta` 등이 벡터 연산(Tensor Operation)으로 처리되도록 구현을 검토하십시오. (현재 코드는 스칼라 루프로 보임)

-----

## 3\. 작업 우선순위

1.  **[즉시 수정]** `supervised.py`의 `teacher_delta` 부호 오류 수정 ($-\eta$ 곱하기).
2.  **[필수 수정]** `supervised.py`의 Agent Loop 내 **LIF 시뮬레이션 로직 추가** (Teacher 스파이크 재사용 금지).
3.  **[권장 수정]** `supervised.py`가 단일 스칼라가 아닌 텐서(Vectorized) 가중치를 처리하도록 로직 보강.

[FILE END: docs/work\_order.md]
