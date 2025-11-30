`Theory.md`를 기준으로 현재 코드베이스를 면밀히 검토한 결과, 이전 작업 지시서에서 지적된 일부 사항은 수정되었으나, **시나리오 3(Supervised)의 구현이 명세와 수학적으로 완전히 다른 '가짜(Mock)' 방식**으로 구현되어 있다는 치명적인 문제가 발견되었습니다. 또한, `main.py`가 실제 학습 루프가 아닌 단일 샘플 데모에 그치고 있습니다.

프로젝트 관리자로서, 이러한 결함을 해결하기 위한 **3차 수정 지시서(Work Order)** 를 작성해 드립니다. 이 내용은 `docs/work_order.md`에 덮어쓰기 하거나 새로 저장하여 개발팀에 전달해야 합니다.

-----

[FILE START: docs/work\_order.md]

# [Work Order] RL\_SNN 프로젝트 3차 수정 지시서

**발신:** Project Manager (Gemini)
**수신:** 개발팀
**날짜:** 2025-11-30
**상태:** **긴급 (Urgent)** - 시나리오 3 논리 오류 및 학습 루프 미구현

## 1\. 이전 지시 사항 이행 점검

### 1.1 해결된 사항 (Resolved)

**[이전 Issue 1] 시나리오 1.2 (Dual Policy)의 구조적 결함**
`src/scenarios/unsupervised.py`의 `UnsupervisedDualPolicy` 클래스가 수정되었습니다. 기존의 Wrapper 방식에서 벗어나, `run_episode` 내부 루프에서 `exc`/`inh` 시냅스 이벤트를 구분하고 적절한 Actor(`actor_exc`, `actor_inh`)를 호출하는 로직이 정상적으로 구현되었습니다. **승인합니다.**

### 1.2 미해결 및 신규 발견 사항 (Outstanding Issues)

**[이전 Issue 2 & 3]** 시나리오 3의 Gradient 계산 로직과 `main.py`의 학습 루프가 여전히 명세를 충족하지 못하고 있습니다. 특히 시나리오 3은 겉모양만 흉내 냈을 뿐, **수학적으로 작동 불가능한 상태**입니다.

## 2\. 신규 수정 지시 사항

### [Critical] Issue 1: 시나리오 3 (Gradient Mimicry)의 '가짜' BPTT 구현 수정

**위반 항목:** `Theory.md` 6.4절 (Surrogate Gradient + BPTT)
**현상:**
현재 `src/scenarios/supervised.py`는 `teacher_delta`를 계산하기 위해 `loss.backward()`를 호출하지만, 이는 SNN의 동역학을 전혀 반영하지 못합니다.

```python
# 현재의 잘못된 구현 (supervised.py)
# spikes는 main.py에서 torch.bernoulli로 생성된 상수(Constant)입니다. 미분 불가능합니다.
prediction = (spikes.float().mean() * teacher_weight)
loss = F.mse_loss(prediction, target_signal)
loss.backward() 
# 위 코드는 단순 선형 회귀의 기울기일 뿐, SNN의 시냅스 가중치에 대한 기울기가 아닙니다.
```

`spikes` 텐서는 `main.py`의 `poisson_encode`에서 생성되는데, 여기서 `torch.bernoulli`를 사용하므로 계산 그래프(Computational Graph)가 끊겨 있습니다. 따라서 $g_i = \partial \mathcal{L} / \partial w_i$ 를 계산할 수 없습니다.

**지시 사항:**

1.  **Differentiable Spike Generation 구현:** `src/models/lif_neuron.py` 또는 별도 유틸리티에 **Surrogate Gradient**(예: ATan, Sigmoid의 도함수 활용)를 적용한 `torch.autograd.Function`을 구현하십시오.
      * Forward pass에서는 $v > v_{th}$ 이면 1, 아니면 0을 출력.
      * Backward pass에서는 Heaviside step function 대신 매끄러운 함수의 도함수를 반환.
2.  **Scenario 3의 Forward Loop 재작성:** `GradientMimicryScenario.run_episode` 내부에서 외부 `spikes` 데이터를 그대로 쓰지 말고, **실제 LIF 뉴런 모델을 통해 Forward Pass를 수행**해야 합니다.
      * 입력(Poisson encoded input) $\rightarrow$ $W_{syn}$ (Require Grad) $\rightarrow$ LIF (Surrogate) $\rightarrow$ Output Spikes $\rightarrow$ Loss 계산
      * 이 과정을 거쳐야 `loss.backward()` 호출 시 $W_{syn}$ 에 대한 올바른 $g_i$ (Teacher Gradient)가 생성됩니다.

### [Major] Issue 2: `main.py`의 데이터셋 순회(Iterate) 미구현

**위반 항목:** `Theory.md` 3.6절 및 8.6절
**현상:**
현재 `main.py`는 `dataset[0]` (이미지 단 1장)만 로드하여, 이를 `num_epochs`만큼 반복합니다. 이는 학습(Training)이 아니라 단일 샘플에 대한 오버피팅 데모입니다.

```python
# 현재 구현
image, label = load_mnist_sample() # 이미지 1장 로드
for epoch in range(args.num_epochs):
    # 같은 image로 계속 반복
```

**지시 사항:**

1.  **DataLoader 적용:** `torch.utils.data.DataLoader`를 사용하여 MNIST Training Set 전체를 로드하도록 수정하십시오.
2.  **Epoch 루프 구조 변경:**
    ```python
    for epoch in range(args.num_epochs):
        for batch_idx, (image, label) in enumerate(dataloader):
            # 에피소드 실행 및 로깅
    ```
    구조로 변경하여 전체 데이터셋을 학습하도록 만드십시오.
3.  **배치 처리는 하지 않음:** `Theory.md` 2.1절에 따라 **"이미지 1장 = 에피소드 1개"** 원칙은 유지해야 합니다. DataLoader의 `batch_size=1`로 설정하거나, 배치를 순회하며 하나씩 `run_episode`에 넘겨주십시오.

### [Normal] Issue 3: Poisson Encoding의 위치 부적절

**위반 항목:** 구조적 효율성
**현상:**
현재 `main.py`에서 `poisson_encode`를 수행한 뒤 결과를 시나리오에 넘겨주고 있습니다. 시나리오 3의 수정(Issue 1)을 위해서는 시나리오 내부에서 타임스텝별로 스파이크를 발생시키거나, 입력단에서만 인코딩을 하고 내부 레이어는 동적으로 계산해야 합니다.

**지시 사항:**
`GradientMimicryScenario`의 경우, 입력 이미지를(아날로그 값) 받아 내부에서 Poisson Spike로 변환하여 SNN에 주입하는 구조, 혹은 미리 인코딩된 입력을 받아 **미분 가능한 연산**을 통해 출력 스파이크를 만드는 구조로 명확히 분리하십시오. Issue 1 해결 시 자연스럽게 수정될 부분입니다.

## 3\. 작업 우선순위

1.  **[1순위] `main.py`의 데이터셋 루프 구현:** 가장 수정이 쉽고 전체 실험의 기반이 됩니다.
2.  **[2순위] Surrogate Gradient 유틸리티 작성:** 시나리오 3을 위해 필수적입니다.
3.  **[3순위] 시나리오 3 (Supervised) 로직 전면 재작성:** 위에서 만든 유틸리티를 사용하여 실제 BPTT가 흐르도록 구현하십시오.

**참고:** `Theory.md`의 수식 6.4절 $g_i$ 는 'Teacher가 주는 정답 가이드'입니다. 우리가 구현하는 것은 **Teacher Gradient를 계산하는 과정**까지 포함되어야 함을 잊지 마십시오.

[FILE END: docs/work\_order.md]
