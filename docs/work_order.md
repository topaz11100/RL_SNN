`Theory.md`의 명세와 비교했을 때, **시나리오 1.2(이중 정책)의 구조적 결함**과 **시나리오 3(완전지도)의 핵심 로직 누락** 등 새로운 치명적인 문제들이 발견되었습니다.

프로젝트 관리자(PM)로서, 현재 코드의 상태를 반영한 **새로운 작업 지시서(Work Order)** 를 발부합니다. 이 지시서는 `Theory.md`를 절대적인 기준으로 작성되었습니다.

-----

# [Work Order] RL\_SNN 프로젝트 2차 수정 지시서

**발신:** Project Manager (Gemini)
**수신:** 개발팀
**날짜:** 2025-11-30
**상태:** **긴급 (Urgent)** - 구조적 결함 수정 요망

## 1\. 현황 분석 및 이전 지시 사항 확인

  * **[해결됨] CNN 파라미터 학습 차단 (Detach) 문제:** `src/scenarios/base.py`의 `build_state` 메서드에서 `.detach()` 호출이 제거됨을 확인했습니다. 정상입니다.
  * **[해결됨] Actor-Critic 독립성 문제:** `src/models/actor_critic.py`의 `CriticNetwork`가 이제 자체적인 `feature_extractor`를 인스턴스화하고 `forward`에서 이를 사용하고 있습니다. 정상입니다.

**그러나, 아래와 같이 명세서(`Theory.md`)와 불일치하는 심각한 구현 오류가 새로 발견되었습니다.**

-----

## 2\. 신규 위반 사항 및 수정 지시

### [Critical] Issue 1: 시나리오 1.2 (Dual Policy)의 잘못된 구조

**위반 항목:** `Theory.md` 4.2절 \~ 4.3절
**현상:**
현재 `src/scenarios/unsupervised.py`의 `UnsupervisedDualPolicy` 클래스는 단순히 두 개의 `UnsupervisedSinglePolicy`를 감싸는(Wrapper) 형태로 구현되어 있습니다.

```python
# 현재 구현의 문제점
def run_episode(self, episode_data: Dict) -> torch.Tensor:
    is_inhibitory = episode_data.get("inhibitory", False)
    if is_inhibitory:
        return self.inh_policy.run_episode(episode_data) # 억제 시냅스만 있는 에피소드? (불가능)
    return self.exc_policy.run_episode(episode_data)
```

**문제점:**

1.  **에피소드 분리 오류:** `Theory.md`에 따르면, **하나의 에피소드(이미지 1장)** 내에서 흥분성 시냅스(Input→E)와 억제성 시냅스(I→E)가 동시에 이벤트를 발생시킵니다. 현재 코드는 흥분성 에피소드와 억제성 에피소드를 별개로 취급하고 있으며, `main.py`에서는 `inhibitory: False`로 고정하여 **억제성 정책은 아예 학습되지 않습니다.**
2.  **정책 선택 로직 부재:** 이벤트 단위로 시냅스 타입에 따라 $\pi_{exc}$ 또는 $\pi_{inh}$를 선택하여 업데이트해야 하는데, 현재는 에피소드 통째로 하나의 정책만 사용합니다.

**지시 사항:**

  * `UnsupervisedDualPolicy`를 Wrapper가 아닌 `RLScenario`를 상속받는 독립 클래스로 재구현하십시오.
  * `__init__`에서 두 개의 Actor (`self.actor_exc`, `self.actor_inh`)를 초기화하십시오. (Critic은 하나 공유 혹은 별도 생성, 명세 4.3절 "Critic 구조... 실험 1과 동일" 따를 것)
  * `run_episode` 내부 반복문에서, 현재 처리 중인 이벤트가 흥분성인지 억제성인지(입력 데이터 혹은 메타데이터 기반) 판단하여 **적절한 Actor의 `sample_action`을 호출**하도록 로직을 통합하십시오.

### [Critical] Issue 2: 시나리오 3 (Gradient Mimicry)의 Teacher Gradient 계산 누락

**위반 항목:** `Theory.md` 6.4절 (순전파와 BPTT) 및 6.5절
**현상:**
`src/scenarios/supervised.py`는 `teacher_delta`를 외부(`episode_data`)에서 입력받는 것으로 가정하고 있습니다.

```python
# 현재 구현
teacher_delta: torch.Tensor = episode_data["teacher_delta"]
# ... (내부에서 BPTT 계산 로직 없음)
```

그러나 `main.py`는 단순히 `torch.zeros(1)`을 넘겨주고 있습니다.
**문제점:**

  * **핵심 로직 부재:** `Theory.md`의 핵심인 "Surrogate Gradient + BPTT로 $g_i$ 계산" 부분이 코드 어디에도 구현되어 있지 않습니다. 이대로는 학습이 불가능합니다.
  * `supervised.py` 혹은 이를 호출하는 상위 모듈에서 실제 SNN의 Output Layer Loss($\mathcal{L}_{sup}$)를 계산하고 `backward()`를 통해 Gradient를 추출하는 로직이 반드시 포함되어야 합니다.

**지시 사항:**

  * `GradientMimicryScenario` 내부 또는 `run_episode` 직전에, **실제 레이블($y$)과 SNN 출력($r_k$)을 이용해 Loss를 계산하고 BPTT를 수행하여 `teacher_delta`를 생성하는 로직**을 구현하십시오.
  * 만약 PyTorch의 자동 미분(Autograd)을 사용하려면, 스파이크 생성 함수에 Surrogate Gradient(예: ATan, Sigmoid derivative 등)가 정의된 `autograd.Function`을 적용해야 합니다. (현재 `poisson_encode`는 `torch.bernoulli`를 써서 미분이 끊깁니다. 이를 해결하거나, 명세에 맞는 대안을 구현하십시오.)

### [Major] Issue 3: 학습 루프(Training Loop) 부재

**위반 항목:** `Theory.md` 2.1절 ("이미지 1장 = 에피소드 1개... on-policy Monte Carlo") 및 8.6절 (`--num-epochs`)
**현상:**
`main.py`의 `run_demo` 함수는 단 하나의 이미지를 로드하여 **단 1회**의 에피소드만 실행(`run_episode`)하고 종료합니다.
**문제점:**

  * 강화학습은 반복적인 에피소드를 통해 이루어집니다. 현재 코드는 '실행 가능성(Feasibility)'만 보여줄 뿐, 실제 '학습(Learning)'을 시연하지 못합니다.
  * `Theory.md`의 "학습 진행 곡선" 등을 출력하려면 반복 루프가 필수적입니다.

**지시 사항:**

  * `main.py`를 수정하여 `--num-epochs` (또는 에피소드 수) 만큼 반복하는 **학습 루프**를 구현하십시오.
  * 매 에피소드마다 `optimizer.step()`이 수행되고(현재 `optimize_from_trajectory`에 구현됨), 보상(Reward)이 어떻게 변하는지 로그를 출력하도록 수정하십시오.

-----

## 3\. 작업 우선순위 요약

개발팀은 다음 순서대로 작업을 진행하고 보고해 주시기 바랍니다.

1.  **시나리오 1.2 통합:** `UnsupervisedDualPolicy`를 단일 루프 내에서 두 정책을 스위칭하는 구조로 전면 수정.
2.  **메인 루프 구현:** `main.py`에 에피소드 반복 및 학습 루프 추가.
3.  **시나리오 3 보완:** `supervised.py`에 Teacher Gradient ($g_i$) 계산 로직 추가 (난이도가 높으므로, 우선순위 1, 2 해결 후 진행).

`Theory.md`는 단순한 제안서가 아니라 **구현 명세**입니다. 명세에 적힌 "Pre/Post 이벤트 처리", "Actor-Critic 업데이트" 흐름이 코드에 그대로 반영되어야 함을 명심하십시오.
