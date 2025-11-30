분석 결과, 현재 구현된 코드는 `Theory.md`의 전체적인 구조를 따르고 있으나, **학습이 불가능하게 만드는 치명적인 구현 오류**와 **명세서의 엄격한 요건을 위반한 구조적 문제**들이 발견되었습니다.

저는 이 프로젝트의 관리자(PM)로서, `Theory.md`를 절대적 기준으로 삼아 발견된 문제점들을 지적하고, 이를 수정하기 위한 \*\*작업 지시서(Work Order)\*\*를 발부합니다.

-----

# [Work Order] RL\_SNN 프로젝트 수정 지시서

**발신:** Project Manager (Gemini)
**수신:** 개발팀
**날짜:** 2025-11-30
**우선순위:** **Critical** (즉시 수정 요망)

## 1\. 개요 및 현황 평가

현재 코드는 `main.py`를 통한 데모 실행과 모듈화(Scenarios, Models, Utils) 구조는 잘 잡혀 있으나, \*\*RL 에이전트의 학습 메커니즘(Backpropagation)\*\*과 **Actor-Critic의 독립성 원칙**에서 심각한 위배 사항이 발견되었습니다. 이대로는 학습이 전혀 진행되지 않거나 명세서와 다른 동작을 하게 됩니다.

-----

## 2\. 상세 위반 사항 및 수정 지시

### [Critical] Issue 1: CNN 전단 파라미터 학습 차단 (Detach 오용)

**위반 항목:** `Theory.md` 2.4절 및 2.9절 (Actor-Critic 업데이트)
**현상:** `src/scenarios/base.py`의 `build_state` 메서드에서 CNN 특징 추출 후 `.detach()`를 호출하고 있습니다.

```python
features = self.actor.feature_extractor(spike_tensor).detach() # Detach as this is input state
```

**문제점:** `features`를 detach하면 Backpropagation 시 그래디언트가 CNN 레이어까지 흐르지 못합니다. 즉, **CNN 필터는 초기화된 상태에서 영원히 학습되지 않습니다.** CNN은 단순한 전처리기가 아니라 에이전트의 일부로 학습되어야 합니다.
**지시 사항:**

  * `detach()` 호출을 제거하십시오. 그래디언트가 `self.actor.feature_extractor`까지 흘러야 합니다.

### [Critical] Issue 2: Actor와 Critic의 CNN 파라미터 공유 금지 위반

**위반 항목:** `Theory.md` 2.4절 ("정책마다, Critic마다 CNN 전단 파라미터를 공유하지 않는다... 각각 별도의 optimizer로 학습된다")
**현상:**

1.  `src/scenarios/base.py`의 `build_state`에서 `self.actor.feature_extractor`만을 사용하여 상태(features)를 생성합니다.
2.  `src/models/actor_critic.py`의 `CriticNetwork`는 내부적으로 `feature_extractor`를 선언(`__init__`)만 하고, `forward`에서는 외부에서 주입된 `fused_state`를 그대로 사용합니다. 즉, Critic은 자신의 CNN을 쓰지 않고 Actor의 CNN 결과를 입력으로 받습니다.
    **문제점:** Critic의 CNN 파라미터(`critic.feature_extractor`)는 학습 루프에서 완전히 배제되어 있으며, Critic이 Actor의 Feature에 의존하게 됩니다. 이는 독립성을 위반합니다.
    **지시 사항:**

<!-- end list -->

  * `CriticNetwork` 구조 변경: `forward` 메서드가 `fused_state`를 받는 것이 아니라, 원본 스파이크 데이터(`spike_history`)와 메타데이터를 받아 **자신의 CNN을 통과시키도록** 변경하십시오.
  * 또는, `build_state`가 Actor용 state와 Critic용 state를 별도로 리턴하도록 로직을 분리하십시오. (전자 추천)

### [Minor] Issue 3: CLI 파라미터 명세 불일치

**위반 항목:** `Theory.md` 8.1절 \~ 8.4절 (공통 CLI 하이퍼파라미터 정리)
**현상:** `main.py`의 인자들이 명세서와 다릅니다.

  * 구현: `--steps`
  * 명세: `--T-unsup1`, `--T-sup` 등 시나리오별 타임스텝 구분 필요
  * 구현: 클리핑 관련 인자 누락 (`--exc-clip-min` 등)
    **지시 사항:**
  * `argparse` 인자를 `Theory.md` 8장에 정의된 이름(예: `--T-unsup1`, `--exc-clip-min`, `--exc-clip-max`)과 정확히 일치시키십시오.
  * 누락된 클리핑 파라미터를 추가하고 시나리오 클래스에 전달되도록 수정하십시오.

### [Minor] Issue 4: Critic Forward 구조 논리 오류

**위반 항목:** `Theory.md` 2.7절 (Critic 입력 정의)
**현상:** `src/models/actor_critic.py`의 `CriticNetwork.forward`는 `fused_state`를 입력으로 받습니다. 그러나 위 Issue 2에서 지적했듯, `fused_state` 안에 이미 CNN 출력이 포함되어 있다면 Critic 내부의 CNN은 무용지물이 됩니다.
**지시 사항:**

  * `CriticNetwork`도 `ActorNetwork`와 마찬가지로 `spike_history`와 `scalars`(가중치, one-hot 등)를 별도 인자로 받아 내부에서 `feature_extractor`를 통과시킨 후 `mlp`에 넣는 구조가 되어야 합니다.
  * `fuse_state` 유틸리티 함수는 CNN 출력 이후의 결합(Late Fusion)만을 담당하도록 명확히 하십시오.

-----

## 3\. 수정된 코드 예시 (가이드라인)

아래는 **Issue 1, 2, 4**를 해결하기 위해 `src/models/actor_critic.py`와 `src/scenarios/base.py`를 어떻게 고쳐야 하는지에 대한 가이드입니다.

**`src/models/actor_critic.py` 수정안:**

```python
# CriticNetwork 수정: 내부 CNN을 실제로 사용하도록 변경
class CriticNetwork(nn.Module):
    def __init__(self, input_dim_scalars: int): 
        # input_dim_scalars: CNN 출력(16)을 제외한 나머지 스칼라 feature의 차원 (예: 1+2=3)
        super().__init__()
        self.feature_extractor = SpikeHistoryCNN()
        # MLP 입력 차원 = CNN출력(16) + 스칼라 차원
        self.mlp = nn.Sequential(
            nn.Linear(16 + input_dim_scalars, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, spike_history: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        # 1. 자신의 CNN 사용
        cnn_feat = self.feature_extractor(spike_history)
        # 2. Late Fusion
        fused = torch.cat([cnn_feat, scalars], dim=-1)
        # 3. Value 예측
        return self.mlp(fused)
```

**`src/scenarios/base.py` 수정안:**

```python
    def optimize_from_trajectory(self, trajectory: List[TrajectoryEntry], reward: torch.Tensor) -> None:
        # ... (생략) ...
        
        # 중요: Actor와 Critic은 서로 다른 CNN 파라미터를 가지므로,
        # 저장된 fused_state를 재사용하는 것이 아니라,
        # 원본 spike_history를 저장해두었다가 각각 forward를 다시 수행해야 Backprop이 제대로 됨.
        # (현재 구조처럼 fused_state만 저장하면, Critic은 Actor의 CNN gradient에 영향을 줌 - 잘못됨)
        
        # 그러나 현재 코드 구조를 최소한으로 변경하려면:
        # TrajectoryEntry에 'spike_history'와 'scalars'를 별도로 저장해야 함.
        pass 
```

**[PM 코멘트]**: 현재 `base.py`의 구조는 `TrajectoryEntry`에 이미 계산된 `fused_state`를 저장하고 있습니다. 이는 `detach`를 하지 않는다고 가정했을 때, Critic 학습 시 Actor의 CNN까지 Gradient가 전파되는 문제가 발생합니다(파라미터 공유 금지 위반).
따라서 **올바른 구현을 위해서는 `TrajectoryEntry`에 `fused_state` 결과값이 아니라 `(spike_history, scalars)` 원본 데이터를 저장하고, 학습 단계(`optimize_from_trajectory`)에서 Actor와 Critic이 각각 forward pass를 다시 수행해야 합니다.**

-----

## 4\. 결론

개발자는 위 지시사항, 특히 \*\*CNN 파라미터의 학습 차단 문제(Issue 1)\*\*와 \*\*Actor-Critic 독립성 문제(Issue 2)\*\*를 최우선으로 수정하여 주십시오. 수정 후 `python main.py`를 실행하여 Reward가 변동(학습)되는지, 그리고 Code Level에서 Actor와 Critic의 파라미터가 독립적으로 갱신되는지 확인 후 보고 바랍니다.
