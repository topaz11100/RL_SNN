# AGENTS

## 1. 이 문서의 역할

이 문서는 코딩 LLM(이하 **에이전트**)이 `Theory.md` 를 그대로 구현하기 위한 **작업 명세서**다.

* `Theory.md` 는 이 프로젝트의 **절대 기준**이다.
* `modify.md` 는 기존 구현과 `Theory.md` 사이의 불일치점을 분석하고, **“코드를 Theory 쪽으로 맞춘다”** 라는 방향으로 수정 방침을 정리한 문서다.
* 이 `AGENTS.md` 는 `Theory.md` 와 `modify.md` 에서 최종적으로 결정된 방침을 **코드 레벨 작업 단위로 풀어쓴 실행 계획**이다.
* 개념을 바꾸거나 새 알고리즘을 제안하지 말고, **설계는 `Theory.md` 를 100% 따르되 구현 방법만 결정**한다.
* `modify.md` 에서 논의됐던 여러 선택지 중 **“Theory를 바꾸는 선택지”는 모두 배제되었고**, 항상 **코드를 Theory에 맞추는 방향만** 사용한다.

## 2. 전역 목표 (Goal)

**MNIST 기반 SNN + PPO Actor–Critic 시스템**을 구현한다. 핵심 목표는 다음 네 가지 실험 시나리오를 전부 돌릴 수 있는 하나의 코드베이스를 만드는 것이다.

1. 시나리오 1.1: 완전 비지도, 단일 가중치 정책 (Diehl–Cook 스타일 E/I 네트워크)
2. 시나리오 1.2: 완전 비지도, 두 가중치 정책 (흥분/억제 분리)
3. 시나리오 2: 준지도 단일 정책 분류 (SNN 자체가 분류기)
4. 시나리오 3: 완전지도 gradient mimicry (Teacher gradient 모방)

네 시나리오 모두 공통으로

* `Theory.md` 2장에 정의된 **로컬 상태 벡터, 1D CNN 전단, Gaussian Actor, Critic, PPO 손실**을 그대로 사용한다.
* 강화학습은 항상 **MC 리턴 기반 PPO Actor–Critic (γ = 1)** 하나만 쓴다. 별도의 “plain REINFORCE 모드”는 없다.
* 에피소드는 항상 **이미지 단위**이며, 한 에피소드는 이미지 1장을 SNN 에 T 스텝 동안 흘렸을 때 생성되는 **모든 스파이크·시냅스 이벤트 시퀀스 전체**에 대응한다.

## 3. 전역 제약사항 (Constraints)

에이전트는 작업 시 다음 제약을 반드시 지킨다.

0. **기본 원칙**

   * 리포지토리에 이미 존재하는 코드와 `docs/explain.md` 내용이 `docs/Theory.md` 및 이 `AGENTS.md`와 충돌할 경우, 항상 `docs/Theory.md` 및 `AGENTS.md`를 우선하고, 코드를 그에 맞게 리팩토링해야 한다.
   * `modify.md` 에서 제시된 여러 수정 옵션 중 **“Theory를 변경하는 옵션”은 이미 폐기되었다.**  
     구현 시에는 **반드시 코드를 Theory에 맞추는 방향**으로만 수정한다.

1. **설계 변경 금지**

   * `Theory.md` 의 개념, 손실함수, 보상 정의, PPO 구조 등을 바꾸지 않는다.
   * 구현 편의를 위해 **헬퍼 함수, 클래스, 모듈 분리**는 자유지만, 수식과 의미는 동일해야 한다.

2. **강화학습 알고리즘 고정**

   * 모든 시나리오에서 강화학습은 `Theory.md` 2.9절에 정의된 **PPO Actor 손실 + Critic MSE 손실**만 사용한다.
   * γ = 1, 전역 보상 R 을 에피소드 내 모든 이벤트에 동일하게 할당하는 MC 설정을 그대로 따른다.
   * 별도의 “plain REINFORCE 모드”, “TD 기반 Actor–Critic” 등은 구현하지 않는다.

3. **로그 형식**

   * 실험 로그와 최종 결과는 **텍스트 파일**로만 저장한다.
   * `Theory.md` 에 명시된 대로, 결과 저장용으로 **JSON 파일은 사용하지 않는다.**
   * 필요하면 CSV, TSV, 단순 공백/쉼표 구분 텍스트를 사용한다.

4. **언어 및 프레임워크**

   * 언어: Python 3.x
   * 딥러닝 프레임워크: PyTorch
   * 추가 라이브러리는 최소화하며, SNN 동역학(LIF), PPO, 데이터 로더 등은 직접 구현한다.

5. **재현성**

   * 난수 시드를 하나의 CLI 인수(`--seed`)로 받아, Python `random`, NumPy, PyTorch 에 동일하게 설정한다.
   * CPU/GPU 결과 차이는 허용하지만, **같은 환경 + 같은 seed**에서 같은 결과가 나오도록 한다.

6. **성능·구조 트레이드오프**

   * 벡터화를 우선하되, 코드 가독성과 명확성을 해치지 않는 선에서 구현한다.
   * 너무 복잡한 마이크로 최적화는 하지 않는다. (예: 과도한 커스텀 CUDA 커널 금지)

## 4. 리포지토리 구조 권장안

에이전트는 다음과 같은 디렉터리 구조를 기준으로 코드를 작성한다. 필요시 세부 파일명은 변경해도 되지만, **역할 분리는 유지**한다.

```text
project_root/
  AGENTS.md           # 이 문서

  docs/
    Theory.md         # 설계 기준 문서 (수정 금지)
    explain.md        # 현재 구현/실험 설명용 문서들

  main.py             # 진입점, CLI 파싱, scenario 선택, 학습 루프 실행

  snn/
    __init__.py
    lif.py                    # LIF 셀 구현 (벡터화된 전방 시뮬레이션)
    network_diehl_cook.py     # 시나리오 1.x용 E/I 네트워크
    network_semi_supervised.py# 시나리오 2용 네트워크
    network_grad_mimicry.py   # 시나리오 3용 네트워크
    encoding.py               # Poisson 인코딩 및 스파이크 배열 생성

  rl/
    __init__.py
    policy.py          # 1D CNN + MLP 기반 Gaussian Actor
    value.py           # 1D CNN + MLP 기반 Critic
    buffers.py         # 에피소드/이벤트 버퍼, rollout 저장 구조
    ppo.py             # PPO 손실 계산, Actor–Critic 업데이트 루틴

  scenarios/
    __init__.py
    unsup_single.py    # 시나리오 1.1 학습 루프
    unsup_dual.py      # 시나리오 1.2 학습 루프
    semi_supervised.py # 시나리오 2 학습 루프
    grad_mimicry.py    # 시나리오 3 학습 루프

  data/
    mnist.py           # MNIST 로더, train/val/test 분할

  utils/
    cli.py             # argparse 설정 및 공통 CLI 정의
    logging.py         # 텍스트 로그, 결과 디렉터리 관리
    seeds.py           # 시드 고정 함수
    metrics.py         # 정확도, 마진, 보상 통계 계산

  results/
    ...                # scenario별 run-name별 결과 디렉터리
```

코딩을 시작할 때 **리포지토리 구조를 먼저 생성**하고, 이후 단계별로 파일을 채워 넣는 방식으로 진행한다.

## 5. 공통 구현 규칙 (SNN, 상태, PPO)

### 5.1 LIF 뉴런 및 스파이크 배열

1. `lif.py` 에 벡터화된 LIF 셀을 구현한다.

   * 입력: 현재 막전위 `v`, 입력 전류 `I_syn`, 파라미터(τ, v_th, v_reset 등, dt 포함)
   * 출력: 다음 막전위, 스파이크 마스크 (0/1)
   * 배치 차원, 뉴런 차원, 시간 차원을 모두 고려한 **배치 시뮬레이션 함수**를 제공한다.
   * 시간 간격 `dt` 는 기본값을 두되, CLI 인수 `--dt` 로 덮어쓸 수 있어야 한다.

2. **스파이크 배열 관리**

   * 각 뉴런에 대해 길이 T 의 스파이크 배열을 저장한다: `s_j[t] ∈ {0,1}`.
   * 시냅스 수준의 별도 스파이크 배열은 두지 않고, 필요 시 pre/post 뉴런의 스파이크 배열에서 잘라서 사용한다.

3. **에피소드 경계에서 상태 리셋**

   * 새로운 이미지를 주입하기 전, 모든 LIF 뉴런의 막전위, 누적 전류, refractory 상태를 초기값으로 리셋한다.
   * 이렇게 해서 서로 다른 이미지 에피소드의 상태가 섞이지 않도록 한다.

### 5.2 로컬 스파이크 히스토리와 상태 벡터

1. **히스토리 길이 L (`--spike-array-len`)**

   * 고정 길이 L 의 윈도우를 사용한다.
   * 조건: 항상 `L ≤ T` 가 되도록 사용자가 하이퍼파라미터를 선택한다고 가정한다.

2. **로컬 2채널 히스토리 X_i(t)**

   * 시냅스 i 가 뉴런 j → k 를 연결한다고 할 때, 시각 t 에서

     * pre 스파이크 히스토리: `x_i^pre[τ] = s_j[t-τ]`, τ = 0..L-1
     * post 스파이크 히스토리: `x_i^post[τ] = s_k[t-τ]`, τ = 0..L-1
   * 이를 `X_i(t) ∈ {0,1}^{2 × L}` 로 묶어 CNN 입력으로 쓴다.

3. **추가 feature**

   * 항상 포함: 현재 시냅스 가중치 `w_i(t)`
   * 이벤트 타입 원핫 `e_type(e) ∈ {0,1}^2` (pre/post 구분)
   * 완전지도 시나리오(gradient mimicry)에서만 레이어 인덱스 정규화 스칼라 `l_norm,i` 추가
     (`l_norm` 의 스케일링은 CLI 인수 `--layer-index-scale` 로 제어한다.)

4. **최종 상태 벡터 z_i(t)**

   * CNN 전단 출력 `h_i(t) ∈ R^{16}` 와 위 feature 들을 concat 해서 만든다.
   * 비지도·준지도: 차원 19
   * 완전지도: 차원 20
   * 정확한 수식 형태는 `Theory.md` 를 따른다.

### 5.3 Actor (Gaussian 정책) 및 Critic

1. **공통 CNN 전단 구조** (정확한 구조는 `Theory.md` 2.4절)

   * Conv1d(2 → 16, kernel=5, padding=2, stride=1) + ReLU
   * Conv1d(16 → 16, kernel=5, padding=2, stride=1) + ReLU
   * 시간 축 global average pooling → 길이 16 벡터

2. **Actor MLP head**

   * 입력: 상태 벡터 z (차원 d)
   * 구조: d → 32 (ReLU) → 32 (ReLU) → 1 (Tanh)
   * 출력 Tanh 를 평균 m ∈ [-1,1] 로 사용하고, 시나리오/정책별 `σ_policy` 를 곱해 Gaussian 정책을 만든다.
   * 액션 샘플: `Δd_i(t) ~ N(m_i(t), σ_policy^2)`
   * **가중치 업데이트 및 클리핑:**

     * 액션 출력 `Δd_i(t)` 에 대해 시나리오별 내부 스케일 `s_scen` 과 로컬 학습률 `η_w` 를 곱해
       `Δw_i(t) = η_w * s_scen * Δd_i(t)` 를 만든다.
     * 여기서 `η_w` 는 CLI 인수 `--local-lr` 에서 읽는다.
     * **어떠한 시나리오에서도 `0.01 * action` 과 같은 하드코딩된 상수 스케일을 사용하지 않는다.**
       항상 `args.local_lr * s_scen * action` 형태로 구현한다.
     * 그런 다음 시냅스 종류(흥분/억제)에 따라 **사전에 정의된 범위로 in-place 클리핑** 한다.

       * 흥분성 시냅스: `w_exc ∈ [exc_clip_min, exc_clip_max]`
       * 억제성 시냅스: `w_inh ∈ [inh_clip_min, inh_clip_max]`
     * `exc_clip_min`, `exc_clip_max`, `inh_clip_min`, `inh_clip_max` 는 CLI 인수로 전달되는 하이퍼파라미터이며,
       실제 구현에서는 `torch.clamp_` 와 같은 in-place 연산을 사용해 **가중치 업데이트 직후 매 스텝마다** 적용해야 한다.

3. **Critic MLP head**

   * 입력: z
   * 구조: d → 32 (ReLU) → 32 (ReLU) → 1 (선형)
   * 출력: `V_φ(z_i(t))` (스칼라)

4. **파라미터 공유 규칙**

   * Actor 와 Critic 은 CNN 전단까지 포함해 **파라미터를 공유하지 않는다.**
   * 시나리오별, 정책별로 Actor·Critic 을 별도 인스턴스로 둔다.
   * optimizer 도 Actor, Critic 각각 따로 둔다.

### 5.4 이벤트 버퍼와 에피소드 구조

1. **이벤트 정의**

   * 하나의 이벤트 e 는 `(i, t, type)` 으로 표현한다.
   * 여기서 i 는 시냅스 인덱스, t 는 타임스텝, type 은 pre/post 구분이다.

2. **버퍼에 저장할 내용**
   각 이벤트에 대해 최소한 다음 정보를 버퍼에 저장한다.

   * 상태 벡터: `s_e = z_i(t)`
   * 액션: `a_e = Δd_e` (샘플 값)
   * log 확률: `log π_θ(a_e | s_e)` (PPO ratio 계산용)
   * Critic 출력: `V_e = V_φ(s_e)`
   * (옵션) 시냅스/에피소드 인덱스: 나중에 통계 계산·분석용

3. **에피소드·배치 구조**

   * 한 이미지 = 한 에피소드.
   * 구현에서는 여러 이미지를 묶어 **이미지 단위 미니배치**를 구성하고, 모든 이벤트를 concat 해서 하나의 대형 배치로 만든다.
   * 각 이벤트의 보상·리턴은 **이미지 단위 전역 보상 R 에 의해 이미 결정**되어 있다고 가정한다.

### 5.5 PPO 손실 구현

1. **리턴 및 Advantage**

   * 모든 시나리오에서 γ = 1, 에피소드 내 모든 이벤트에 대해 보상 `r_e = R` (전역 보상).
   * 따라서 `G_e = R`, `A_e = R - V_e`.

2. **PPO Actor 손실**

   * rollout 당시 파라미터를 θ_old 라 하고, ratio 를
     `r_e(θ) = π_θ(a_e|s_e) / π_θ_old(a_e|s_e)` 로 정의.
   * 클리핑 범위 ε 는 CLI 인수 `--ppo-eps` 로 받는다.
   * per-event 손실
     `L_e^CLIP(θ) = -min( r_e(θ) A_e, clip(r_e(θ), 1-ε, 1+ε) A_e )` 를 구현하고, 배치 평균을 Actor 손실로 사용한다.

3. **Critic 손실**

   * `L_critic(φ) = mean( (G_e - V_e)^2 ) = mean( (R - V_e)^2 )` 를 구현한다.

4. **전체 PPO 손실**

   * `L_PPO(θ, φ) = E_e[ L_e^CLIP(θ) + c_v (G_e - V_e)^2 ]` 형태로 하나의 스칼라를 계산하되,

     * Actor optimizer 에는 `∂/∂θ` 에 해당하는 부분만 사용하고
     * Critic optimizer 에는 `∂/∂φ` 에 해당하는 부분만 사용한다.

5. **이미지 단위 미니배치 업데이트**

   * N장의 이미지를 모아 한 번에 rollout 한 뒤, 모든 이벤트를 flatten 해서 하나의 배치로 만든다.
   * 이 배치 전체로부터 `L_PPO` 를 계산하고, CLI 인수 `--ppo-epochs` 및 `--ppo-batch-size` 를 사용해
     내부 미니배치를 나눠가며 반복 업데이트를 수행한다.
   * 시나리오 러너는 `train_loader`에서 가져온 **하나의 이미지 단위 미니배치**에 대해, 각 이미지를 에피소드로 시뮬레이션하면서 이벤트 버퍼만 채우고, 배치 내부 루프가 끝난 뒤에야 이 미니배치 전체를 합친 버퍼로 `ppo_update(...)`를 **한 번만** 호출한다.
   * 에피소드가 끝날 때마다 `ppo_update(...)`를 호출하는 방식은 사용하지 않는다.

## 6. 시나리오별 추가 요구사항

아래 내용은 `Theory.md` 3~6장 내용을 구현 관점에서 요약한 것이다. **정확한 수식·상세 정의는 항상 `Theory.md` 를 우선**한다.

### 6.1 시나리오 1.1 (완전 비지도, 단일 정책)

1. **네트워크 구조**

   * Diehl–Cook 스타일 E/I 네트워크를 `network_diehl_cook.py` 에 구현한다.
   * Input(784) → Excitatory(E, `--N-E`) → Inhibitory(I, `--N-E`) 구조.
   * E→I 는 **학습되지 않는 1:1 고정 회로**이며, I→E 는 전결합 억제 시냅스이다.
   * 이때 E→I 연결에는 학습 가능한 가중치 행렬(`nn.Parameter`)을 두지 않고,
     j번째 흥분 뉴런의 스파이크가 항상 j번째 억제 뉴런으로 동일한 강도의 고정 입력을 전달하는 회로로 구현해야 한다.

2. **가중치 정책 배치**

   * 모든 학습 시냅스(Input→E, I→E)에 대해 **단일 Actor–Critic** (`π_single`) 을 사용한다.
   * pre/post 구분은 상태 벡터의 이벤트 타입 원핫으로만 표현한다.
   * Input→E 와 I→E 중 어느 것도 “생성만 하고 업데이트하지 않는” 시냅스가 남지 않도록 한다.
     두 종류 시냅스 모두 동일 정책으로 학습되어야 한다.

3. **전역 보상 R**

   * 목표 발화율 보상 `R_sparse`:

     * 에피소드 동안의 평균 발화율을 `\bar r` 라 하고, 목표 발화율을 `ρ_target` 이라 할 때
       `R_sparse = -(\bar r - ρ_target)^2` 로 정의한다.
       즉, **절댓값이 아닌 제곱 거리**를 사용하며, `\bar r` 가 `ρ_target` 에 가까울수록 보상이 높다.
   * 다양성 보상 `R_div`:

     * 에피소드 누적 winner 히스토그램 `H^{(K)}` (K 번째 에피소드까지의 winner 카운트) 를 두고,
       정규화된 분포 `p_j^{(K)} = H_j^{(K)} / ∑_ℓ H_ℓ^{(K)}` 를 사용한다.
     * 균등 분포 `u_j = 1/N_E` 와의 제곱 거리
       `R_div^{(K)} = -∑_j (p_j^{(K)} - u_j)^2` 를 현재 에피소드의 다양성 보상으로 사용한다.
   * 안정성 보상 `R_stab`:

     * 동일 이미지를 여러 번 보여줄 때 winner 가 바뀌지 않을수록 +1, 바뀌면 -1.
     * MNIST 데이터셋에서 **샘플 인덱스(index)** 를 함께 받아와 `prev_winner[index]` 테이블을 유지한다.
     * 에피소드 종료 시 winner 를 계산하고, 이전 winner 와 비교해

       * 처음 방문인 경우: `R_stab = 0`
       * 이전과 동일: `R_stab = +1`
       * 이전과 다름: `R_stab = -1`
   * 최종 보상: `R = α_sparse R_sparse + α_div R_div + α_stab R_stab`.
   * α 계수와 `ρ_target` 은 모두 CLI 인수로 받는다.

4. **평가 및 로그**

   * Diehl–Cook 뉴런 라벨링 기반 분류 정확도를 구현한다. (학습 후 label assignment, 그 후 평가)
   * 최소한 다음을 텍스트 파일/그래프로 저장한다.

     * `R_sparse`, `R_div`, `R_stab` 변화
     * train/val/test accuracy 변화
     * Δt–Δd 산점도 (pre–post 시간차 vs 액션)
     * 가중치 분포 히스토그램

### 6.2 시나리오 1.2 (완전 비지도, 두 정책)

1. **네트워크 구조**

   * 구조 자체는 시나리오 1.1과 동일하다.

2. **정책 분리**

   * Input→E 시냅스에 Actor `π_exc`
   * I→E 시냅스에 Actor `π_inh`
   * 두 정책의 구조는 동일하지만 파라미터는 독립.

3. **보상 및 학습 절차**

   * 전역 보상 R 정의 및 PPO 업데이트 절차, 로그/산출물 요구사항은 **시나리오 1.1과 동일**하다.
   * Δt–Δd 산점도와 가중치 분포는 흥분/억제 정책별로 **분리해서** 저장한다.

4. **추가 로그**

   * 정책별 Δt–Δd 산점도 (흥분/억제 별로 분리)
   * 정책별 가중치 분포 히스토그램 (학습 전/후 비교)

### 6.3 시나리오 2 (준지도 단일 정책 분류)

1. **네트워크 구조**

   * Input(784) → Hidden LIF 층(`--N-hidden`) → Output(10) LIF 층.
   * 출력 뉴런 인덱스와 숫자 라벨 0~9 를 1:1로 대응시킨다.

2. **정책 배치**

   * Input→Hidden, Hidden→Output 모든 학습 시냅스에 대해 단일 정책 `π_semi` 를 사용한다.
   * 두 종류의 시냅스를 모두 이벤트로 수집하여 하나의 Actor–Critic 에 넣고,
     액션 벡터를 적절히 분할하여 각 레이어 가중치에 적용한다.

3. **전역 보상 R**

   * 기본 분류 보상 `R_cls`:

     * `ŷ = y` (정답) → +1
     * `ŷ ≠ y` (오답) → -1
   * 마진 보상 `R_margin`:

     * 정답 뉴런 발화율 `r_y`, 오답 중 최대 발화율 `r_max,wrong`.
     * 마진 `M = r_y - r_max,wrong`.
     * `R_margin = β * M` (β 는 CLI 인수 `--beta-margin`).
   * 최종 보상: `R = R_cls + R_margin`.

4. **평가 및 산출물**

   * train/val/test accuracy 곡선을 저장.
   * 마진 M 의 분포를 히스토그램으로 저장 (학습 전/후 비교).
   * Δt–Δd 산점도 및 가중치 분포 저장.
   * 모든 업데이트는 `Δw = args.local_lr * s_scen * action` 을 사용해야 하며,
     하드코딩 상수 스케일을 사용하지 않는다.

### 6.4 시나리오 3 (완전지도 gradient mimicry)

1. **네트워크 구조**

   * 입력 784 → 여러 은닉층(예: 256, 128, 64, 32) → 출력 10 LIF.
   * 각 시냅스에 정규화된 레이어 인덱스 `l_norm,i` 를 부여한다.
   * 입력층을 제외한 모든 층 사이의 시냅스가 학습 대상이다.
   * 실제 은닉층 크기는 `Theory.md` 의 설정(예: 256, 128, 64, 32) 을 따르되,
     코드에서는 `network_grad_mimicry.py` 에 고정된 구조로 구현한다.

2. **정책 배치 및 상태**

   * 모든 학습 시냅스에 단일 정책 `π_grad` 사용.
   * 상태 벡터에 `l_norm,i` 를 포함한다 (`--layer-index-scale` 로 크기 조절).

3. **Teacher gradient 추출**

   * 동일 구조의 SNN 을 surrogate gradient + BPTT 로 역전파 가능하게 구현한다.
   * supervised loss `L_sup` (예: softmax(α r_k) + cross-entropy) 를 정의하고,
     각 가중치 w_i 에 대해 `g_i = ∂L_sup/∂w_i` 를 계산한다.
   * Teacher 네트워크는 optimizer 업데이트를 하지 않고, gradient 만 제공한다.

4. **보상 정의**

   * Teacher 기준 업데이트: `Δw_i^teacher = -η_align * g_i` (η_align 은 `--alpha-align`).
   * 에이전트가 실제로 만든 업데이트: `Δw_i^agent` (에피소드 동안 해당 시냅스에서 누적된 업데이트).
   * 시냅스별 보상: `R_i = -(Δw_i^agent - Δw_i^teacher)^2`.
   * 에피소드 전역 보상: `R = mean_{i ∈ S} R_i`
     여기서 S 는 **해당 에피소드 동안 실제 업데이트가 한 번이라도 일어난 시냅스 집합**이다.
     (업데이트가 없는 시냅스는 평균에서 제외한다.)

5. **학습 파이프라인 순서**

   * 한 에피소드(이미지)에 대해 다음 순서를 따른다.

     1. SNN 시뮬레이션 및 이벤트 수집 (로컬 정책 적용, 아직 실제 가중치는 업데이트하지 않거나 “임시”로만 관리)
     2. BPTT 로 Teacher gradient `g_i` 계산
     3. 이벤트 히스토리로부터 `Δw_i^{agent}` 재구성 (에피소드 동안의 누적 업데이트량)
     4. 위의 보상 정의로 전역 보상 R 계산
     5. `EpisodeBuffer` 를 사용해 PPO Actor–Critic 업데이트 수행
     6. 그 이후에야 최종적으로 `Δw_i^{agent}` 를 실제 네트워크 가중치에 적용한다.

   * 즉, “이벤트마다 곧바로 가중치를 바꾸는 구조”가 아니라
     **“에피소드 끝에서 Teacher gradient 와 비교한 뒤, PPO 업데이트를 거쳐 최종 업데이트”** 하는 구조로 구현한다.

6. **산출물**

   * gradient 정렬 지표 곡선:

     * 예: `align_loss = mean_{i ∈ S} (Δw_i^agent - Δw_i^teacher)^2` 를 epoch별로 기록
   * Teacher 직접 supervised 학습 vs RL 정책 학습의 train/val/test accuracy 비교 곡선.
   * Δt–Δd 산점도.
   * `--log-gradient-stats` 가 켜진 경우에만 대형 통계(gradient–Δw 상관, 분포 등)를 추가로 파일로 저장한다.

## 7. CLI 및 실험 실행 규칙

### 7.1 공통 CLI 인수들

1. 시나리오 및 시간 관련

   * `--scenario`: `{unsup1, unsup2, semi, grad}` 중 하나.
   * `--T-unsup1`, `--T-unsup2`, `--T-semi`, `--T-sup`: 시나리오별 타임스텝.
   * `--dt`: LIF 시뮬레이션 시간 간격. `lif.py` 의 파라미터를 이 값으로 덮어쓴다.
   * `--spike-array-len`: 로컬 스파이크 히스토리 길이 L.
   * `--batch-size-images`: 이미지 단위 미니배치 크기.
   * `--num-epochs` 또는 `--num-episodes`: 전체 학습 반복 수.

2. 네트워크 크기 관련

   * `--N-E`: Diehl–Cook 흥분 뉴런 수.
   * `--N-hidden`: 준지도 네트워크의 hidden 뉴런 수.
   * gradient mimicry 네트워크의 은닉층 크기는 코드에 고정하되, 필요하면 별도 CLI 인수로 확장 가능하다.

3. 실행 및 로깅 관련

   * `--seed`: 난수 시드.
   * `--run-name`: 결과 디렉터리 이름에 사용될 문자열.
   * `--log-interval`: 학습 중 로그를 찍는 에포크/에피소드 간격.

### 7.2 정책/최적화 관련 CLI

1. 정책 관련

   * `--sigma-unsup1`, `--sigma-unsup2`, `--sigma-semi`, `--sigma-sup`: 정책별 Gaussian 표준편차.
   * `--layer-index-scale`: gradient mimicry 시나리오에서 `l_norm` 의 스케일을 조절하는 스칼라.

2. 최적화 및 PPO 관련

   * `--lr-actor`, `--lr-critic`: Actor, Critic 학습률.
   * `--ppo-eps`: PPO 클리핑 범위 ε.
   * `--ppo-epochs`: 한 미니배치에 대해 PPO 업데이트 반복 횟수.
   * `--ppo-batch-size`: PPO 내부에서 이벤트를 나눌 때 사용할 미니배치 크기.

### 7.3 보상 관련 CLI

* `--rho-target`, `--alpha-sparse`, `--alpha-div`, `--alpha-stab`
* `--beta-margin`
* `--alpha-align`

각 계수는 `Theory.md` 에 정의된 보상 항들을 그대로 구현하기 위한 스칼라이며,
실험 시에는 이 값들만 조정해서 다양한 설정을 테스트한다.

### 7.4 로그 및 결과 디렉터리 규칙

1. 결과 디렉터리 구조

   * `results/{scenario}/{run-name}/` 구조를 기본으로 한다.
   * `run-name` 은 CLI 인수 `--run-name` 에서 읽는다.
     예: `unsup1_default`, `semi_margin0.5` 등.

2. 로그 파일

   * 각 run 디렉터리 내에는 최소한 다음을 포함한다.

     * `log.txt`: 하이퍼파라미터와 주요 지표를 사람이 읽을 수 있는 텍스트로 기록.
     * `metrics_train.txt`, `metrics_val.txt`, `metrics_test.txt`: 에포크/에피소드별 정확도·보상·마진 등의 숫자 로그.
     * 필요한 경우, Δt–Δd 산점도, 히스토그램 등을 이미지 파일로 저장.

   * `--log-interval` 은 `log.txt` 및 metrics 파일에 기록하는 빈도를 제어하는 데 사용한다.

   * gradient 관련 대형 통계는 `--log-gradient-stats` 가 `True` 인 경우에만 기록한다.

### 7.5 가중치 클리핑 및 로컬 학습률 관련 CLI

1. 클리핑 관련

   * `--exc-clip-min`, `--exc-clip-max`
     흥분성 시냅스 가중치의 최소/최대값. Actor 업데이트 이후 `w_exc` 에 대해
     `torch.clamp_(w_exc, exc_clip_min, exc_clip_max)` 와 같이 in-place 로 항상 적용한다.
   * `--inh-clip-min`, `--inh-clip-max`
     억제성 시냅스 가중치의 최소/최대값. 마찬가지로 업데이트 이후 `w_inh` 에 대해 in-place 클리핑을 수행한다.

2. 로컬 학습률 관련

   * `--local-lr` (또는 시나리오별로 `--local-lr-unsup1`, `--local-lr-unsup2` 등)
     로컬 학습률 `η_w` 를 제어하는 인수. 모든 시나리오에서
     `Δw_i(t) = η_w * s_scen * Δd_i(t)` 의 형태가 되도록, 하드코딩된 상수 대신 CLI 값에서 읽어 사용한다.
   * 구현에서는 **모든 시나리오에서 동일한 규칙**으로 위 인수들을 사용해야 하며,
     weight update 직후 흥분/억제 시냅스를 각각 지정된 범위로 클리핑하는 것을 빠뜨리면 안 된다.
   * `0.01 * action` 과 같은 상수 스케일이 코드에 남아 있다면, 모두 `args.local_lr * action` 구조로 리팩토링한다.

## 8. 에이전트 응답 스타일 (Codex용)

에이전트가 사용자 요청에 응답할 때 지켜야 할 규칙은 다음과 같다.

1. **맥락 파악 우선**

   * 항상 `docs/Theory.md` 와 `AGENTS.md` 를 전제로 삼고, 필요하면 해당 파일 이름과 섹션을 명시적으로 언급한다.
   * 설계와 다르게 보이는 코드를 작성해야 한다면,
     “설계를 바꾸는 방향”이 아니라 **항상 `Theory.md` 를 그대로 둔 채 코드를 수정하는 방향**만 제안한다.
     `modify.md` 단계에서 이미 설계 변경 옵션은 배제되었다고 가정한다.

2. **출력 형태**

   * 가능하면 **완성된 코드 파일 전체** 또는 **적용 가능한 patch** 형식으로 답한다.
   * 어떤 파일의 어떤 부분을 수정하는지, 줄 단위 설명을 덧붙인다.

3. **설명 vs 코드**

   * 사용자가 "코드만" 을 요구하면 최소한의 주석만 포함한 코드 블록만 출력한다.
   * 그 외에는 변경 의도와 구현 선택 이유를 간단히 한국어로 설명한 뒤 코드 블록을 제시한다.

4. **테스트 및 검증**

   * 새 기능을 추가하면, 간단한 **self-check 함수** 또는 **짧은 테스트 스크립트 예시**도 함께 제공해 사용자가 바로 실행해 볼 수 있게 한다.

## 9. Definition of Done (최소 완료 기준)

이 프로젝트에서 “기본 구현이 끝났다”고 간주하기 위한 최소 조건은 다음과 같다.

1. `main.py` 에서 `--scenario` 를 바꾸어 네 시나리오를 모두 실행할 수 있다. (코드 레벨에서 오류 없이 실행)

2. 각 시나리오에 대해

   * MNIST 를 로드하고 에피소드 단위로 SNN 시뮬레이션을 수행한다.
   * PPO Actor–Critic 업데이트가 정상적으로 돌아가며 손실이 finite 값을 유지한다.

3. `results/{scenario}/{run-name}/` 에 텍스트 로그와 기본 지표 파일이 생성된다.

4. 적어도 소규모 학습(예: 몇 에포크)에서

   * 비지도 실험: 발화율이 목표 발화율 근처로 수렴하려는 경향과 winner 다양성/안정성에 의미 있는 변화가 관찰된다.
   * 준지도/완전지도 실험: train accuracy 및 마진/gradient 정렬 지표가 초기 값보다 개선되는 경향이 나타난다.

위 조건을 만족하면, 이후에는 하이퍼파라미터 조정, 추가 분석 도구, 코드 리팩토링 등 **고도화 작업**을 별도의 요청으로 진행한다.
