### 1. 프로젝트 개요 및 개발 방향성

본 문서는 **로컬 스파이크 히스토리 기반의 RL 에이전트(Actor-Critic)를 이용한 SNN 시냅스 가중치 학습** 프로젝트의 구현 명세서이다. 개발자는 제공된 `Theory.md`를 절대적 기준으로 삼아 구현해야 하며, 임의의 해석을 배제하고 아래 명세에 따라 모듈을 구성해야 한다.

가장 중요한 목표는 **기존의 하드코딩된 STDP 규칙을 신경망(CNN+MLP) 기반의 학습 가능한 정책으로 대체**하는 것이다. 이를 위해 각 시냅스가 마치 하나의 작은 생명체처럼 자신의 상황(로컬 상태)을 보고 행동(가중치 조절)을 결정하는 구조를 만들어야 한다.

### 2. 기술 스택 및 환경 설정

**언어 및 프레임워크**
* **Python 3.8+**
* **PyTorch 1.12+** (GPU 가속 필수, Tensor 연산 최적화)
* **Torchvision**: MNIST 데이터셋 로드용
* **Numpy, Matplotlib**: 데이터 처리 및 시각화(산점도, 곡선 등)

**디렉토리 구조 제안**
* `src/`: 소스 코드
    * `models/`: SNN 뉴런, 시냅스 에이전트, CNN 전단 등 핵심 모델
    * `scenarios/`: 4가지 실험 시나리오별 클래스 (Experiment 1~4)
    * `utils/`: 데이터 로더, 로깅, 보상 계산 함수, 시각화
* `main.py`: CLI 진입점

### 3. 핵심 모듈 상세 명세

#### 3.1. 로컬 상태 처리 및 CNN 전단 (Feature Extractor)

모든 정책(Actor)과 가치함수(Critic)가 공유하는 구조이지만, 파라미터는 공유하지 않는 **1D CNN 모듈**이다.

* **입력 데이터**: 시냅스 $i$의 pre/post 스파이크 히스토리
    * 형태: $2 \times L$ (채널 2: pre, post / 길이 $L$: 타임스텝)
    * 구현 시 `Queue`나 `Rolling Buffer`를 사용하여 최근 $L$개의 $0, 1$ 스파이크를 유지해야 한다.
* **CNN 아키텍처** (Theory 2.4 준수)
    1.  **Conv1d**: 입력 채널 2, 출력 채널 16, 커널 5, 패딩 2, 스트라이드 1, **ReLU**
    2.  **Conv1d**: 입력 채널 16, 출력 채널 16, 커널 5, 패딩 2, 스트라이드 1, **ReLU**
    3.  **Global Average Pooling**: 시간 축(L)에 대해 평균을 취함
    4.  **출력**: $h_{i}(t) \in \mathbb{R}^{16}$

#### 3.2. Actor-Critic 네트워크 (Policy & Value Head)

CNN 특징 벡터 $h_{i}(t)$와 추가 정보를 결합하여 최종 출력을 내는 MLP이다.

* **입력 벡터 $z_{i}(t)$ 구성** (Late Fusion)
    * 기본: $[h_{i}(t) ; w_{i}(t) ; \mathbf{e}_{type}(e)]$ (차원: $16 + 1 + 2 = 19$)
    * **주의**: 실험 4(Scenario 3)의 경우 레이어 인덱스 $l_{norm,i}$가 추가되어 차원이 $20$이 됨.
    * $\mathbf{e}_{type}(e)$: pre 이벤트면 $(1, 0)$, post 이벤트면 $(0, 1)$인 원핫 벡터.
* **MLP 공통 구조**
    1.  Linear ($d \to 32$) + ReLU
    2.  Linear ($32 \to 32$) + ReLU
* **Head 분기**
    * **Actor Head**: Linear ($32 \to 1$) $\to$ **Tanh** $\to$ 평균 $m_{i}(t)$ 출력
    * **Critic Head**: Linear ($32 \to 1$) $\to$ $V_{\phi}(z_{i}(t))$ 출력 (Activation 없음)
* **Action 샘플링 (Actor)**
    * $m_{i}(t)$를 평균으로 하고, CLI로 입력받은 $\sigma_{policy}$를 표준편차로 하는 정규분포에서 $\Delta d_{i}(t)$ 샘플링.
    * 수식: $\Delta d_{i}(t) \sim \mathcal{N}(m_{i}(t), \sigma_{policy}^{2})$

#### 3.3. LIF 뉴런 모델

표준 Leaky Integrate-and-Fire 모델을 구현한다.

* **상태 변수**: 막전위 $v(t)$, 스파이크 유무 $s(t)$
* **미분 방정식**: $\tau_{m} \frac{dv(t)}{dt} = -v(t) + R I_{syn}(t)$
* **동작 로직**:
    1.  입력 전류 $I_{syn}$ 적분하여 $v(t)$ 갱신.
    2.  $v(t) \ge v_{\theta}$ 인지 확인.
    3.  참이면 $s(t)=1$, $v(t)$ 리셋 (Soft/Hard 리셋은 하이퍼파라미터). 아니면 $s(t)=0$.

### 4. 실험 시나리오별 구현 가이드

각 시나리오는 별도의 클래스로 관리하며, 공통적으로 **에피소드 단위 업데이트**를 수행한다.

#### 4.1. 실험 1 (시나리오 1.1): 완전 비지도 단일 정책

* **네트워크 구조**: Diehl & Cook (Input $\to$ Exc $\to$ Inh $\to$ Exc)
    * Input $\to$ Exc: 학습 대상 (흥분성)
    * Exc $\leftrightarrow$ Inh: 1:1 고정 연결 (E가 쏘면 I도 쏨)
    * Inh $\to$ Exc: 학습 대상 (억제성)
* **정책**: **단일 Actor-Critic** 사용. 모든 시냅스(흥분/억제 불문)가 동일한 네트워크 공유.
* **보상 ($R$) 계산**:
    * 에피소드 종료 후 계산.
    * **희소성($R_{sparse}$)**: 평균 발화율이 $\rho_{target}$에서 멀어지면 페널티. $-(\bar{r} - \rho_{target})^{2}$
    * **다양성($R_{div}$)**: Winner 뉴런 히스토그램이 균등 분포에서 멀어지면 페널티. $-\sum (p_{j} - 1/N_{E})^{2}$
    * **안정성($R_{stab}$)**: 동일 이미지 재입력 시 Winner가 바뀌면 $-1$, 유지되면 $+1$.
    * 최종 $R$은 위 3가지의 가중 합.

#### 4.2. 실험 2 (시나리오 1.2): 완전 비지도 두 정책

* **구조**: 실험 1과 동일.
* **정책**: **두 개의 Actor-Critic** ($\pi_{exc}, \pi_{inh}$) 분리.
    * Input $\to$ Exc 시냅스는 $\pi_{exc}$ 사용.
    * Inh $\to$ Exc 시냅스는 $\pi_{inh}$ 사용.
* **보상**: 실험 1과 완전히 동일한 $R$ 사용.

#### 4.3. 실험 3 (시나리오 2): 준지도 단일 정책

* **구조**: Feedforward (Input $\to$ Hidden $\to$ Output).
* **출력**: Output 뉴런 10개 (Digit 0~9). 가장 많이 발화한 뉴런이 예측값 $\hat{y}$.
* **정책**: 단일 $\pi_{semi}$ 사용.
* **보상 ($R$) 계산**:
    * **분류($R_{cls}$)**: 맞으면 $+1$, 틀리면 $-1$.
    * **마진($R_{margin}$)**: $\beta \ast (r_{y} - r_{max,wrong})$. 정답 뉴런과 오답 1등 뉴런의 발화율 차이.

#### 4.4. 실험 4 (시나리오 3): 완전지도 Gradient Mimicry

* **핵심**: Supervised Learning의 Gradient를 흉내 내는 것이 목표.
* **Teacher Gradient**:
    * 매 에피소드마다 BPTT(Backpropagation Through Time)를 수행하여 $\frac{\partial \mathcal{L}_{sup}}{\partial w_{i}}$ 계산 (Surrogate Gradient 사용).
    * Teacher 업데이트량: $\Delta w_{i}^{teacher} = -\eta_{align} \ast g_{i}$
* **보상 ($R$)**:
    * 에이전트가 수행한 실제 $\Delta w_{i}^{agent}$와 Teacher의 제안값 간의 차이(MSE)의 음수값.
    * $R = -\frac{1}{|\mathcal{S}|} \sum (\Delta w_{i}^{agent} - \Delta w_{i}^{teacher})^{2}$
* **입력 특징**: $z_{i}^{sup}(t)$에 레이어 위치 정보 $l_{norm,i}$ 추가.

### 5. 학습 및 업데이트 로직 (핵심)

이 프로젝트는 **Online On-policy Monte Carlo** 방식을 따른다. 배치 학습이 아니다.

**에피소드 루프 (이미지 1장)**
1.  **초기화**: 뉴런 상태, 스파이크 히스토리 큐 초기화.
2.  **타임스텝 진행 ($t = 1 \dots T$)**:
    * 입력 이미지를 포아송 스파이크로 변환하여 주입.
    * 스파이크 발생 시 해당 뉴런에 연결된 시냅스들 **이벤트 생성**.
    * **Action 수행**:
        * 이벤트 $e$ 발생 $\to$ $z_{i}(t)$ 구성 $\to$ Actor가 $\Delta d_{e}$ 출력.
        * 가중치 즉시 갱신: $w_{i} \leftarrow w_{i} + \text{scale} \ast \Delta d_{e}$. (클리핑 적용)
    * **데이터 저장**: 버퍼에 $(s_{e}, a_{e}, V_{e})$ 저장. $r_{e}$는 아직 모름.
3.  **에피소드 종료 후**:
    * 전역 보상 $R$ 계산 (시나리오별 로직).
    * 버퍼의 모든 데이터에 대해 $r_{e} = R$, $G_{e} = R$ (할인율 $\gamma=1$ 이므로).
    * **Advantage**: $A_{e} = R - V_{e}$.
4.  **파라미터 업데이트**:
    * Actor Loss: $L_{actor} = -\frac{1}{E} \sum A_{e} \log \pi(a_{e}|s_{e})$
    * Critic Loss: $L_{critic} = \frac{1}{E} \sum (R - V_{e})^{2}$
    * Optimizer(Adam 등)로 $\theta, \phi$ 갱신.
    * 버퍼 비움 (On-policy).

### 6. 로깅 및 평가 명세

**파일 포맷 규칙**
* **JSON 사용 금지**. 모든 로그는 텍스트 파일(.txt) 또는 CSV, 시각화 이미지(.png)로 저장한다.

**필수 산출물**
1.  **$\Delta t$ - $\Delta d$ 산점도**:
    * X축: pre-post 스파이크 시간차 ($\Delta t$).
    * Y축: 에이전트가 출력한 가중치 변화량 ($\Delta d$).
    * 의미: 에이전트가 STDP와 유사한 곡선을 학습했는지, 혹은 전혀 다른 방식을 쓰는지 확인용.
2.  **학습 곡선**:
    * X축: 에피소드 수.
    * Y축: 보상 구성요소($R_{sparse}$ 등), Train/Test 정확도.
3.  **정확도 평가 (Diehl-Cook 라벨링)**:
    * 비지도 학습(실험 1, 2)의 경우, 학습이 끝난 후 Train 셋 전체를 돌려 각 뉴런이 어떤 숫자에 반응하는지 라벨링(Labeling)하는 과정을 반드시 거쳐야 함.

### 7. CLI 파라미터 요구사항

`argparse`를 사용하여 `Theory.md` 8장에 명시된 모든 하이퍼파라미터를 제어할 수 있어야 한다.

**주요 인자 예시**
* `--scenario`: 1.1, 1.2, 2, 3 중 선택
* `--T-unsup1`, `--T-sup`: 시나리오별 타임스텝
* `--sigma-policy`: 탐험 노이즈 표준편차
* `--rho-target`, `--alpha-sparse`: 보상 관련 계수
* `--run-name`: 결과 저장 폴더명

---
**개발자 참고사항**: 이 프로젝트는 단순한 분류기 제작이 아니라, **"시냅스가 스스로 학습하는 규칙을 학습하는 것"**이 핵심이다. Actor 네트워크가 내놓는 $\Delta d$값이 시간이 지남에 따라 어떤 분포를 보이는지 관찰하는 것이 연구의 본질이므로, 로깅과 시각화 모듈 구현에 각별히 신경 써주길 바란다.