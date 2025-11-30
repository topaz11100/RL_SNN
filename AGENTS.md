### AGENTS: RL-SNN 코드베이스와 Theory.md 정렬 작업

`Theory.md`는 프로젝트의 **단일 진실 소스(single source of truth)** 이다. 이 문서의 작업 항목은 **코드가 Theory.md와 1:1로 대응하도록 정렬시키는 것**을 목표로 한다.

### 0. 공통 아키텍처 및 인프라

**G0.1 CLI 하이퍼파라미터 정비**
* [ ] `main.py`의 `parse_args()`를 Theory.md 8장의 항목과 일치하도록 확장한다.
    * 시뮬레이션: `--dt`
    * SNN 구조: `--N-E`, `--N-hidden`, LIF 파라미터(`--lif-tau-m`, `--lif-v-threshold`, `--lif-v-reset`)
    * 학습: `--lr-actor`, `--lr-critic`
    * Gradient Mimicry: `--alpha-align`, `--log-gradient-stats`
    * 관리: `--log-interval` (기존 유지), `--run-name`
* [ ] 추가된 인자들을 각 시나리오 클래스(`__init__`)로 올바르게 전달하도록 `main.py` 로직을 수정한다.

**G0.2 시냅스별 상태 관리 구조체 (치명적 오류 수정)**
* [ ] 현재 `RollingSpikeBuffer`가 시나리오 클래스에 단 하나만 존재하여 모든 시냅스가 이력을 공유하는 오류가 있다.
* [ ] 시냅스 개수($N_{pre} \times N_{post}$)만큼의 버퍼를 관리하는 `SynapseStateManger` 혹은 유사한 구조를 도입한다.
* [ ] 메모리 효율을 위해 실제 Deque 리스트 대신, `(N_synapse, 2, History_L)` 형태의 텐서로 관리하고 `roll` 연산으로 업데이트하는 방식을 권장한다.

**G0.3 Actor-Critic 가중치 업데이트 스케일**
* [ ] 현재는 Actor 출력 `Δd`를 그대로 가중치에 더하고 있다.
* [ ] Theory 2.6에 따라 `Δw = update_scale * Δd` 형태로 스케일링을 적용한다.
* [ ] 흥분/억제 시냅스별로 다른 `update_scale`과 `clipping_range`를 적용할 수 있도록 `optimize_from_trajectory` 로직을 보강한다.

### 1. 시나리오 1.1 (UnsupervisedSinglePolicy) 정렬 작업

**U1.1 Diehl-Cook E/I SNN 네트워크 구현**
* [ ] `run_episode`가 단순히 입력 텐서를 루프 도는 것이 아니라, 실제 뉴런 간 연결을 가진 SNN을 시뮬레이션해야 한다.
* [ ] 다음 구조를 갖는 `DiehlCookNetwork` 클래스(또는 이에 준하는 로직)를 구현한다.
    * 입력층(Poisson) $\to$ 흥분층($N_E$, LIF) : 가변 가중치 (학습 대상)
    * 흥분층 $\to$ 억제층($N_I$, LIF) : 1:1 고정 연결
    * 억제층 $\to$ 흥분층 : All-to-All 가변 가중치 (학습 대상)
* [ ] 시뮬레이션 루프 내에서 발생하는 모든 스파이크 이벤트를 포착하여 Actor-Critic의 입력으로 사용한다.

**U1.2 Winner 결정 및 보상 로직 고도화**
* [ ] `WinnerTracker`를 Theory 3.4.2(다양성)와 3.4.3(안정성)에 맞춰 수정한다.
    * 단순 히스토리가 아닌, 에피소드(이미지) 인덱스별 Winner 기록 저장소 필요.
    * `reward_stability`: 동일 이미지가 재등장했을 때만 이전 Winner와 비교하도록 로직 수정.
* [ ] 희소성 보상($R_{sparse}$) 계산 시 단일 뉴런이 아닌 전체 흥분층의 평균 발화율 $\bar{r}$을 사용하도록 집계 로직을 수정한다.

**U1.3 평가 파이프라인 구축**
* [ ] Diehl-Cook 방식의 **뉴런 라벨링(Neuron Labeling)** 절차를 구현한다.
    * 학습 후 Training Set 전체를 흘려보내며 각 뉴런이 가장 많이 반응한 클래스를 해당 뉴런의 라벨로 할당.
* [ ] Test Set에 대해 `Class Score`를 계산하여 정확도를 측정하는 함수를 작성한다.

### 2. 시나리오 1.2 (UnsupervisedDualPolicy) 정렬 작업

**U2.1 정책 라우팅 구현**
* [ ] U1.1에서 만든 네트워크 구조를 공유하되, 이벤트 발생 시 시냅스 종류에 따라 정책을 분기한다.
    * Input $\to$ Exc 시냅스 이벤트: `actor_exc` 사용
    * Inh $\to$ Exc 시냅스 이벤트: `actor_inh` 사용
* [ ] `run_episode` 내부에서 이벤트 타입(Exc/Inh)을 식별하여 올바른 Actor를 호출하고 Trajectory를 분리 저장한다.

**U2.2 정책별 분석 로그**
* [ ] $\Delta t - \Delta d$ 산점도(STDP 커브)를 흥분성 정책과 억제성 정책 각각 별도 파일로 저장하는 로직을 추가한다.

### 3. 시나리오 2 (SemiSupervisedScenario) 정렬 작업

**S2.1 MLP형 SNN 구조 구현**
* [ ] Theory 5.2에 명시된 `Input(784) -> Hidden(LIF) -> Output(10, LIF)` 구조를 구현한다.
* [ ] 현재 코드처럼 단순히 입력 스파이크만 처리하는 것이 아니라, Hidden 층과 Output 층의 막전위 및 스파이크 역학을 시뮬레이션해야 한다.

**S2.2 마진 기반 보상 정합성 확보**
* [ ] 에피소드 종료 후 Output Layer 10개 뉴런의 발화율을 집계한다.
* [ ] 정답 라벨 뉴런의 발화율($r_y$)과 오답 뉴런 중 최대 발화율($r_{max, wrong}$)을 찾아 마진 $M$을 계산한다.
* [ ] 현재 코드는 단일 뉴런/단순 텐서 기준이므로, 이를 10개 출력 뉴런 기준으로 변경해야 한다.

### 4. 시나리오 3 (GradientMimicryScenario) 정렬 작업

**G3.1 Teacher Network 및 BPTT 구현**
* [ ] 단순 단일 뉴런 예제가 아닌, **Deep SNN (Multi-layer)** 구조를 구현한다.
* [ ] Teacher Network에 대해 `Surrogate Gradient`를 적용한 Backpropagation Through Time (BPTT)를 구현하여, 각 시냅스별 이상적인 그래디언트 $g_i$를 계산한다.
* [ ] 이 $g_i$는 학습되지 않으며, 오직 기준값(Target)으로만 사용됨을 명시한다.

**G3.2 시냅스별 Update 재구성**
* [ ] Actor가 출력한 액션 $\Delta d$를 에피소드 동안 누적하여 에이전트의 총 업데이트량 $\Delta w^{agent}$를 계산한다.
* [ ] Teacher의 제안 업데이트 $\Delta w^{teacher} = -\eta_{align} g_i$ 와의 MSE를 보상으로 계산한다.
* [ ] 현재 코드의 `reward_mimicry` 함수는 맞으나, 입력으로 들어가는 `agent_delta`와 `teacher_delta`가 전체 시냅스에 대해 올바르게 매핑되어야 한다.

### 5. 로깅 및 산출물

**L5.1 파일 기반 로깅 시스템**
* [ ] JSON 형식을 배제하고 텍스트 파일(`metrics.txt`)에 로그를 남긴다.
* [ ] 실험 결과 디렉토리 구조를 `results/<scenario>/<run_name>/` 으로 정형화한다.

**L5.2 필수 시각화 자료**
* [ ] **STDP 커브:** Pre-Post 간격($\Delta t$)에 따른 액션($\Delta d$) 분포 산점도.
* [ ] **학습 곡선:** 에피소드 진행에 따른 Reward 및 Accuracy 변화 그래프.
* [ ] **가중치 분포:** 학습 전후 가중치 히스토그램 비교.
