# AGENTS: RL-SNN 코드베이스와 Theory.md 정렬 작업

Theory.md는 프로젝트의 **단일 진실 소스(single source of truth)** 다.  
이 문서의 작업 항목은 **코드가 Theory.md와 1:1로 대응하도록 정렬시키는 것**을 목표로 한다.

---

## 0. 공통 아키텍처 및 인프라

### G0.1 CLI 하이퍼파라미터 정비

- [ ] `main.py`의 `parse_args()`를 Theory.md 8장의 항목과 일치하도록 확장한다.
  - 시뮬레이션 관련: `--dt`
  - SNN 구조: `--N-E`, `--N-hidden`, LIF 파라미터(`--lif-tau-m`, `--lif-v-threshold`, `--lif-v-reset` 등)
  - 정책/크리틱: `--lr-actor`, `--lr-critic`
  - gradient mimicry: `--alpha-align`, `--log-gradient-stats`
  - 실험 관리: `--seed`, `--log-interval`, `--run-name`
- [ ] 새로 추가된 인자들을 각 시나리오 클래스 생성자 및 내부에서 실제로 사용하도록 연결한다.

### G0.2 공통 LIF 기반 SNN 모듈화

- [ ] Theory.md 3, 4, 5, 6장에서 요구하는 SNN 구조를 지원하는 공통 모듈을 만든다.
  - Diehl–Cook E/I 네트워크 (입력–E–I) 구조
  - 준지도/완전지도용 Input–Hidden–Output 구조
  - 다층 LIF 네트워크 (gradient mimicry용)
- [ ] 각 시나리오가 직접 스파이크 배열을 조작하는 대신,
  - “이미지 → Poisson 인코딩 → SNN 시뮬레이션 → 뉴런 스파이크 배열”을 공통 경로에서 얻도록 리팩토링한다.

### G0.3 로컬 상태 z_i(t)의 “시냅스-로컬”화

- [ ] 현재는 weight와 spike history가 거의 전 시냅스를 공유하는 스칼라/평균값 수준이다.
- [ ] 각 SNN 모듈에서 **시냅스 단위로 pre/post 뉴런 인덱스를 추적**하고,
  - 해당 뉴런의 스파이크 히스토리만 RollingSpikeBuffer에 밀어 넣어
  - `z_i(t) = [h_i(t); w_i(t); e_type; (layer_pos)]`가 **진짜 해당 시냅스의 로컬 정보**를 반영하도록 수정한다.
- [ ] 필요하다면 `RollingSpikeBuffer`를 “시냅스별 인스턴스” 혹은 “시냅스 인덱스로 인덱싱되는 버퍼 배열” 형태로 일반화한다.

### G0.4 Actor–Critic 가중치 업데이트 스케일

- [ ] 현재는 Actor가 내놓은 `Δd`를 그대로 weight에 더하고 있다.
- [ ] Theory.md 2.6 내용에 맞게,
  - 시나리오별/시냅스 타입별 스케일 팩터와 학습률(예: `update_scale`, `synapse_lr`)을 도입하고
  - `Δw = update_scale * Δd` 형태로 weight를 업데이트하도록 수정한다.
- [ ] 흥분/억제 시냅스에 대해 다른 클리핑 범위를 사용하는 현 구조는 유지하되,
  - 스케일 팩터도 타입별로 독립적으로 설정할 수 있게 설계한다.

---

## 1. 시나리오 1.1 (UnsupervisedSinglePolicy) 정렬 작업

### U1.1 Diehl–Cook E/I SNN 실제 구현

- [ ] Theory.md 3.2에 따라 다음 구조를 구현한다.
  - 입력층: 784
  - 흥분층(E): `--N-E`
  - 억제층(I): `N_I = N_E`
  - 연결:
    - Input→E: 학습 가능한 흥분성 시냅스
    - E→I: 1:1 고정 흥분성 시냅스
    - I→E: all-to-all 학습 가능한 억제성 시냅스
- [ ] 이 구조를 사용하는 forward 시뮬레이션(단순한 시간전개 + LIF 업데이트)을 구현한다.
- [ ] `UnsupervisedSinglePolicy.run_episode()`에서
  - 현재처럼 Poisson 첫 픽셀만 쓰는 대신,
  - 위 SNN을 전체 이미지에 대해 시뮬레이션하고,
  - 각 시냅스 이벤트(뉴런 스파이크 발생)를 순회하며 로컬 상태를 구성하고 Actor–Critic을 호출하도록 리팩토링한다.

### U1.2 winner, 희소성, 다양성, 안정성 보상 정합화

- [ ] winner 정의를 Theory와 동일하게,
  - 흥분층 뉴런들의 발화율 벡터 `r_j = S_j / T`에서 `argmax_j r_j`로 얻도록 수정한다.
- [ ] `WinnerTracker`를 “단순 history 리스트”에서
  - **뉴런 수 N_E에 대한 누적 히스토그램**을 유지하는 current 구현은 유지하되,
  - 안정성 보상을 위해 “이미지 인덱스 → 마지막 winner”를 따로 저장하는 구조(예: dict)로 확장한다.
- [ ] `reward_stability`를 Theory 3.4.3 정의에 맞게
  - 같은 이미지가 재방문되었을 때만 이전 winner와 비교하도록 변경한다.

### U1.3 평가 및 산출물 파이프라인

- [ ] Diehl–Cook 뉴런 라벨링 절차를 별도 함수로 구현한다.
  - 학습된 모델에 대해 train 데이터를 흘려보내며 뉴런별 클래스 평균 반응량 `\bar r_{j,y}`를 계산
  - 뉴런 라벨 `L_j` 할당 후, 기준에 따라 예측 라벨 `\hat y` 계산
- [ ] train/val/test accuracy를 측정하고
  - 에피소드 수 혹은 “사용한 이미지 수”에 따른 곡선을 그려 파일로 저장한다.
- [ ] pre/post 스파이크 타임스탬프와 Actor 액션 `Δd`를 이용해
  - Theory 3.6에 맞는 Δt–Δd 산점도를 저장한다 (Scenario 1.1 전용).
- [ ] 모든 실험 설정, 최종 보상/accuracy, 주요 하이퍼파라미터를
  - **텍스트 로그 파일(예: `.txt`)**로 남긴다(JSON 사용 금지).

---

## 2. 시나리오 1.2 (UnsupervisedDualPolicy) 정렬 작업

### U2.1 억제 시냅스 경로 활성화

- [ ] 1.1에서 구현한 Diehl–Cook SNN 구조를 재사용한다.
- [ ] `UnsupervisedDualPolicy.run_episode()`에서
  - Input→E 시냅스 이벤트는 `actor_exc`로
  - I→E 시냅스 이벤트는 `actor_inh`로 라우팅하도록 수정한다.
- [ ] `main.py`에서 `run_episode` 호출 시,
  - 현재처럼 `pre_exc=encoded[:,0]`, `pre_inh=0` 같은 데모 인자를 넘기지 말고
  - SNN 시뮬레이터가 생성한 실제 excitatory/inhibitory pre/post 스파이크를 넘기도록 변경한다.

### U2.2 정책별 통계 및 비교 로그

- [ ] 정책별 Δt–Δd 산점도를 별도 파일로 저장한다 (`pi_exc`, `pi_inh` 각각).
- [ ] 학습 전/후 흥분성/억제성 가중치 분포(히스토그램)를 정책별로 분리하여 저장한다.
- [ ] Scenario 1.1과 동일한 뉴런 라벨링 기반 accuracy 측정을 수행하고,
  - 두 시나리오(단일 정책 vs 두 정책)를 공통 축에서 비교하는 그래프를 작성한다.

---

## 3. 시나리오 2 (SemiSupervisedScenario) 정렬 작업

### S2.1 Hidden–Output LIF SNN 구축

- [ ] Theory.md 5.2에 따라
  - 입력 784 – Hidden LIF(`--N-hidden`) – Output LIF(10) 구조를 구현한다.
- [ ] 출력층 뉴런 인덱스와 클래스 라벨(0~9)을 1:1로 매핑한다.
- [ ] `SemiSupervisedScenario.run_episode()`에서
  - 현재 사용하는 “픽셀 스파이크” 대신,
  - 위 SNN에 Poisson 인코딩된 이미지를 주입하여 나온 Hidden/Output 스파이크를 기반으로 이벤트/보상을 계산하도록 수정한다.

### S2.2 분류 보상/마진 계산 바로잡기

- [ ] 에피소드 종료 후,
  - 출력층 뉴런별 발화율 `r_k = s_k / T_semi`를 계산한다.
  - 예측 라벨 `\hat y = argmax_k r_k`를 사용한다.
- [ ] 마진을
  - `M = r_y - max_{k≠y} r_k`로 계산하고,
  - `reward_classification(predicted == y, M, beta_margin)`에 그대로 전달한다.
- [ ] `reward_classification` 구현은 Theory 5.4와 이미 일치하므로, 호출 인자를 올바른 출력 뉴런 기준으로 맞추는 것이 핵심이다.

### S2.3 산출물 생성

- [ ] train/val/test accuracy를 에피소드 혹은 epoch에 따라 기록하는 곡선을 생성한다.
- [ ] 마진 분포 히스토그램을 학습 전/후 두 시점에서 저장하여 출력층 뉴런 분리 정도를 시각화한다.
- [ ] STDP Δt–Δd 산점도 및 가중치 히스토그램을 저장한다.
- [ ] 텍스트 로그에
  - 최종 분류 성능
  - 사용한 하이퍼파라미터
  - 학습 곡선 요약
  을 기록한다.

---

## 4. 시나리오 3 (GradientMimicryScenario) 정렬 작업

### G3.1 Teacher SNN 구조 확장

- [ ] Teacher 네트워크를 Theory 6.2에 맞게
  - 입력 784 – 여러 LIF 히든층 – 출력 10 LIF 뉴런 구조로 확장한다.
- [ ] supervised loss는
  - 출력 발화율 `r_k`에 softmax(α r_k)를 적용한 뒤 cross-entropy로 정의한다.
- [ ] BPTT를 통해 각 시냅스 weight에 대한 gradient `g_i = ∂L_sup/∂w_i`를 계산하도록 수정한다.

### G3.2 로컬 정책과 시냅스별 업데이트

- [ ] 현재는 단일 스칼라 action을 전 weight에 broadcast하고 있다.
- [ ] SNN에서 발생하는 각 시냅스 이벤트(e = (i,t,type))마다
  - 해당 시냅스의 weight `w_i`
  - 해당 pre/post 뉴런의 스파이크 히스토리
  - layer_pos (정규화된 레이어 인덱스)
  를 기반으로 Actor–Critic을 호출하여 **시냅스별 `Δd_i(t)`를 샘플**하도록 수정한다.
- [ ] 에피소드 동안의 `Δd_i(t)`를 합산하여 `Δw_i^{agent}`를 얻고,
  - Teacher의 `Δw_i^{teacher} = -η_align g_i`와 비교하여 보상을 계산한다.

### G3.3 gradient mimicry 보상, baseline 학습 및 로그

- [ ] `reward_mimicry`는 이미 `-mean((Δw_agent - Δw_teacher)^2)` 형태이므로,
  - 위에서 재구성한 per-synapse `Δw_i` 벡터를 그대로 사용한다.
- [ ] Theory 6.6에 따라
  - 동일 구조의 SNN을 Teacher gradient로만 supervised 학습시키는 baseline을 추가 구현하고,
  - RL 기반 모델과의 train/val/test accuracy 및 loss 비교 곡선을 저장한다.
- [ ] `--alpha-align`, `--log-gradient-stats` CLI 인자를 연결하여
  - gradient–Δw alignment 통계를 텍스트 로그로 남긴다.

---

## 5. 로깅/결과 디렉터리 구조

### L5.1 결과 디렉터리 및 로그 파일

- [ ] `results/<scenario>/<state-type>/<policy-type>/<run-name>/` 형태의 결과 디렉터리를 도입하고,
  - 각 시나리오가 산출물을 해당 디렉터리에 쓰도록 한다.
- [ ] 각 run마다 텍스트 로그 파일(예: `metrics.txt`)을 생성하여
  - 최종 보상, accuracy, 주요 하이퍼파라미터, seed, 실행 시간 등을 기록한다.
- [ ] JSON 형식 로그는 사용하지 않는다(Theory.md 규칙 유지).

### L5.2 STDP 및 기타 시각화 저장

- [ ] `spike_delta_times(pre_spikes, post_spikes)` 유틸을 실제 SNN 스파이크에 연결하여
  - Δt–Δd 산점도를 시나리오별로 저장한다.
- [ ] 가중치 히스토그램, 마진 히스토그램, gradient–Δw alignment 그래프 등
  - Theory.md 각 시나리오 섹션에서 요구하는 시각화를 모두 생성한다.

---

## 6. 재현성 및 실험 관리

### R6.1 시드 고정 및 로그 주기

- [ ] `--seed` 인자를 받아 torch, numpy, random의 시드를 고정한다.
- [ ] `--log-interval`마다
  - 현재 에피소드/epoch, 최근 reward/accuracy, 주요 통계를 로그로 남기고
  - 필요시 stdout과 파일 양쪽에 기록한다.

### R6.2 문서 동기화

- [ ] 위 작업들 완료 후,
  - Theory.md를 절대적 기준으로 하여 실제 구현 사이에 다시 차이가 없는지 점검한다.
