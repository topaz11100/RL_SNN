# 함수 및 역할 설명

## utils 모듈
### utils/cli.py
- `build_parser() -> argparse.ArgumentParser`
  - **인수**: 없음.
  - **역할**: `Theory.md` 2장 전역 CLI 요구 사항에 맞춰 시나리오, 타임스텝, PPO/보상 하이퍼파라미터를 받을 수 있는 파서 생성.
  - **출력**: 구성된 `ArgumentParser` 객체.
  - **Theory 연계**: 실험 실행 시 필요한 시드, 타임스텝(`T_*`), 스파이크 히스토리 길이 L, Gaussian 정책 σ, PPO ε/에폭 등 `Theory.md` 2.9절의 PPO 설정과 보상 가중치 인자를 노출한다.
- `parse_args() -> argparse.Namespace`
  - **인수**: 없음.
  - **역할**: 위 파서를 이용해 CLI 인수를 파싱.
  - **출력**: 파싱된 네임스페이스.
  - **Theory 연계**: 재현성 있는 실험 설정을 위해 모든 시나리오 공통 하이퍼파라미터를 수집한다.

### utils/seeds.py
- `set_global_seed(seed: int, deterministic: Optional[bool] = True) -> None`
  - **인수**: `seed`(고정할 시드), `deterministic`(CuDNN 결정적 모드 설정 여부).
  - **역할**: Python `random`, NumPy, PyTorch(CPU/GPU) 시드를 동일하게 설정하고 CuDNN 결정적 실행을 옵션으로 적용.
  - **출력**: 없음.
  - **Theory 연계**: `Theory.md` 전역 제약의 재현성 요구 사항을 구현.

### utils/logging.py
- `create_result_dir(scenario: str, run_name: Optional[str] = None) -> str`
  - **인수**: `scenario`(실험 시나리오 이름), `run_name`(옵션; 미지정 시 timestamp 사용).
  - **역할**: `results/{scenario}/{run}` 경로를 생성하여 로그/메트릭 파일 저장 위치를 마련.
  - **출력**: 생성된 디렉터리 경로 문자열.
  - **Theory 연계**: `AGENTS.md`의 결과 디렉터리 규칙을 반영.
- `get_logger(log_dir: str) -> logging.Logger`
  - **인수**: `log_dir`(로그를 남길 디렉터리).
  - **역할**: `log.txt`에 기록하고 콘솔에도 출력하는 로거 반환.
  - **출력**: 설정된 `Logger` 객체.
  - **Theory 연계**: 텍스트 기반 로그 저장 요구를 충족.

## data 모듈
### data/mnist.py
- `get_mnist_dataloaders(batch_size_images: int, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader]`
  - **인수**: `batch_size_images`(이미지 배치 크기), `seed`(train/val 분할 시 재현성용 시드).
  - **역할**: MNIST를 다운로드하고 train/val/test DataLoader를 생성하며 train을 고정된 비율로 분할.
  - **출력**: `(train_loader, val_loader, test_loader)`.
  - **Theory 연계**: 모든 시나리오의 입력 데이터 준비 단계.

## snn 모듈
### snn/lif.py
- `LIFParams`
  - **필드**: `tau`, `v_th`, `v_reset`, `v_rest`, `dt`, `R`.
  - **역할**: `Theory.md`의 LIF 미분방정식 파라미터 묶음 데이터클래스.
  - **Theory 연계**: 막전위 업데이트 식 `v[t+1] = v[t] + (-(v[t]-v_rest)+R*I_syn)*dt/tau`를 정의하는 상수.
- `lif_step(v: Tensor, I_syn: Tensor, params: LIFParams) -> Tuple[Tensor, Tensor]`
  - **인수**: 현재 막전위 `v`, 시냅스 전류 `I_syn`, 파라미터 `params`.
  - **역할**: 한 타임스텝 동안 막전위 적분 후 임계치 초과 시 스파이크 생성 및 리셋.
  - **출력**: `(v_next, spikes)`.
  - **Theory 연계**: `Theory.md` 2.2절의 LIF 업데이트와 발화/리셋 규칙 구현.
- `lif_forward(I: Tensor, params: LIFParams) -> Tuple[Tensor, Tensor]`
  - **인수**: 입력 전류 `I`(배치×뉴런×시간), 파라미터 `params`.
  - **역할**: 타임스텝 반복으로 벡터화된 막전위/스파이크 시퀀스를 시뮬레이션.
  - **출력**: `(V, S)` 막전위와 스파이크열.
  - **Theory 연계**: LIF 셀을 시간 전개하여 시나리오별 네트워크 동역학을 재현.

### snn/encoding.py
- `poisson_encode(images: Tensor, T: int, max_rate: float = 1.0) -> Tensor`
  - **인수**: `images`(배치×1×28×28), `T`(시뮬레이션 길이), `max_rate`(픽셀 당 최대 발화 확률).
  - **역할**: 픽셀 밝기에 비례한 확률로 독립 Poisson 스파이크열을 생성해 입력 시퀀스로 변환.
  - **출력**: `(batch, 784, T)` 스파이크 텐서.
  - **Theory 연계**: `Theory.md` 2.1절의 Poisson 입력 인코딩 단계.

### snn/network_diehl_cook.py
- `DiehlCookNetwork.__init__(n_input: int = 784, n_exc: int = 100, n_inh: int = 100, exc_params: Optional[LIFParams] = None, inh_params: Optional[LIFParams] = None)`
  - **역할**: 입력→흥분(E)→억제(I) 가중치 매트릭스와 LIF 파라미터 초기화.
  - **Theory 연계**: 시나리오 1.x의 Diehl–Cook E/I 구조 정의.
- `forward(input_spikes: Tensor) -> Tuple[Tensor, Tensor]`
  - **인수**: `input_spikes`(배치×784×T).
  - **역할**: T 스텝 동안 E/I 층 막전위와 스파이크를 순차적으로 업데이트하여 흥분/억제 스파이크열을 반환.
  - **출력**: `(exc_spikes, inh_spikes)`.
  - **Theory 연계**: `Theory.md` 2.3절의 E/I 순환 및 억제 피드백 구현.

### snn/network_semi_supervised.py
- `SemiSupervisedNetwork.__init__(n_input: int = 784, n_hidden: int = 256, n_output: int = 10, hidden_params: Optional[LIFParams] = None, output_params: Optional[LIFParams] = None)`
  - **역할**: 입력→은닉→출력 LIF 계층과 가중치 초기화.
  - **Theory 연계**: 시나리오 2 분류용 SNN 구조.
- `forward(input_spikes: Tensor) -> Tuple[Tensor, Tensor]`
  - **인수**: `input_spikes`(배치×784×T).
  - **역할**: 은닉/출력 LIF를 시간 전개하여 출력 스파이크와 평균 발화율 산출.
  - **출력**: `(output_spikes, firing_rates)`.
  - **Theory 연계**: `Theory.md` 2.4절의 준지도 분류 SNN 시뮬레이션.

### snn/network_grad_mimicry.py
- `_surrogate_heaviside(x: Tensor, slope: float = 5.0) -> Tensor`
  - **역할**: 스파이크 비선형을 부드럽게 근사하는 시그모이드 surrogate.
  - **Theory 연계**: 시나리오 3에서 BPTT를 가능하게 하는 surrogate gradient 기법.
- `GradMimicryNetwork.__init__(n_input: int = 784, n_hidden: int = 256, n_output: int = 10, hidden_params: Optional[LIFParams] = None, output_params: Optional[LIFParams] = None)`
  - **역할**: Teacher/에이전트 공유용 입력-은닉-출력 가중치와 파라미터 초기화.
  - **Theory 연계**: `Theory.md` 6.4절의 gradient mimicry 실험 네트워크.
- `forward(input_spikes: Tensor) -> Tuple[Tensor, Tensor]`
  - **인수**: `input_spikes`(배치×784×T).
  - **역할**: surrogate 스파이크를 사용해 은닉/출력 발화율을 계산, 역전파 가능하게 유지.
  - **출력**: `(output_spikes, firing_rates)`.
  - **Theory 연계**: Teacher gradient 계산과 에이전트 시뮬레이션을 위한 differentiable forward.

## rl 모듈
### rl/policy.py
- `_CNNFront.forward(spike_history: torch.Tensor) -> torch.Tensor`
  - **인수**: `spike_history`(배치×2×L; pre/post 히스토리).
  - **역할**: `Theory.md` 2.7절의 1D CNN 전처리로 시간 평균 특징 추출.
  - **출력**: `(batch, 16)` 특징 벡터.
- `GaussianPolicy.__init__(sigma: float, extra_feature_dim: int = 0)`
  - **역할**: CNN 전단과 MLP 헤드를 구성하고 정책 분포 표준편차를 버퍼로 저장.
  - **Theory 연계**: `Theory.md` 2.7절 Gaussian Actor 구조.
- `GaussianPolicy.forward(spike_history: torch.Tensor, extra_features: Optional[torch.Tensor] = None, actions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
  - **역할**: 스파이크 히스토리(및 추가 특징)를 인코딩해 평균 `mean`을 산출, 주어진 액션의 로그확률 또는 샘플 액션을 반환.
  - **출력**: `(action, log_prob, mean)`.
  - **Theory 연계**: 상태 z로부터 Gaussian 정책 π(a|z)를 평가/샘플링.

### rl/value.py
- `_CNNFront.forward(spike_history: torch.Tensor) -> torch.Tensor`
  - **역할**: 정책과 동일한 CNN 구조로 특징 추출.
  - **Theory 연계**: Actor와 같은 입력 표현을 쓰는 Critic 요구사항.
- `ValueFunction.__init__(extra_feature_dim: int = 0)`
  - **역할**: CNN+MLP 구성으로 V(z) 예측기 정의.
  - **Theory 연계**: `Theory.md` 2.8절 Critic 구조.
- `ValueFunction.forward(spike_history: torch.Tensor, extra_features: Optional[torch.Tensor] = None) -> torch.Tensor`
  - **역할**: 상태 특징을 결합해 스칼라 가치 추정.
  - **출력**: `(batch,)` 값 함수.
  - **Theory 연계**: PPO의 베이스라인 추정.

### rl/buffers.py
- `EpisodeBuffer.__init__()`
  - **역할**: 상태/행동/로그확률/가치/보상 리스트 초기화.
  - **Theory 연계**: 에피소드 단위 rollout 저장.
- `append(state: torch.Tensor, action: torch.Tensor, log_prob: torch.Tensor, value: torch.Tensor) -> None`
  - **역할**: 이벤트별 기록을 detach하여 버퍼에 추가.
  - **Theory 연계**: MC PPO에서 θ_old 정보를 유지.
- `finalize(R: torch.Tensor) -> None`
  - **역할**: 에피소드 전역 보상 R을 모든 이벤트에 브로드캐스트하여 저장.
  - **Theory 연계**: γ=1 MC 리턴 공유.
- `get_batch() -> Tuple[torch.Tensor, ...]`
  - **역할**: 스택된 텐서 배치를 반환; 보상 미설정 시 오류.
  - **Theory 연계**: PPO 업데이트 입력 준비.
- `__len__() -> int`
  - **역할**: 저장된 이벤트 수 반환.

### rl/ppo.py
- `ppo_update(actor: nn.Module, critic: nn.Module, buffer: EpisodeBuffer, optimizer_actor: torch.optim.Optimizer, optimizer_critic: torch.optim.Optimizer, ppo_epochs: int, batch_size: int, eps_clip: float = 0.2, c_v: float = 1.0, extra_features: Optional[torch.Tensor] = None)`
  - **인수**: Actor/Critic 모듈, 버퍼, 최적화기, PPO 에폭/배치, 클리핑 ε, 가치 가중치 `c_v`, 선택적 추가 특징.
  - **역할**: 버퍼 배치를 섞어가며 클립드 PPO 손실과 가치 MSE를 계산, 두 네트워크를 각각 업데이트.
  - **출력**: 없음.
  - **Theory 연계**: `Theory.md` 2.9절의 PPO Actor 손실 `-min(rA, clip(r)A)`와 Critic MSE 구현.

## scenarios 모듈
### scenarios/unsup_single.py
- `_ensure_metrics_file(path: str) -> None`
  - **역할**: 비지도 단일 정책 학습용 메트릭 파일에 헤더를 생성.
  - **Theory 연계**: 결과 텍스트 로그 요구.
- `_compute_rewards(exc_spikes: torch.Tensor, rho_target: float, alpha_sparse: float, alpha_div: float, alpha_stab: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`
  - **역할**: 발화율 기반 희소성, 다양성, 안정성 보상을 계산하고 가중합을 반환.
  - **Theory 연계**: `Theory.md` 2.5절 비지도 보상 구성요소.
- `_prepare_state(input_spikes: torch.Tensor, exc_spikes: torch.Tensor, L: int) -> torch.Tensor`
  - **역할**: 입력/흥분 스파이크 평균을 최근 L 타임스텝 히스토리로 묶어 상태 z를 생성.
  - **Theory 연계**: 로컬 상태 벡터 정의(전/후 스파이크 히스토리).
- `run_unsup1(args, logger)`
  - **역할**: MNIST 로딩, Poisson 인코딩, Diehl–Cook 네트워크 시뮬레이션, 상태→정책/가치 평가, 보상 계산, PPO 업데이트, 메트릭 로깅까지 한 에포크 루프 수행.
  - **출력**: 없음(로그/파일 기록).
  - **Theory 연계**: 시나리오 1.1 전체 학습 흐름 구현.

### scenarios/unsup_dual.py
- `_ensure_metrics_file(path: str) -> None`
  - **역할**: 비지도 이중 정책 메트릭 헤더 생성.
- `_compute_rewards(...)`
  - **역할**: 흥분층 스파이크로 희소성/다양성/안정성 보상 계산.
  - **Theory 연계**: 시나리오 1.2 보상 동일.
- `_prepare_state(pre_spikes: torch.Tensor, post_spikes: torch.Tensor, L: int) -> torch.Tensor`
  - **역할**: 선택한 pre/post 스파이크 평균으로 상태 z 구성(입력→E, I→E 각각 사용).
  - **Theory 연계**: 로컬 상태 정의.
- `run_unsup2(args, logger)`
  - **역할**: 두 정책(흥분/억제)으로 Poisson 인코딩된 배치를 처리하고, 보상을 기반으로 각 정책/가치를 PPO 업데이트하며 메트릭을 기록.
  - **Theory 연계**: 시나리오 1.2의 분리된 정책 학습 파이프라인.

### scenarios/semi_supervised.py
- `_ensure_metrics_file(path: str, header: str) -> None`
  - **역할**: train/val/test 메트릭 파일 헤더 생성.
- `_prepare_state(pre_spikes: torch.Tensor, post_spikes: torch.Tensor, L: int) -> torch.Tensor`
  - **역할**: 입력/출력 스파이크 평균으로 상태 z 생성.
- `_compute_reward_components(firing_rates: torch.Tensor, labels: torch.Tensor, beta_margin: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
  - **역할**: 분류 정확 보상 `R_cls`, 마진 `M` 기반 보상 `R_margin`, 합산 `R_total`을 계산.
  - **Theory 연계**: `Theory.md` 5.3절 분류 보상 정의.
- `_evaluate(network: SemiSupervisedNetwork, loader, device, args) -> Tuple[float, float, float]`
  - **역할**: 주어진 데이터 로더에서 발화율로 예측 후 정확도, 마진, 보상 평균을 산출.
  - **Theory 연계**: 준지도 시나리오 평가 절차.
- `run_semi(args, logger)`
  - **역할**: Poisson 인코딩된 MNIST와 라벨을 사용해 네트워크 시뮬레이션, 보상 계산, PPO 업데이트, train/val/test 메트릭 기록.
  - **Theory 연계**: 시나리오 2 학습 루프.

### scenarios/grad_mimicry.py
- `_ensure_metrics_file(path: str, header: str) -> None`
  - **역할**: gradient mimicry 시나리오용 메트릭 헤더 작성.
- `_prepare_state(pre_spikes: torch.Tensor, post_spikes: torch.Tensor, L: int) -> torch.Tensor`
  - **역할**: 입력/출력 스파이크 평균 히스토리로 상태 z 생성.
- `_compute_reward(delta_agent: torch.Tensor, delta_teacher_in: torch.Tensor, delta_teacher_out: torch.Tensor) -> torch.Tensor`
  - **역할**: 에이전트 업데이트와 Teacher 업데이트의 차이를 음의 제곱으로 계산해 정렬 보상 산출.
  - **Theory 연계**: `Theory.md` 6.4절 보상 `-(Δw_agent - Δw_teacher)^2` 평균.
- `_evaluate(network: GradMimicryNetwork, loader, device, args) -> Tuple[float, float]`
  - **역할**: 발화율 기반 예측 정확도와 마진 기반 보상 근사치를 평가.
- `run_grad(args, logger)`
  - **역할**: RL 네트워크 forward→정책 업데이트, Teacher surrogate gradient 계산, 정렬 보상 산출, PPO 업데이트, train/val/test 메트릭 기록.
  - **Theory 연계**: 시나리오 3의 gradient mimicry 학습 루프.

## main.py
- `main()`
  - **역할**: CLI 파싱, 시드 설정, 결과 디렉터리 생성 및 로거 초기화 후 시나리오별 러너를 호출.
  - **Theory 연계**: 네 시나리오를 선택적으로 실행하는 진입점.
