# 함수 및 역할 설명

## utils 모듈
### utils/cli.py
- `build_parser() -> argparse.ArgumentParser`
  - **인수**: 없음.
  - **역할**: `Theory.md` 2장 전역 CLI 요구 사항에 맞춰 시나리오, 타임스텝, PPO/보상 하이퍼파라미터를 받을 수 있는 파서 생성. 기본 타임스텝을 100으로 설정해 스파이크 히스토리 길이 `L`(기본 20)과의 제약 `L ≤ T`를 만족하도록 한다.
  - **추가 인자**: `--events-per-image`로 이미지당 최대 이벤트 수(K)를 제한해 Theory 2.9.3의 리저버 샘플링 요구를 충족하며, 지도/준지도 클립 인자는 `--w-clip-min/max`로 노출해 문서 표기와 일치시킨다.
  - **출력**: 구성된 `ArgumentParser` 객체.
  - **Theory 연계**: 실험 실행 시 필요한 시드, 타임스텝(`T_*`), 스파이크 히스토리 길이 L, Gaussian 정책 σ, PPO ε/에폭 등 `Theory.md` 2.9절의 PPO 설정과 보상 가중치 인자를 노출한다.
  - **최적화**: PPO 이벤트 미니배치 기본값을 512로 높여(`--ppo-batch-size`) 대규모 이벤트 스트림에서도 그라디언트 노이즈를 줄이고 안정적인 SGD를 유지한다.
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

### utils/event_utils.py
- `gather_events(pre_spikes: torch.Tensor, post_spikes: torch.Tensor, weights: torch.Tensor, window: int, buffer: EventBatchBuffer, connection_id: int, *, l_norm: float | None = None, valid_mask: torch.Tensor | None = None, padded_pre: torch.Tensor | None = None, padded_post: torch.Tensor | None = None, max_pairs: int = 131072) -> None`
  - **역할**: `Theory.md` 2.7절의 pre/post 스파이크 히스토리를 희소 인덱싱으로 수집하고, 반환 없이 `EventBatchBuffer.reserve`에서 받은 슬라이스에 직접 기록한다. `max_pairs` 블록 처리와 `padded_pre/post` 재사용으로 거대 임시 텐서와 중복 패딩을 제거하며, extras에는 호출 시점의 가중치 스냅샷(+선택적 정규화, 이벤트 타입 원핫)을 저장해 시뮬레이션 타이밍의 파라미터가 Actor 평가 시점에 그대로 전달됨을 코드 상에서 보증한다.
  - **출력**: 없음. 스파이크 히스토리는 `torch.bool`로 유지해 VRAM/대역폭을 1/4 수준으로 줄이고, Actor/Critic는 forward 직전에 `float()` 캐스팅 후 CNN 처리한다.
  - **Theory 연계/최적화**: triple-copy(리스트→`torch.cat`→`buffer.add`) 제거로 GPU 메모리 대역폭과 커널 런칭 오버헤드를 줄이며, `Theory.md` 2.9.3의 이벤트 미니배치 규칙을 유지한다. `padded_pre/padded_post`는 호출자가 사전 캐싱한 패딩 텐서를 넘겨 중복 `F.pad` 실행을 없애도록 설계되어, 동일 스파이크 텐서가 여러 연결에서 재사용될 때 GPU 커널 발행을 최소화한다.

## data 모듈
### data/mnist.py
- `get_mnist_dataloaders(batch_size_images: int, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader]`
  - **인수**: `batch_size_images`(이미지 배치 크기), `seed`(train/val 분할 시 재현성용 시드).
  - **역할**: MNIST를 다운로드하고 train/val/test DataLoader를 생성하며 train을 고정된 비율로 분할한다. 각 샘플은 `(image, label, index)` 튜플로 반환되어 안정성 보상 계산 시 동일 이미지의 이전 winner 뉴런을 추적할 수 있다.
  - **출력**: `(train_loader, val_loader, test_loader)`.
  - **Theory 연계**: 모든 시나리오의 입력 데이터 준비 단계 및 R_stab 계산을 위한 데이터셋 인덱스 제공.

## snn 모듈
### snn/lif.py
- `LIFParams`
  - **필드**: `tau`, `v_th`, `v_reset`, `v_rest`, `dt`, `R`.
  - **역할**: `Theory.md`의 LIF 미분방정식 파라미터 묶음 데이터클래스.
  - **Theory 연계**: 막전위 업데이트 식 `v[t+1] = v[t] + (-(v[t]-v_rest)+R*I_syn)*dt/tau`를 정의하는 상수.
- `lif_dynamics(v: Tensor, I: Tensor, params: LIFParams, surrogate: bool = False, slope: float = 5.0) -> Tuple[Tensor, Tensor]`
  - **인수**: 현재 막전위 `v`, 시냅스 전류 `I`, 파라미터 `params`, surrogate 사용 여부와 시그모이드 기울기 `slope`.
- **역할**: `Theory.md` 2.2절 수식에 따라 막전위를 적분하고, 업데이트된 막전위 기준으로 hard/surrogate 스파이크를 생성한 뒤 detach 기반 리셋을 적용한다. 모든 단계는 out-of-place로 계산해 `torch.func.vmap`과 같은 함수형 변환에서도 상태 공유가 일어나지 않는다.
  - **출력**: `(v_next, spikes)`.
  - **Theory 연계**: 공통 LIF 동역학 구현부로 모든 네트워크가 공유하는 단일 소스.
- `lif_step(v: Tensor, I_syn: Tensor, params: LIFParams) -> Tuple[Tensor, Tensor]`
  - **인수**: 현재 막전위 `v`, 시냅스 전류 `I_syn`, 파라미터 `params`.
  - **역할**: hard Heaviside 스파이크를 사용하는 한 타임스텝 LIF 업데이트 래퍼. 내부적으로 `lif_dynamics(..., surrogate=False)`를 호출한다.
  - **출력**: `(v_next, spikes)`.
  - **Theory 연계**: `Theory.md` 2.2절의 LIF 업데이트와 발화/리셋 규칙 구현.
- `lif_forward(I: Tensor, params: LIFParams) -> Tuple[Tensor, Tensor]`
  - **인수**: 입력 전류 `I`(배치×뉴런×시간), 파라미터 `params`.
  - **역할**: 타임스텝 반복으로 벡터화된 막전위/스파이크 시퀀스를 시뮬레이션.
  - **출력**: `(V, S)` 막전위와 스파이크열.
  - **Theory 연계**: LIF 셀을 시간 전개하여 시나리오별 네트워크 동역학을 재현.
- `LIFCell.__init__(params: LIFParams, surrogate: bool = False, slope: float = 5.0)` / `forward(v: Tensor, I: Tensor) -> Tuple[Tensor, Tensor]`
  - **역할**: 학습 파라미터 없이 한 스텝 LIF 동역학을 수행하는 모듈. surrogate=True 시 시그모이드 근사 스파이크를 사용하며 reset에서는 gradient가 끊기도록 `lif_dynamics`를 호출한다.
  - **Theory 연계**: `Theory.md` 2.2절 LIF 셀을 모듈화해 모든 네트워크에서 동일한 동역학을 적용.

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
- **역할**: T 스텝 동안 E/I 층 막전위와 스파이크를 순차적으로 업데이트하여 흥분/억제 스파이크열을 반환. 변하지 않는 억제→흥분 마스크는 루프 외부에서 한 번만 적용해 타임스텝 반복 시의 불필요한 연산을 줄였다.
  - **출력**: `(exc_spikes, inh_spikes)`.
  - **Theory 연계**: `Theory.md` 2.3절의 E/I 순환 및 억제 피드백 구현.
  - **최적화**: 입력 스파이크×가중치 matmul을 시간 루프 밖에서 한 번만 계산해 `I_exc_all`로 저장하고, 루프 내에서는 슬라이싱만 사용해 GPU 커널 발행을 최소화한다.

### snn/network_semi_supervised.py
- `SemiSupervisedNetwork.__init__(n_input: int = 784, n_hidden: int = 256, n_output: int = 10, hidden_params: Optional[LIFParams] = None, output_params: Optional[LIFParams] = None)`
  - **역할**: 입력→은닉→출력 LIF 계층과 가중치 초기화.
  - **Theory 연계**: 시나리오 2 분류용 SNN 구조.
- `forward(input_spikes: Tensor) -> Tuple[Tensor, Tensor, Tensor]`
  - **인수**: `input_spikes`(배치×784×T).
  - **역할**: 은닉/출력 LIF를 시간 전개하여 은닉/출력 스파이크와 평균 발화율을 산출.
  - **출력**: `(hidden_spikes, output_spikes, firing_rates)`.
  - **Theory 연계**: `Theory.md` 2.4절의 준지도 분류 SNN 시뮬레이션.
  - **최적화**: 입력→은닉, 은닉→출력 matmul을 시간 루프 밖에서 각각 `I_hidden_all`, `I_output_all`로 계산해 타임스텝마다의 반복 matmul을 제거하고 GPU 커널 런칭을 줄였다.

### snn/network_grad_mimicry.py
- `GradMimicryNetwork.__init__(n_input: int = 784, hidden_sizes: Optional[List[int]] = None, n_output: int = 10, hidden_params: Optional[LIFParams] = None, output_params: Optional[LIFParams] = None)`
  - **역할**: Teacher/에이전트 공유용 입력-다중 은닉-출력 가중치와 파라미터 초기화, surrogate LIF 셀 생성.
  - **Theory 연계**: `Theory.md` 6.4절의 gradient mimicry 실험 네트워크.
- `forward(input_spikes: Tensor) -> Tuple[List[Tensor], Tensor, Tensor]`
  - **인수**: `input_spikes`(배치×784×T).
  - **역할**: 타임스텝마다 surrogate LIF로 은닉/출력 막전위를 적분하면서 `(batch, neurons, T)` 크기의 버퍼를 **사전 할당**하고 인덱싱으로 채운다. 본 로직을 `@torch.jit.script` 함수로 분리해 타임루프를 커널 수준으로 퓨전, Python 오버헤드를 제거했고 `T=0`이면 즉시 빈 텐서를 반환한다.
  - **출력**: `([hidden_spikes_per_layer], output_spikes, firing_rates)`로 각 층/출력의 스파이크열과 평균 발화율.
  - **Theory 연계**: Teacher gradient 계산과 에이전트 시뮬레이션을 위한 differentiable forward.
  - **최적화**: 입력→은닉, 은닉→출력 행렬곱을 시간 루프 전에 3차원 텐서로 미리 계산하고 루프 내부에서는 슬라이스만 사용해 타임스텝당 matmul을 제거했다.

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
  - **역할**: 스파이크 히스토리(및 추가 특징)를 인코딩해 평균 `mean`을 산출, 주어진 액션의 로그확률 또는 샘플 액션을 반환. 입력 히스토리는 `bool` 저장을 가정하며 forward 내부에서 `float()` 캐스팅 후 CNN에 투입된다.
  - **출력**: `(action, log_prob, mean)`.
  - **Theory 연계**: 상태 z로부터 Gaussian 정책 π(a|z)를 평가/샘플링.

### rl/value.py
- `_CNNFront.forward(spike_history: torch.Tensor) -> torch.Tensor`
  - **역할**: 정책과 동일한 CNN 구조로 특징 추출. `torch.bool` 히스토리를 `float()`로 변환해 CNN에 입력한다.
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
  - **역할**: 스파이크 히스토리, 추가 특징, 행동, 로그확률, 가치, 보상 리스트 초기화.
  - **Theory 연계**: 에피소드 단위 rollout 저장.
- `EventBatchBuffer`
  - **역할**: 이벤트 단위 on-policy 배치를 GPU 상 연속 메모리에 저장한다. 초기 용량을 한 번(기본 4096개) 넉넉히 확보한 뒤 길이가 초과되면 두 배 확장해 리스트 기반 append 대비 메모리 파편화와 재할당을 줄인다.
  - **동작/재사용**: `reserve`가 요청 개수에 맞춰 슬라이스를 반환하며 길이를 즉시 증가시킨다. `gather_events`는 반환 없이 이 슬라이스에 직접 기록해 리스트→cat→copy triple-copy 경로를 제거한다. 스파이크 히스토리는 `torch.bool`로 저장하고 extras/인덱스는 detch된 값을 사용한다. 학습 루프에서는 에포크/배치마다 `reset()`으로 길이만 0으로 돌려 동일한 GPU 버퍼를 재사용하고, `batch_idx`는 항상 현재 이미지 미니배치의 로컬 인덱스(0~B-1)만 허용해 이벤트-이미지 매핑을 유지한다.
  - **Theory 연계/최적화**: `Theory.md` 2.9.3/2.9.4의 이미지 미니배치 기반 이벤트 업데이트를 유지하면서 GPU 메모리 연속성을 확보해 Actor 재연산 시 파편화를 방지한다. `subsample_per_image(k)`로 각 이미지별 최대 `k`개 이벤트만 남기는 리저버 샘플링을 수행해 VRAM 폭주를 막고 per-image PPO 제약을 맞춘다. 에피소드 ID 필드는 미사용이라 제거해 GPU 사용량을 절약했다.
  - **초기 용량 가이드**: 러너들은 `batch_size_images * spike_array_len * (784 + \text{layer widths})` 추정식과 최소 100,000 이벤트 중 큰 값을 사용해 초기 용량을 정해 학습 초반 재할당 오버헤드를 없앤다.
- `append(state: torch.Tensor, extra_features: torch.Tensor, action: torch.Tensor, log_prob: torch.Tensor, value: torch.Tensor) -> None`
  - **역할**: 이벤트별 로컬 상태와 추가 특징을 포함해 detach한 기록을 버퍼에 추가.
  - **Theory 연계**: MC PPO에서 θ_old 정보를 유지하고 상태 z_i(t)에 요구되는 보조 feature를 함께 보존.
- `finalize(R: torch.Tensor) -> None`
  - **역할**: 에피소드 전역 보상 R을 모든 이벤트에 브로드캐스트하여 저장.
  - **Theory 연계**: γ=1 MC 리턴 공유.
- `get_batch() -> Tuple[torch.Tensor, ...]`
  - **역할**: 스택된 상태, 추가 특징, 행동, 로그확률, 가치, 보상 배치를 반환; 보상 미설정 시 오류.
  - **Theory 연계**: PPO 업데이트 입력 준비.
- `extend(other: EpisodeBuffer) -> None`
  - **역할**: 다른 에피소드 버퍼 내용을 순서를 보존한 채 병합하여 이미지 미니배치 단위 업데이트를 가능하게 한다.
- `__len__() -> int`
  - **역할**: 저장된 이벤트 수 반환.

### rl/ppo.py
- `ppo_update(actor: nn.Module, critic: nn.Module, buffer: EpisodeBuffer, optimizer_actor: torch.optim.Optimizer, optimizer_critic: torch.optim.Optimizer, ppo_epochs: int, batch_size: int, eps_clip: float = 0.2, c_v: float = 1.0)`
  - **인수**: Actor/Critic 모듈, 버퍼, 최적화기, PPO 에폭/배치, 클리핑 ε, 가치 가중치 `c_v`.
  - **역할**: 버퍼에 저장된 상태·추가 특징·행동을 섞어가며 클립드 PPO 손실과 가치 MSE를 계산, 두 네트워크를 각각 업데이트한다. 추가 특징은 버퍼 내부에 포함되어 관리한다.
  - **출력**: 없음.
  - **Theory 연계**: `Theory.md` 2.9절의 PPO Actor 손실 `-min(rA, clip(r)A)`와 Critic MSE 구현. 각 미니배치에서 advantage를 `(A - mean(A)) / (std(A) + 1e-8)`로 표준화해 보상 스케일 변화에 대한 학습 안정성을 높인다.
- `ppo_update_events(actor: nn.Module, critic: nn.Module, states: torch.Tensor, extras: torch.Tensor, actions_old: torch.Tensor, log_probs_old: torch.Tensor, returns: torch.Tensor, advantages: torch.Tensor, optimizer_actor: torch.optim.Optimizer, optimizer_critic: torch.optim.Optimizer, ppo_epochs: int, batch_size: int, eps_clip: float = 0.2, c_v: float = 1.0)`
  - **역할**: 이벤트 히스토리 배치를 바로 받아 PPO 업데이트를 수행한다. 입력이 비어 있을 때는 즉시 반환해 보상 계산 흐름을 끊지 않고 GPU/CPU 동기화를 피한다.
  - **Theory 연계**: 이미지 단위 이벤트 미니배치 업데이트(Theory 2.9.3)를 안전하게 수행하며, 미니배치별 advantage 표준화로 스케일 변화에 강한 클립드 PPO를 적용한다.

## scenarios 모듈
### scenarios/unsup_single.py
- `_ensure_metrics_file(path: str) -> None`
  - **역할**: 비지도 단일 정책 학습용 메트릭 파일에 헤더를 생성.
  - **Theory 연계**: 결과 텍스트 로그 요구.
- `run_unsup1(args, logger)`
  - **역할**: MNIST 로딩, Poisson 인코딩, Diehl–Cook 네트워크 시뮬레이션, 이벤트별 로컬 상태(전/후 스파이크 히스토리, 현재 가중치, 이벤트 타입) 수집, per-image 안정성 보상(데이터셋 인덱스 기반 winner 추적 포함)과 희소성/다양성 보상 합산, 에피소드 버퍼 병합 후 PPO 업데이트, 메트릭 로깅까지 한 에포크 루프 수행.
  - **세부 구현**: 에포크 루프 시작 시 `s_scen = 1.0`을 정의하고, 모든 `_scatter_updates` 호출을 `Δw = args.local_lr * s_scen * action` 구조로 적용해 `AGENTS.md` 5.3.2절의 로컬 학습률 수식을 명시적으로 따른다. 스파이크 히스토리는 `utils.event_utils.gather_events`의 희소 인덱싱 경로를 사용해 OOM을 방지한다. 이벤트 버퍼는 에포크 외부에서 한 번 생성 후 매 배치 `reset()`하여 GPU 메모리 재할당 없이 재사용하며, 전체 관측 카운터(`total_seen`) 역시 GPU 스칼라 텐서로 누적해 H2D 전송을 제거했다.
  - **세부 구현**: 에포크 루프 시작 시 `s_scen = 1.0`을 정의하고, 모든 `_scatter_updates` 호출을 `Δw = args.local_lr * s_scen * action` 구조로 적용해 `AGENTS.md` 5.3.2절의 로컬 학습률 수식을 명시적으로 따른다. 스파이크 히스토리는 `utils.event_utils.gather_events`의 희소 인덱싱 경로를 사용해 OOM을 방지하고, 수집 직후 `events_per_image`(K)만큼 리저버 샘플링해 이미지별 이벤트 수를 제한한 뒤 PPO를 수행한다. 이벤트 버퍼는 에포크 외부에서 한 번 생성 후 매 배치 `reset()`하여 GPU 메모리 재할당 없이 재사용하며, 전체 관측 카운터(`total_seen`) 역시 GPU 스칼라 텐서로 누적해 H2D 전송을 제거했다.
  - **최적화**: 보상과 발화율 통계를 GPU 텐서로 누적하고 에포크 종료 시 한 번만 CPU로 이동시켜 평균을 계산해 배치 단위 동기화를 줄였다. per-batch `.item()`/`.cpu()` 호출과 리스트 누적을 제거했으며, Actor/Critic 추론은 `args.event_batch_size` 단위의 이벤트 미니배치로 분할해 긴 스파이크 시퀀스에서도 VRAM 피크를 제어한다. 동일 스파이크 텐서를 `F.pad`한 결과를 캐싱해 다중 연결 수집 시 패딩 커널을 재사용한다.
  - **출력**: 없음(로그/파일 기록).
  - **Theory 연계**: 시나리오 1.1 전체 학습 흐름 구현.
  - **버퍼 용량**: 입력(784)과 흥분 뉴런(`N_E`) 합산에 `batch_size_images`·`spike_array_len`을 곱하고 최소 100k로 하한을 둬 초기 이벤트 버퍼 재할당을 제거한다.

### scenarios/unsup_dual.py
- `_ensure_metrics_file(path: str) -> None`
  - **역할**: 비지도 이중 정책 메트릭 헤더 생성.
- `run_unsup2(args, logger)`
  - **역할**: 두 정책(흥분/억제)으로 Poisson 인코딩된 배치를 처리하고, 전/후 스파이크 히스토리·가중치·이벤트 타입으로 구성한 로컬 상태를 이벤트별로 수집한다. 각 이미지에 대해 희소성/다양성/안정성 보상을 계산해 에피소드 버퍼에 채운 뒤, 미니배치 전체를 대상으로 PPO 업데이트를 수행하며 메트릭을 기록한다.
  - **세부 구현**: 에포크 진입 시 `s_scen = 1.0`을 설정하고, 흥분/억제 경로 모두 `_scatter_updates`에 `args.local_lr * s_scen * action`을 전달해 로컬 업데이트가 문서상의 `η_w * s_scen * Δd_i(t)` 형식을 유지한다. 히스토리 수집은 `gather_events`의 희소 인덱스 버퍼를 사용해 두 경로 모두 메모리 사용을 제한하고, 동일 스파이크 텐서를 사전 패딩/캐싱(`padded_pre/post`)하여 중복 패딩을 제거한다. 수집된 이벤트는 곧바로 `events_per_image` 한도에 맞춰 이미지별 리저버 샘플링을 거친 뒤 PPO에 전달된다. 이벤트 버퍼는 에포크 밖에서 한 번만 생성하고 각 배치마다 `reset()`해 cuda 재할당을 방지하며, 이미지 누적 카운터 `total_seen`도 GPU 텐서로 증가시켜 H2D 동기화를 없앴다.
  - **최적화**: 보상/발화 통계를 GPU 텐서로 유지하고 delta_t/delta_d 히스토리 역시 에포크 종료 시점에만 CPU로 옮겨 기록해 배치마다의 동기화를 제거했다. per-batch `.item()`/`.cpu()` 수집을 없애고 GPU 누적 스칼라/텐서를 사용했으며, Actor/Critic 추론은 이벤트 배치를 `args.event_batch_size`로 나눠 처리해 OOM 위험을 줄였다.
  - **Theory 연계**: 시나리오 1.2의 분리된 정책 학습 파이프라인.
  - **버퍼 용량**: 입력 784와 흥분/억제 뉴런(`2 * N_E`)을 합산한 추정치를 `batch_size_images * spike_array_len`에 곱하고 최소 100k로 보강해 양 경로 이벤트 수집 시 재할당을 방지한다.

### scenarios/semi_supervised.py
- `_ensure_metrics_file(path: str, header: str) -> None`
  - **역할**: train/val/test 메트릭 파일 헤더 생성.
- `_compute_reward_components(firing_rates: torch.Tensor, labels: torch.Tensor, beta_margin: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
  - **역할**: 분류 정확 보상 `R_cls`, 마진 `M` 기반 보상 `R_margin`, 합산 `R_total`을 계산.
  - **Theory 연계**: `Theory.md` 5.3절 분류 보상 정의.
- `_evaluate(network: SemiSupervisedNetwork, loader, device, args) -> Tuple[float, float, float]`
  - **역할**: 주어진 데이터 로더에서 발화율로 예측 후 정확도, 마진, 보상 평균을 산출.
  - **Theory 연계**: 준지도 시나리오 평가 절차.
- `run_semi(args, logger)`
  - **역할**: Poisson 인코딩된 MNIST와 라벨을 사용해 네트워크 시뮬레이션, 출력 이벤트별 로컬 상태(히스토리+가중치+이벤트 타입)를 통해 행동을 산출하고 가중치를 개별적으로 업데이트한다. 각 이미지의 분류 보상을 에피소드 버퍼에 담아 미니배치 단위로 PPO를 업데이트하며 train/val/test 메트릭을 기록한다.
  - **최적화**: `args.event_batch_size`에 맞춘 이벤트 미니배치로 Actor/Critic를 평가해 긴 시퀀스에서도 VRAM 사용을 일정하게 유지한다. 입력/은닉/출력 스파이크 패딩은 한 번만 수행해 `gather_events`에 전달하며, 가중치 업데이트는 `index_put_` 누산으로 GPU 내에서 원자적으로 적용한다. 이벤트 수집 후에는 `events_per_image` K값으로 이미지당 이벤트를 리저버 샘플링하여 PPO에 전달해 VRAM 폭주를 방지한다.
  - **세부 구현**: 에포크 시작에 `s_scen = 1.0`을 선언하고 입력/은닉→출력 경로 모두 `_scatter_updates` 호출을 `args.local_lr * s_scen * action` 형태로 통일해 로컬 학습률 수식을 충족하며, 각 업데이트 후 가중치를 `w_clip_min/max` 범위로 클리핑한다. 로컬 상태 구성은 `gather_events` 기반이라 희소 스파이크만을 대상으로 하며, 이벤트 버퍼를 에포크 외부에서 한 번 생성 후 배치마다 `reset()`해 GPU 재할당 없이 PPO 업데이트에 재사용한다.
  - **최적화**: 정확도·마진·보상을 GPU 텐서로 누적한 뒤 에포크 종료 시 집계해 CPU 동기화 비용을 최소화한다. per-batch `.item()`/`.cpu()` 동기화를 제거하고 누적 스칼라 텐서를 GPU에 유지한다.
  - **Theory 연계**: 시나리오 2 학습 루프.
  - **버퍼 용량**: 입력(784), 은닉(`N_hidden`), 출력(10) 뉴런 수를 합산한 추정치와 최소 100k 이벤트 중 큰 값을 택해 초기 버퍼를 확보, 이벤트 수집 초반의 재할당을 제거한다.

### scenarios/grad_mimicry.py
- `_ensure_metrics_file(path: str, header: str) -> None`
  - **역할**: gradient mimicry 시나리오용 메트릭 헤더 작성.
- `_evaluate(network: GradMimicryNetwork, loader, device, args) -> Tuple[float, float]`
  - **역할**: 발화율 기반 예측 정확도와 마진 기반 보상 근사치를 평가.
- `run_grad(args, logger)`
    - **역할**: Poisson 인코딩된 MNIST를 입력으로 에이전트 네트워크를 시뮬레이션하고, 입력→은닉/은닉→출력 이벤트별 로컬 상태(히스토리, 현재 가중치, 정규화된 레이어 인덱스, 이벤트 타입)를 통해 Δd를 산출해 각 시냅스에 개별적으로 적용한다. 에피소드 동안 누적된 에이전트 업데이트 `Δw_agent`와 Teacher 역할을 하는 함수형 호출의 per-sample gradient `Δw_teacher` 간 제곱 오차 평균을 보상으로 사용하며, 이미지 미니배치 단위로 PPO Actor–Critic을 학습한다.
    - **세부 구현**: 에포크 진입 시 `s_scen = 1.0`을 정의하고, 모든 `_scatter_updates` 호출이 `args.local_lr * s_scen * action`을 사용하도록 해 로컬 학습률 수식 `Δw_i(t) = η_w * s_scen * Δd_i(t)`을 코드에 명시한다. PPO 업데이트는 `ppo_batch_size`와 이벤트 수를 비교해 미니배치 단위로 진행해 `Theory.md` 2.9.3/2.9.4의 배치 요구를 충족하며, 이벤트 수집 직후 `events_per_image`로 이미지당 이벤트를 리저버 샘플링해 VRAM 사용을 억제한다. 이후 레이어별 델타를 가중치에 적용하고 `w_clip_min/max` 범위로 클리핑한 뒤 Teacher 대비 정렬 보상을 계산한다. 이벤트 수집은 `gather_events`의 블록 기반 희소 윈도우 선택과 레이어 정규화 스칼라(`l_norm`)를 extras에 포함하는 방식으로 통일하며, 동일 스파이크열을 pre/post로 재사용할 때는 패딩 캐시를 공유해 중복 `F.pad`를 줄인다. 이벤트 버퍼는 에포크 외부에서 한 번만 생성하고 배치마다 `reset()`하여 GPU 재할당을 없앤다.
    - **최적화**: Teacher per-sample gradient는 별도 네트워크 복사 없이 `torch.func.functional_call`에 `network`의 파라미터를 `detach().requires_grad_(True)`로 묶어 전달해 수행한다. `vmap(grad(loss_fn))` 호출은 기본적으로 64 샘플 이하로 청크 분할해 OOM을 방지하며, `--grad-chunk-size`가 양수일 때 해당 값으로 덮어쓴다. 네트워크 forward는 사전 할당된 출력 버퍼를 사용해 동기화를 줄이고, 보상·정렬 통계 및 델타 히스토리는 GPU에 유지한 채 에포크 종료 후에만 CPU로 이동해 로그/시각화를 수행한다. per-batch `.item()`/`.cpu()` 호출 대신 GPU 누적 스칼라를 사용해 동기화를 제거했고, `mask = agent_deltas[li].ne(0)` 로직을 유지해 스파이크가 발생한 시냅스만 보상/업데이트에 반영하고, 활성 시냅스 비율을 `active_ratio`로 로깅해 스파이크 소실을 모니터링한다.
    - **dtype 일관성**: Agent 델타와 정렬 보상 누적 버퍼를 가중치 dtype으로 초기화해 float16/float32 혼합 환경에서도 추가 캐스팅 없이 GPU에서 곧바로 합산된다.
    - **버퍼 용량**: 입력(784)과 모든 은닉/출력 레이어 폭을 합산한 추정치와 최소 100k 이벤트 중 큰 값을 초기 용량으로 사용해 다층 네트워크에서도 재할당을 피한다.
    - **Theory 연계**: 시나리오 3의 gradient mimicry 학습 루프.

## main.py
- `main()`
  - **역할**: CLI 파싱, 시드 설정, 결과 디렉터리 생성 및 로거 초기화 후 시나리오별 러너를 호출.
  - **Theory 연계**: 네 시나리오를 선택적으로 실행하는 진입점.
