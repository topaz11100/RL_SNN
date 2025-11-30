# Implementation Summary

이 문서는 `Theory.md`와 `implementation.md`를 바탕으로 구축한 코드 구조와 함수 동작을 설명한다. 모든 함수에 독스트링을 추가해 설계 의도를 코드에 직접 남겼다.

## 폴더 구조
- `main.py`: CLI 진입점으로 시나리오 선택 및 데모 실행을 담당한다.
- `src/models/`: 신경망 구성요소
  - `cnn_frontend.py`: pre/post 스파이크 히스토리를 임베딩하는 1D CNN(`SpikeHistoryCNN`).
  - `actor_critic.py`: Gaussian 정책(`ActorNetwork`), Critic(`CriticNetwork`), 상태 융합(`fuse_state`), 손실 계산(`actor_critic_step`).
  - `lif_neuron.py`: LIF 파라미터/상태 데이터클래스와 한 스텝 시뮬레이션(`LIFNeuron`).
- `src/utils/`: 유틸리티
  - `spike_buffer.py`: 고정 길이 스파이크 히스토리 버퍼(`RollingSpikeBuffer`).
  - `poisson_encoding.py`: MNIST 픽셀을 포아송 스파이크로 변환(`poisson_encode`), Δt 계산(`spike_delta_times`).
  - `reward.py`: 시나리오별 보상 구성 요소와 위너 추적기(`WinnerTracker`).
- `src/scenarios/`: 실험 시나리오
  - `base.py`: 공통 Actor–Critic 관리와 트래젝터리 최적화(`RLScenario`).
  - `unsupervised.py`: 비지도 단일/이중 정책 시나리오(`UnsupervisedSinglePolicy`, `UnsupervisedDualPolicy`).
  - `semi_supervised.py`: 준지도 단일 정책(`SemiSupervisedScenario`).
  - `supervised.py`: 완전지도 Gradient Mimicry(`GradientMimicryScenario`).

## 주요 구현 내용
- **CNN 전단**: Theory 2.4에 맞춰 두 개의 Conv1d-ReLU 블록 후 시간축 평균 풀링으로 16차원 특징을 생성한다.
- **Actor/Critic 헤드**: Theory 2.6/2.7의 32-32-1 MLP 구조와 Tanh 출력(Actor)을 사용하며, 고정 표준편차 Gaussian 정책을 샘플링한다.
- **로컬 상태 융합**: CNN 특징, 현재 가중치, 이벤트 타입 원핫, (필요 시) 레이어 위치를 concat하여 시나리오별 입력 차원을 만족시킨다.
- **LIF 뉴런**: 막전위 적분, 임계값 비교, soft/hard reset 옵션을 포함한 단일 스텝 업데이트를 제공한다.
- **보상 모듈**: 희소성/다양성/안정성, 분류 정확도·마진, 그래디언트 모사 보상을 함수로 분리해 Theory의 정의를 코드로 반영했다.
- **시나리오 클래스**: 각 실험이 요구하는 보상 계산과 정책 구성을 담당하며, 공통 `optimize_from_trajectory`로 Actor–Critic 업데이트를 수행한다.
- **CLI 데모**: `main.py`에서 MNIST 샘플을 포아송 인코딩 후 선택된 시나리오에 맞춰 한 에피소드 데모를 실행한다.

## 함수 동작 설명
- `SpikeHistoryCNN.forward`: (배치, 2, L) 스파이크 히스토리를 16차원 임베딩으로 변환.
- `ActorNetwork.sample_action`: 정책 평균을 Tanh로 제한한 뒤 주어진 `sigma`를 사용해 Gaussian에서 액션을 샘플링하고 로그 확률을 반환.
- `CriticNetwork.forward`: 로컬 상태의 예상 리턴 값을 예측.
- `fuse_state`: CNN 특징과 메타데이터(가중치, 이벤트 타입, 선택적 레이어 위치)를 연결해 시나리오 입력 벡터를 구성.
- `actor_critic_step`: 보상을 Advantage로 변환해 Actor/Critic 손실을 계산.
- `LIFNeuron.step`: LIF 방정식을 적분하고 스파이크 여부에 따라 voltage를 리셋.
- `RollingSpikeBuffer.push/to_tensor/batch`: 최근 L 스텝 pre/post 스파이크를 유지하고 텐서로 변환하거나 배치 차원으로 확장.
- `poisson_encode`: 픽셀값을 발화율로 보고 T 스텝 동안 베르누이 샘플링해 스파이크 행렬 생성.
- `reward_*` 함수들: Theory의 보상 항들을 그대로 코드화.
- 시나리오별 `run_episode`: 주어진 스파이크/가중치 데이터를 순회하며 상태를 만들고 트래젝터리를 수집한 뒤 시나리오 보상을 계산해 최적화 수행.
- `main.run_demo`: CLI 인자에 따라 시나리오를 선택하고 단일 MNIST 샘플에 대해 데모 보상을 출력.

## 결과 및 산출물
- 각 모듈은 Theory.md의 구조(2장 공통 구성, 3~6장 시나리오 요구)를 반영한 클래스로 구현되었다.
- 실행 시 `python main.py --scenario 1.1` 등의 명령으로 선택한 시나리오의 데모 보상 로그를 확인할 수 있다.
- 프로젝트 폴더는 `src/`에 핵심 코드, `docs/`에 설계·설명 문서, `paper/`에 기존 자료가 위치하는 형태를 유지한다.
