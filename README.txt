# mistral-interview


Mistral AI나 글로벌 최첨단 AI 기업의 System Design 면접 및 실무에서 핵심이 되는 주요 주제들을 하나씩 차근차근, 깊이 있게 정리해 드리겠습니다.
첫 번째 주제는 가장 기본이 되면서도 면접관들이 가장 많이 묻는 1. GPU 메모리 계산 및 병렬 처리(Parallelism) 전략입니다.
주제 1: GPU 메모리 계산 및 병렬 처리 (Parallelism)
LLM 배포 시스템을 설계할 때 첫 번째 관문은 "우리가 가진/필요한 GPU 메모리가 얼마인가?"와 "단일 GPU에 안 올라가는 모델을 어떻게 쪼개서 올릴 것인가?"를 명확한 수식과 구조로 계산해 내는 것입니다.
1. LLM 메모리 구조 계산법 (Precision & GPU RAM)
GPU 메모리는 크게 ① 모델 가중치(Weights), ② KV 캐시(KV Cache), ③ 오버헤드/Activation 세 가지로 구성됩니다.
① 모델 가중치 (Model Weights)
기본 공식: 파라미터 수(B) × 정밀도(Bytes)
정밀도별 바이트 수:
FP16 / BF16 (16-bit): 파라미터당 $2\text{ bytes}$
INT8 (8-bit Quantization): 파라미터당 $1\text{ byte}$
INT4 / AWQ / GPTQ (4-bit Quantization): 파라미터당 $0.5\text{ bytes}$
예시 (Mistral / Llama 70B 모델 기준):
FP16: $70\text{B} \times 2\text{ bytes} = \mathbf{140\text{ GB}}$
INT8: $70\text{B} \times 1\text{ byte} = \mathbf{70\text{ GB}}$
INT4 (AWQ): $70\text{B} \times 0.5\text{ bytes} = \mathbf{35\text{ GB}}$
② KV 캐시 (KV Cache Memory)
LLM이 토큰을 하나씩 생성할 때 이전 토큰들의 Key/Value 값을 재계산하지 않고 메모리에 저장해 두는 공간입니다. 동시 접속자(Batch Size)와 문맥 길이(Context Length)가 늘어날수록 급격히 증가합니다.
KV Cache 공식 (Transformer 기준):

$$\text{KV Cache Size} = 2 \times (\text{Layers}) \times (\text{Hidden Size}) \times (\text{Sequence Length}) \times (\text{Batch Size}) \times (\text{Bytes per Precision})$$
실무적 관점:
예를 들어 A100 $80\text{GB}$ GPU에서 FP16 $70\text{B}$ 모델을 띄우려면 가중치만 $140\text{GB}$이므로 GPU 1대로는 모델조차 못 올립니다.
GPU 2대($160\text{GB}$)로 나누어 올리면 $140\text{GB}$를 가중치가 차지하고, 남는 $20\text{GB}$로 KV Cache를 감당해야 하므로 동시 처리 수(Batch Size)에 극심한 한계가 생깁니다.
따라서 GPU 4대($320\text{GB}$) 또는 8대($640\text{GB}$)로 쪼개어 가중치 비중을 줄이고 KV Cache용 메모리를 여유 있게 확보하는 설계가 필요합니다.
2. 병렬 처리 전략 (Parallelism Types)
하나의 GPU에 담을 수 없는 모델을 여러 GPU로 분산시키는 기법입니다.
       [ Input Tokens ]
              │
   ┌──────────┴──────────┐
   ▼                     ▼
[ GPU 0 ]             [ GPU 1 ]   <-- Tensor Parallelism (Layer 내부 쪼개기: NVLink 필수)
(Weight/8)           (Weight/8)
   │                     │
   └──────────┬──────────┘
              ▼
          [ Layer 1 ]
              │
              ▼
          [ Layer 2 ]             <-- Pipeline Parallelism (Layer 단위 쪼개기: PCIe/Network)
           (GPU 2/3)

1) Tensor Parallelism (TP - 텐서 병렬화)
개념: 하나의 Transformer Layer 안의 Matrix Multiplication(MatMul) 연산 자체를 여러 GPU에 분할하여 계산합니다.
특징:
매 Layer마다 All-Reduce 통신이 일어납니다.
통신 병목이 매우 크므로, 반드시 NVLink로 연결된 단일 노드(Single-Node, 예: 8×A100/H100) 내에서 주로 사용합니다.
적용 예시: 70B 모델을 단일 노드의 GPU 8대에 나누어 실시간 추론 레이턴시(Latency)를 최소화할 때 (TP=8).
2) Pipeline Parallelism (PP - 파이프라인 병렬화)
개념: 모델의 Layer 전체(예: 총 80개 레이어)를 순차적으로 쪼개어 각 GPU에 배치합니다. (GPU 0: 120 레이어, GPU 1: 2140 레이어 ...)
특징:
GPU 간 통신량이 TP에 비해 적어 서로 다른 노드(Multi-Node) 간 대규모 분산 서빙/학습 시 적합합니다.
앞 단계 GPU 계산이 끝날 때까지 뒷 단계 GPU가 기다리는 버블(Bubble) 현상이 발생하므로 Micro-batching 전략이 필요합니다.
3) Sequence Parallelism & Context Parallelism
개념: 32k, 128k 등 초장문(Long Context) 입력 처리 시, 문맥 자체를 분할하여 여러 GPU에 나누어 계산합니다.
3. MoE (Mixture of Experts) 모델의 특수성
Mistral AI의 대표 모델인 Mixtral 8x7B / Mixtral 8x22B는 MoE 아키텍처를 사용합니다.
메모리 계산 주의점:
Mixtral 8x7B의 전체 파라미터 수는 약 $47\text{B}$입니다. (단순 $8 \times 7 = 56\text{B}$가 아님, Router 및 Shared Layer 때문)
따라서 메모리에 올릴 때는 $47\text{B}$ 전체에 해당하는 메모리($FP16\text{ 기준 약 } 95\text{GB}$)가 확보되어야 합니다.
연산 속도(Throughput):
추론 시에는 8개 전문가 중 토큰당 2개의 Expert만 활성화되므로, 실제 계산량은 약 $13\text{B}$ 모델 수준의 빠르기를 보입니다.
즉, 메모리는 $47\text{B}$만큼 먹지만 속도는 $13\text{B}$처럼 빠른 효율성을 제공합니다.
4. 실무/면접에서의 적용 및 대답 포인트
질문 예시: "금융 고객이 Mistral 70B 모델을 기반으로 Latency 100ms 미만의 실시간 채팅을 원합니다. GPU 아키텍처를 어떻게 구성하시겠습니까?"
추천 답변 구조:
메모리 도출: "Mistral 70B FP16 가중치는 $140\text{GB}$입니다. KV Cache와 Peak Activation을 고려하면 최소 $200\text{GB}+$ 이상의 GPU RAM이 필요합니다."
병렬화 선택: "Single Request Latency를 최우선으로 해야 하므로, NVLink 통신 속도가 확보된 단일 서버 내에서 Tensor Parallelism 8 (TP=8)을 적용하여 A100 $80\text{GB} \times 8$대 또는 H100 $\times 8$대에 올리겠습니다."
최적화 옵션 제안: "만약 GPU 예산 절감이 중요하다면, AWQ INT4 양자화를 적용하여 가중치를 $35\text{GB}$로 줄이고, TP=2 또는 TP=4 (A100 $80\text{GB} \times 2$대) 구조로 변경하여 TCO를 60% 이상 낮추는 트레이드오프를 고객에게 제시하겠습니다."
다음 주제로 넘어가서 심도 있게 알아볼까요? 아래 목록 중 원하시는 순서나 주제를 말씀해 주세요.
주제 2: vLLM & TensorRT-LLM 서빙 엔진 최적화 (PagedAttention, Continuous Batching, Chunked Prefill 등)
주제 3: K8s / EKS 기반 Auto-scaling & Cold Start 해결 (KEDA, Spegel, DaemonSet, NVMe Caching)
주제 4: Smart LLM Router & Multi-Agent 시스템 구축 (Context-Aware Routing, LangGraph)
어떤 주제부터 계속 진행할까요?
