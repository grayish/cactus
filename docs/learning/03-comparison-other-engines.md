# 03. 다른 추론 엔진과 무엇이 다른가

> 이 장은 Cactus를 llama.cpp, ExecuTorch, MLC-LLM, ONNX Runtime, LiteRT(TFLite), MNN, MLX 와 나란히 놓고
> "설계 철학·타깃·백엔드·양자화·모델 변환 경로" 관점에서 비교한다.
> 다른 엔진에 대한 설명은 2026년 중반 시점의 공개 정보를 바탕으로 하며, 각 프로젝트는 빠르게 바뀌므로
> 세부 사항은 해당 프로젝트 문서로 재확인하는 것이 좋다. Cactus 쪽 사실은 이 리포지토리 코드와 문서에 근거한다.

---

## 1. 한 문장 요약

| 엔진 | 한 문장 정체성 |
|---|---|
| **Cactus** | "폰·웨어러블용 하이브리드 엣지-클라우드 AI 엔진". ARM CPU를 1급 타깃으로 삼고, 자체 양자화(CQ)·자체 그래프·자체 커널을 수직 통합. 텍스트+비전+오디오+전사+RAG+클라우드 핸드오프를 하나의 C API로 제공 |
| llama.cpp / ggml | 범용 C/C++ LLM 런타임. GGUF 포맷, 가장 넓은 하드웨어 백엔드(CPU, Metal, CUDA, Vulkan, SYCL, HIP, OpenCL …)와 가장 큰 커뮤니티 |
| ExecuTorch | PyTorch 공식 온디바이스 런타임. `torch.export` → `.pte`, 하드웨어별 *delegate*(XNNPACK, CoreML, MPS, QNN, MediaTek, Vulkan, Ethos-U …) |
| MLC-LLM | TVM 기반 컴파일러 접근. 모델을 사전에 컴파일해 Vulkan/Metal/OpenCL/WebGPU/CUDA 커널을 생성 |
| ONNX Runtime | ONNX 그래프 + 실행 프로바이더(EP: CPU, NNAPI, CoreML, QNN, XNNPACK, DirectML, CUDA …). 범용 ML 런타임에 GenAI 확장 |
| LiteRT (구 TFLite) / MediaPipe | Google의 모바일 런타임. GPU delegate, 벤더 delegate, LiteRT-LM 및 MediaPipe LLM Inference API |
| MNN (Alibaba) | 경량 모바일 추론 프레임워크. CPU/OpenCL/Vulkan/Metal, MNN-LLM 확장 |
| MLX | Apple 실리콘 전용 배열 프레임워크. 통합 메모리 활용, 연구·Mac 로컬 용도에 강함 |

---

## 2. 설계 철학의 차이

### 2.1 Cactus: "CPU가 보편 타깃이다" + 수직 통합

Cactus가 다른 엔진과 가장 크게 갈라지는 지점은 **무엇을 1급 타깃으로 보는가**이다.

- `blog/lfm2.5_350m.md`의 "Why CPU, Not GPU or Dedicated Accelerators" 절이 철학을 직접 서술한다.
  디코드는 메모리 대역폭 바운드라 GPU의 FLOPS 우위가 의미 없고, 모바일 GPU는 UI 렌더링과 경쟁하며 배터리를 소모하고,
  ANE/Hexagon 같은 NPU는 벤더 포맷과 제한된 양자화 스킴에 종속된다는 논리다.
- 그래서 커널 레이어는 **ARM NEON + DOTPROD + I8MM** 위에 손으로 튜닝되고 (`cactus-kernels/src/matmul.cpp`),
  GPU(Metal)와 NPU(ANE)는 "특정 상황에서 얹는 가속"으로 위치한다 (11장 참고).
- 양자화(CQ), 파일 포맷(`CACT` 매직의 mmap 컨테이너), 그래프, 커널, 엔진, 토크나이저, 변환기까지 **모두 자체 구현**이다.
  외부 의존성은 사실상 libcurl(클라우드 핸드오프), picojson, stb_image 정도다.

이 선택의 장점은 (1) 모든 ARM 기기에서 동일한 코드 경로, (2) 양자화 포맷과 커널을 함께 설계할 수 있어 CQ 같은
공격적인 저비트 기법을 런타임에 곧바로 반영 가능, (3) 백그라운드 실행에 유리한 단일 코어 디코드 정책이다.
단점은 (1) NVIDIA GPU·x86 서버 같은 비모바일 환경은 관심 밖, (2) 커뮤니티 규모와 지원 모델 수가 llama.cpp에 비해 작다는 점이다.

### 2.2 llama.cpp: "모든 하드웨어, 모든 모델"

- ggml은 백엔드 추상화 위에 수십 개의 하드웨어 백엔드를 갖고 있고, GGUF는 사실상 표준 포맷이 되었다.
- 양자화는 K-quants, i-quants(imatrix) 등 다양하지만 기본적으로 **선형 레이어 중심**이다.
  Cactus 문서(`docs/cactus_quants.md`)는 바로 이 지점을 비판한다. Gemma 4 E2B처럼 per-layer 임베딩이 모델의 60%를 차지하는
  구조에서는 임베딩·모달리티 타워까지 같은 레시피로 양자화해야 하며, CQ는 그것을 목표로 설계됐다.
- llama.cpp는 멀티모달(비전, 최근에는 오디오)도 지원하지만, "텍스트+비전+오디오+전사(Whisper/Parakeet)+임베딩+벡터 인덱스+RAG+클라우드 핸드오프"를
  **하나의 C 함수군**(`cactus_engine.h`)으로 제공하는 Cactus와는 제품 범위가 다르다.
  (llama.cpp 진영에서는 whisper.cpp 등 별도 프로젝트가 담당한다.)

### 2.3 ExecuTorch: "PyTorch 그래프를 그대로 내보내고, 하드웨어는 delegate가 처리"

- `torch.export`로 ATen 그래프를 잡고, 백엔드별 delegate가 부분 그래프를 가져가는 구조다.
- **Cactus 변환기도 `torch.export`를 사용한다** (`docs/cactus_transpiler.md`의 6단계 파이프라인, `python/cactus/transpile/`,
  새 `python/cactus/transpiler/`). 즉 "PyTorch 모델을 ATen 그래프로 캡처한다"는 앞단은 같다.
- 차이는 뒷단이다. ExecuTorch는 캡처된 그래프를 최대한 그대로 실행하고 하드웨어 최적화를 delegate에 맡기는 반면,
  Cactus는 **패턴 융합(RMSNorm, RoPE, attention, MLP, LSTM 등)을 자체 IR에서 수행한 뒤 자체 그래프 op으로 낮추고**,
  그 op들은 자체 커널에 1:1로 대응한다. 그래서 Cactus는 delegate 없이도 융합된 커널을 쓴다.
- ExecuTorch의 강점은 벤더 NPU delegate(QNN, MediaTek, Ethos-U 등)의 폭이다. Cactus는 현재 Apple ANE만 부분 지원한다(11장).

### 2.4 MLC-LLM: "컴파일러가 커널을 생성"

- TVM으로 모델별·기기별 커널을 생성하고 Vulkan/Metal/OpenCL/WebGPU 등 GPU를 기본 실행 대상으로 삼는다.
- Cactus는 "손으로 쓴 커널 + 손으로 쓴 Metal 셰이더"다. 자동 튜닝(AutoTVM 같은)은 없다.
  대신 상수(타일 크기, 스레드 임계값)를 코드에서 직접 조정한다(11장의 튜닝 노브).

### 2.5 ONNX Runtime / LiteRT: "범용 ML 런타임 + LLM 확장"

- 두 프레임워크는 원래 CNN 시대의 범용 런타임이고, LLM은 확장(ORT GenAI, LiteRT-LM/MediaPipe LLM API)으로 얹혔다.
- 벤더 NPU 접근성이 넓다(NNAPI 는 deprecated 흐름, QNN EP, CoreML EP 등).
- Cactus는 반대로 **LLM/VLM/ASR 워크로드에서 출발한 그래프**라, KV 캐시(`ops_cache.cpp`), 하이브리드 INT8/FP16 어텐션,
  샘플링 op 등이 그래프 1급 시민이다.

### 2.6 MLX: "Apple 실리콘 전용 연구 프레임워크"

- Mac에서 개발자 경험이 매우 좋고 통합 메모리를 잘 활용하지만, iOS/Android 앱 배포용 엔진은 아니다.
- Cactus는 Mac에서도 돌지만 목표는 폰·웨어러블·라즈베리파이까지이고, Swift/Kotlin/Flutter/RN 바인딩이 제품의 일부다.

---

## 3. 항목별 비교표

| 항목 | Cactus | llama.cpp | ExecuTorch | MLC-LLM | ORT / LiteRT |
|---|---|---|---|---|---|
| 1급 타깃 | ARM CPU(NEON/dotprod/i8mm) | CPU + 모든 GPU | 벤더 delegate | GPU (컴파일) | CPU + EP/delegate |
| GPU | Apple Metal (자체 셰이더 `cactus_kernels.metal`) | Metal, CUDA, Vulkan, HIP, SYCL, OpenCL | MPS, Vulkan, CoreML | Vulkan, Metal, OpenCL, WebGPU, CUDA | GPU delegate / CUDA / DirectML |
| NPU | Apple ANE (오디오·비전 인코더 한정, CoreML `.mlpackage`) | 제한적 | QNN, CoreML, MediaTek, Ethos-U 등 | 없음/제한 | QNN, CoreML, NNAPI(레거시) |
| Android GPU/NPU | **없음** (CPU 전용) | Vulkan/OpenCL | Vulkan/QNN/MediaTek | Vulkan/OpenCL | GPU delegate/QNN |
| 모델 포맷 | 자체 `CACT` mmap 컨테이너 + `graph.cactus` 번들 | GGUF | `.pte` | 컴파일된 lib + 파라미터 | ONNX / `.tflite` |
| 양자화 | CQ(회전+코드북) 1/2/3/4bit, 혼합 2.54/3.26, INT8, TurboQuant-H 임베딩 | K/I-quants 등 | 백엔드 의존 | 그룹 양자화 | 백엔드 의존 |
| 임베딩·타워 양자화 | 동일 레시피로 전부 | 부분적 | 백엔드 의존 | 부분적 | 부분적 |
| KV 캐시 | INT8 (하이브리드 INT8/FP16 어텐션 커널) | FP16/양자화 옵션 | 백엔드 의존 | FP16 | 백엔드 의존 |
| 변환 경로 | `cactus convert` (HF → CQ 가중치 → torch.export → IR 융합 → CactusGraph) | `convert_hf_to_gguf.py` | `torch.export` + 파티셔너 | TVM relax | 각종 exporter |
| 멀티모달 | 텍스트·비전·오디오·전사(Whisper/Parakeet)·임베딩 한 엔진 | 텍스트·비전(+오디오 일부) | 모델별 예제 | 텍스트 중심 | 모델별 |
| 하이브리드 클라우드 | **내장** (confidence probe + 핸드오프, `cloud.cpp`) | 없음 | 없음 | 없음 | 없음 |
| RAG / 벡터 인덱스 | 내장 (`index.cpp`, `rag.cpp`) | 없음 | 없음 | 없음 | 없음 |
| 툴 콜링 | 내장 파서 + Tool RAG | 템플릿 수준 | 없음 | 부분 | 없음 |
| 바인딩 | Swift, Kotlin/KMP, Flutter, RN, Python, Rust, C | 매우 많음 | Swift/Kotlin/Python | Swift/Kotlin/JS/Python | 매우 많음 |
| 자동 튜닝 | 없음 (수동 상수) | 없음 | 없음 | 있음(TVM) | 없음 |
| x86 | 스칼라/제한적 (개발·테스트용) | 완전 지원 | 지원 | 지원 | 완전 지원 |

---

## 4. Cactus만의 기술적 차별점 (코드로 확인 가능한 것)

1. **CQ 커널이 활성화에 Hadamard 변환을 적용**한다. 가중치가 아닌 FP16 활성화를 회전시킨 뒤
   "코드북 룩업 + norm 스케일 + matmul"을 한 번의 NEON 패스로 처리한다 (`docs/cactus_quants.md` 마지막 절, `cactus-kernels/src/matmul.cpp`).
   대부분의 엔진은 가중치 디양자화 후 일반 GEMM을 호출하는 구조다.
2. **임베딩 테이블까지 2bit**로 내린다. TurboQuant-H는 per-layer 임베딩(PLI)을 그룹 128 Hadamard + Lloyd-Max 코드북으로 2.125 bit/elem까지 압축해
   Gemma 4 E2B를 4.8GB → 2.9GB로 줄인다 (`blog/turboquant-h.md`).
3. **디코드는 단일 코어, 프리필은 멀티 코어**라는 명시적 정책. 백그라운드 실행 시 OS의 스로틀링/킬을 피하려는 제품 관점의 결정이다 (`blog/lfm2.5_350m.md`).
4. **하이브리드 추론이 엔진 안에 있다.** 65k 파라미터 프로브가 hidden state를 읽어 confidence를 내고,
   임계값 아래면 클라우드로 자동 핸드오프한다 (`docs/cactus_hybrid.md`, `cactus-engine/src/cloud.cpp`).
5. **zero-copy mmap 번들**. 가중치는 `mmap(PROT_READ, MAP_SHARED)`로 매핑되고 OS 페이지 캐시가 워킹셋을 관리한다.
   Apple에서는 355MB 모델이 56~85MB RAM으로 도는 이유이며, Android에서 300MB대로 벌어지는 이유도 같은 절에 설명돼 있다.
6. **그래프가 LLM 친화적**이다. KV 캐시 op, 하이브리드 어텐션, 샘플링, MoE, 재귀/컨볼루션(LFM2), DSP(멜 스펙트로그램), 이미지 전처리가 모두 그래프 op이다
   (`cactus-graph/src/ops_*.cpp`).

---

## 5. 언제 Cactus를 고르고, 언제 다른 것을 고르나

**Cactus가 맞는 경우**
- iOS/Android/웨어러블/라즈베리파이 등 ARM 기기에서 2B 이하 모델을 **배터리 친화적으로**, 백그라운드에서도 돌려야 할 때
- 텍스트·비전·오디오·전사·RAG·툴콜을 한 SDK로 끝내고 싶을 때
- 3bit 이하 양자화가 필요할 때 (CQ는 3bit에서 GPTQ/AWQ/HQQ 대비 최대 +32pt, 2bit에서 유일하게 생존)
- 온디바이스가 못 푸는 질문만 클라우드로 보내는 하이브리드 구조를 원할 때

**다른 엔진이 맞는 경우**
- NVIDIA/AMD GPU 서버, x86 데스크톱이 주 타깃 → llama.cpp, vLLM 등
- Qualcomm Hexagon, MediaTek APU 등 **Android NPU**를 반드시 써야 할 때 → ExecuTorch(QNN), ORT(QNN EP), LiteRT
- 수백 개의 아키텍처를 즉시 GGUF로 받아 실험하고 싶을 때 → llama.cpp
- Mac에서 연구용으로 빠르게 파인튜닝·실험 → MLX

---

## 6. 학습자를 위한 대응표 (개념 이름 매핑)

다른 엔진에 익숙한 사람이 Cactus 코드를 읽을 때 유용한 용어 대응이다.

| 개념 | llama.cpp | ExecuTorch | Cactus |
|---|---|---|---|
| 텐서 연산 그래프 | `ggml_cgraph` | `ExecutionPlan` (in `.pte`) | `CactusGraph` (`cactus-graph/cactus_graph.h`) |
| 백엔드 커널 | `ggml-cpu`, `ggml-metal` … | delegate / portable kernels | `cactus-kernels/src/*.cpp`, `cactus_kernels.metal` |
| 모델 파일 | GGUF | `.pte` | `CACT` weights + `graph.cactus` + `manifest.json` |
| 고수준 추론 API | `llama.h` | C++ `Module` | `cactus_engine.h` (`cactus_init`, `cactus_complete` …) |
| 변환 스크립트 | `convert_hf_to_gguf.py` | `export` 스크립트 + 파티셔너 | `cactus convert` (`python/cactus/cli/convert.py` → `transpile/` / `transpiler/`) |
| 양자화 타입 | `Q4_K_M`, `IQ2_XS` … | 백엔드별 | `CQ4`, `CQ3`, `CQ2`, `CQ1`, `INT8`, 혼합 `2.54`/`3.26` |
| KV 캐시 | `llama_kv_cache` | 모델 코드 안 | `cactus-graph/src/ops_cache.cpp` |
| 채팅 템플릿 | Jinja(minja) | 앱 코드 | `cactus-engine/src/chat_tools.h`, `gemma_tools.h` |

다음 장(04)에서는 실제로 설치하고 첫 추론을 돌리는 튜토리얼로 넘어간다.
