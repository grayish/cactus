# Cactus 아키텍처 분석 및 학습 가이드

> 이 폴더는 Cactus(v2.2.1, 커밋 `fa094ce`) 코드베이스를 **처음 배우는 사람부터 커널을 튜닝하려는 사람까지** 단계적으로 안내하기 위해 작성된 한국어 학습 자료다.
> 모든 서술은 이 리포지토리의 소스 코드와 공식 문서를 직접 읽고 확인한 내용에 근거하며, 가능한 한 `파일:줄` 형태로 출처를 남겼다.
> 코드가 바뀌면 줄 번호는 어긋날 수 있으니 함수명·상수명으로 다시 찾아 읽는 것을 권한다.

## 이 자료가 답하려는 질문

| 질문 | 장 |
|---|---|
| Cactus는 전체적으로 어떻게 생겼는가? 디렉터리와 4개 레이어는 어떻게 연결되나? | [01. 전체 구조](01-overview-architecture.md) |
| 어떤 OS/칩/언어를 지원하고, 어떻게 빌드·패키징되나? 지원하지 않는 것은? | [02. 멀티플랫폼 지원](02-multiplatform.md) |
| llama.cpp, ExecuTorch, MLC, ONNX Runtime, LiteRT 와 무엇이 다른가? | [03. 다른 엔진과의 비교](03-comparison-other-engines.md) |
| 설치부터 첫 추론, Python/C API, Graph API까지 손으로 따라 하려면? | [04. 튜토리얼: 시작하기](04-tutorial-getting-started.md) |
| 튜토리얼 이후 복잡한 부분까지 어떤 순서로 학습해야 하나? | [05. 학습 로드맵](05-learning-path.md) |
| CPU 커널(NEON/dotprod/CQ matmul/스레딩)은 어떻게 짜여 있나? | [06. Kernels 심화](06-kernels-deep-dive.md) |
| 계산 그래프, 메모리 풀, mmap 가중치 포맷, KV 캐시는? | [07. Graph 심화](07-graph-deep-dive.md) |
| 엔진(C API, 생성 루프, 토크나이저, RAG, 하이브리드 클라우드)은? | [08. Engine 심화](08-engine-deep-dive.md) |
| CQ 양자화와 TurboQuant-H는 정확히 무엇을 하나? | [09. 양자화](09-quantization.md) |
| HuggingFace 모델은 어떻게 Cactus 번들이 되나? | [10. 변환기/트랜스파일러](10-transpiler.md) |
| 가속기 백엔드(CPU/GPU/NPU)의 종류, 선택 로직, 튜닝 노브, 그리고 XPU 등 없는 것은? | [11. 가속기 백엔드와 튜닝](11-accelerator-backends.md) |

## 30초 요약

```
 ┌──────────────────────────────┐   Swift / Kotlin / Flutter / RN / Rust / Python
 │   bindings/, python/         │   (모두 얇은 FFI, 네이티브 로직 없음)
 └──────────────┬───────────────┘
                │ C ABI (cactus_engine.h: cactus_init / cactus_complete / ...)
 ┌──────────────▼───────────────┐
 │   cactus-engine  (C++20)     │   manifest.json 을 읽어 컴포넌트 그래프를 실행하는 런타임
 │   토크나이저, 채팅템플릿,     │   프리필/디코드 루프, 샘플링, 툴콜 파싱, RAG, KV 압축,
 │   하이브리드 클라우드 핸드오프 │   Whisper/Parakeet ASR, 텔레메트리
 └──────────────┬───────────────┘
                │ CactusGraph::load / execute
 ┌──────────────▼───────────────┐
 │   cactus-graph               │   ~125종 op, 노드=텐서, 버킷 풀 메모리, mmap 가중치(CACT),
 │                              │   직렬화 그래프(CGRF), INT8 KV 캐시 op, Metal 융합 플래너
 └──────────────┬───────────────┘
                │ cactus_* 커널 호출 / Metal 인코더
 ┌──────────────▼───────────────┐
 │   cactus-kernels             │   ARM NEON FP16 + SDOT, CQ 코드북 matmul, flash 어텐션,
 │                              │   스레드 풀, Metal 셰이더(120 커널), Apple Accelerate 위임
 └──────────────────────────────┘
        ▲
        │ 오프라인: python/cactus/convert (CQ 양자화) + python/cactus/transpiler (torch.export → IR → 그래프)
```

- **가중치 포맷과 커널을 함께 설계**(CQ: Hadamard 회전 + 코드북, 1~4bit)한 것이 최대 차별점이다.
- **모델 아키텍처는 C++에 없다.** Gemma 4, Qwen, LFM2 등의 forward는 Python 트랜스파일러가 그래프로 굽고, 엔진은 그것을 실행만 한다.
- **가속기는 CPU(ARM64 전용), Apple Metal, 그리고 (현재 비활성 상태의) Apple ANE** 세 가지다. Android GPU/NPU, Vulkan, CUDA, x86는 없다.

## 권장 읽기 순서

- **처음이라면**: 01 → 04 → 05 → 03
- **앱에 붙이려는 개발자**: 02 → 04 → 08(§1 C API) → 11(§선택 로직)
- **성능/커널 엔지니어**: 06 → 11 → 07 → 09
- **모델 변환/양자화 연구자**: 09 → 10 → 07(§3 포맷) → 06(§2 CQ 커널)

## 함께 읽을 공식 문서

- `README.md`, `llms.txt` — 프로젝트 개요와 CLI 목록
- `docs/cactus_engine.md`, `docs/cactus_graph.md`, `docs/cactus_kernels.md`, `docs/cactus_index.md` — API 레퍼런스
- `docs/cactus_quants.md`, `blog/turboquant-h.md` — 양자화
- `docs/cactus_transpiler.md`, `python/cactus/transpiler/CONTRIBUTING.md` — 변환 파이프라인 (전자는 구 파이프라인 기준, 10장 참고)
- `docs/cactus_hybrid.md` — 하이브리드 추론
- `blog/lfm2.5_350m.md` — "왜 CPU인가"라는 설계 철학
