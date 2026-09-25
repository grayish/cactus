# 01. 전체 구조: 레이어, 디렉터리, 데이터 흐름

## 1. 한 장짜리 그림

Cactus는 "온디바이스 추론 엔진"이지만 실제로는 **오프라인 컴파일러(Python) + 런타임(C++) + 바인딩**의 3부작이다.

```
        ┌─────────────── 오프라인 (개발자 PC / CI) ───────────────┐
        │  HuggingFace 체크포인트                                  │
        │      │  python/cactus/convert  (CQ 양자화, 토크나이저 export)│
        │      ▼                                                   │
        │  *.weights (CACT) + config.txt + vocab.txt + ...         │
        │      │  python/cactus/transpiler (torch.export → IR → 융합 → 그래프) │
        │      ▼                                                   │
        │  components/<name>/*.cactus (CGRF) + components/manifest.json     │
        └──────────────────────────┬───────────────────────────────┘
                                   │  번들 디렉터리 (weights/<model>-cq<bits>/)
        ┌──────────────────────────▼───────────────────────────────┐
        │  런타임 (폰 / Mac / 라즈베리파이)                          │
        │  cactus-engine ── cactus-graph ── cactus-kernels          │
        │       ▲                                                   │
        │  bindings (Swift/Kotlin/Flutter/RN/Rust) · python ctypes   │
        └──────────────────────────────────────────────────────────┘
```

README의 4층 그림(Engine / Graph / Kernels / Quants)에서 "Quants"는 별도 라이브러리가 아니라
**Python 변환기(`python/cactus/convert/quantization/cq.py`)와 커널(`cactus-kernels/src/matmul.cpp`)에 나뉘어 구현된 포맷**이라는 점에 유의한다.

## 2. 디렉터리 지도

| 경로 | 역할 | 규모/특징 |
|---|---|---|
| `cactus-kernels/` | ARM NEON/Metal 커널. `cactus_kernels.h`가 유일한 공개 헤더 | `matmul.cpp` 3031줄, `cactus_kernels.metal` 3946줄, `metal_backend.mm` 2519줄 |
| `cactus-graph/` | 계산 그래프. `cactus_graph.h`(1307줄) + `src/{core,builder,execute,io,ops_*,metal_plan,metal_runtime,graph_ffi}.cpp` | 정적 라이브러리 `cactus_graph`, 커널을 `add_subdirectory`로 포함 |
| `cactus-engine/` | C API 런타임. `cactus_engine.h` + `src/{init,complete,model,transcribe,stream,embed,rag,index,cloud,telemetry*,tokenizer,bpe,sp,constraints,kv_compress,npu*}.cpp` | `model.cpp` 5902줄, `engine.h` 1475줄 |
| `cactus-engine/tests/` | 엔진 테스트 + `run.cpp`(대화형 REPL, `cactus run`의 실체) + `transcribe.cpp` + iOS/Android 온디바이스 러너 | |
| `python/cactus/` | CLI(`cli/`), ctypes 바인딩(`bindings/cactus.py`, 3343줄), 변환기(`convert/`), 신·구 트랜스파일러(`transpiler/`, `transpile/`), OpenAI 호환 서버(`server.py`) | PyPI 패키지 `cactus-compute` |
| `bindings/` | Swift(모듈맵), Kotlin(JNI/KMP), Flutter(Dart FFI), React Native(브리지), Rust(`extern "C"`) | 모두 헤더/소스 복사 방식, 패키지 저장소 배포 없음 |
| `apple/`, `android/` | xcframework / `.so` 빌드 스크립트와 CMake | Apple: `libcactus_engine-{device,simulator}.a`, `cactus-{ios,macos}.xcframework`. Android: `libcactus_engine.so` (arm64-v8a) |
| `cactus-code/` | `cactus code` 명령이 실행하는 터미널 코딩 에이전트 (pi 포크, Node) | 로컬 서버의 OpenAI API를 호출 |
| `docs/`, `blog/`, `mkdocs.yml` | 공식 문서 | `docs.cactuscompute.com` |
| `.github/workflows/` | CI: 빌드, 커널/그래프 테스트, Python 테스트, 온디바이스 추론 테스트, PyPI/Homebrew 릴리스 | |

## 3. 레이어별 책임 경계

### 3.1 cactus-kernels: "숫자를 빨리 계산한다"
- 입력은 raw 포인터와 크기, 출력도 raw 포인터. 메모리를 소유하지 않는다.
- 스레드 풀(`src/threading.h`)이 여기에 있고, 상위 레이어는 커널 내부 병렬화만 이용한다(그래프는 노드를 순차 실행).
- Apple에서는 큰 행렬에 Accelerate(cblas), 그리고 Metal 셰이더 및 인코더도 이 레이어에 있다.
- 자세한 내용: 06장, 11장.

### 3.2 cactus-graph: "누가 언제 어떤 커널을 호출할지 정한다"
- `CactusGraph`는 `GraphNode{id, op_type, input_ids, output_buffer, params}` 벡터다. **노드가 곧 텐서**다(`cactus_graph.h:444-451`).
- 빌더 메서드(`graph.matmul(a,b)`)는 형상만 계산하고 메모리는 잡지 않는다. `execute()`가 삽입 순서대로 노드를 돌면서 버킷 풀에서 버퍼를 받아 커널을 호출하고, 마지막 소비자가 지나가면 반환한다.
- 가중치는 `mmap_weights()`로 파일을 그대로 매핑해 노드의 `external_data`로 쓴다(zero-copy).
- KV 캐시(`ops_cache.cpp`), MoE, DeltaNet, AltUp, 샘플링, STFT, 이미지 전처리가 모두 op이다.
- Metal 모드에서는 `metal_plan.cpp`가 패턴 융합을 하고 노드 단위로 GPU 인코딩을 시도하며 실패하면 CPU로 떨어진다.
- 그래프는 `graph.cactus`(CGRF) 바이너리로 직렬화/역직렬화된다.
- 자세한 내용: 07장.

### 3.3 cactus-engine: "모델을 제품 기능으로 만든다"
- **모델 클래스 계층이 없다.** 단일 `cactus::engine::Model`(`src/engine.h:674-1190`)이 `components/manifest.json`을 읽어 컴포넌트 그래프들을 로드하고, 어떤 컴포넌트가 있는지에 따라 **디코드 경로(route)** 를 고른다(`model.cpp:737-805`).
- 토크나이저(BPE/SentencePiece), 하드코딩된 패밀리별 채팅 템플릿, 프리필/디코드 루프, 샘플링, 정지 시퀀스, 툴콜 파싱과 제약 디코딩, thinking 분리, confidence 계산과 클라우드 핸드오프, 자동 RAG, 벡터 인덱스, KV 토큰 축출(KeyDiff), Whisper/Parakeet ASR과 스트리밍 전사, 텔레메트리가 여기에 있다.
- 자세한 내용: 08장.

### 3.4 Python: "모델을 번들로 굽고, 개발자 UX를 제공한다"
- `cactus convert` = `convert/`(HF → CQ `.weights`) + `transpiler/`(torch.export → IR → 융합 → `graph.cactus`).
- `cactus run`은 번들을 확보(로컬 → 캐시 → HF 다운로드 → 로컬 빌드)한 뒤 네이티브 `bin/run` 바이너리를 띄운다.
- `cactus serve`는 FastAPI로 `/v1/chat/completions`, `/v1/audio/transcriptions`, `/v1/embeddings`를 제공한다.
- 자세한 내용: 10장, 02장.

## 4. 번들 디렉터리: 레이어를 잇는 계약

런타임이 읽는 파일만 추리면 다음과 같다(`model.cpp:676, 919-1163`, `tokenizer.cpp`).

```
weights/gemma-4-e2b-it-cq4/
  config.txt                     # key=value (model_type, hidden_size, kv_compress_* ...)
  vocab.txt, merges.txt          # 토크나이저
  tokenizer_config.txt, special_tokens.json, tokenizer.json(added_tokens용), chat_template.jinja2(존재/ChatML 여부만 확인)
  *.weights                      # 텐서 1개 = 파일 1개, 84바이트 CACT 헤더 + 스케일 블롭 + 패킹 인덱스
  handoff_probe.bin              # (선택) 하이브리드 confidence 프로브
  components/
    manifest.json                # 컴포넌트 목록, 입출력 노드 id, 가중치 바인딩(node_id → path), 캐시 상태 노드 id, 프롬프트/미디어 스타일
    decoder_step/graph.cactus    # 직렬화 그래프(CGRF)
    decoder_prefill_chunk/graph.cactus
    lm_encoder_step/..., vision_encoder/..., audio_encoder*/...
  runtime_plan.json              # 새 트랜스파일러가 남기는 실행 계획 (manifest 생성 원본)
  transpiler_ir/*.json           # 디버그용 IR (런타임은 읽지 않음)
```

핵심 계약 두 가지:
1. **가중치 바인딩은 노드 id 기준**이다. 그래프 파일에는 가중치가 없고 INPUT 노드만 있으며, manifest의 `bound_constant_bindings[{node_id, path}]`로 로드 시 `bind_mmap_weights()`가 연결한다(`model.cpp:1123-1163`).
2. **컴포넌트 이름이 곧 실행 경로다.** `decoder_step`이 있으면 `DIRECT_DECODER_STEP`, `lm_encoder_step + decoder_step`이면 `CACHED_STEP`, Whisper/Needle처럼 `runtime_route=encoder_cross_kv_decoder_step` 메타데이터가 있으면 `ENCODER_CROSS_KV_STEP` 등이다.

## 5. 데이터 흐름: 요청 하나의 일생

`cactus_complete()` 호출부터 첫 토큰이 콜백으로 나가기까지(상세는 08장 §11):

1. options/messages JSON 파싱 → (코퍼스가 있으면) 자동 RAG로 시스템 메시지 보강 → 툴 파싱 및 Tool-RAG로 상위 k개만 남김
2. confidence 임계값 결정(프로브 있음 0.5 / Gemma 4 0.81 / 기본 0.7) → 패밀리별 채팅 템플릿 렌더 → 토큰화
3. 임계값 ≥ 1.0이면 바로 클라우드
4. 프리필: `decoder_prefill_chunk` 그래프를 청크 단위로 `execute()`, KV 상태를 `decoder_step`으로 이동, 첫 토큰은 argmax
5. 디코드 스텝: `lm_encoder_step`(임베딩/PLE) → 출력 복사 → `decoder_step.execute()` → 샘플링. 4096 토큰 이상이면 KV 축출
6. 첫 토큰의 마진 기반 confidence가 임계값 미만이면 스트리밍 전에 클라우드 시도, 실패 시 로컬 계속
7. `callback(decode(token), id, user_data)`로 토큰 전달, 정지 시퀀스까지 반복 → thinking/툴콜 분리 → 응답 JSON → 텔레메트리

각 단계에서 그래프는 `execute()` 한 번, 그래프 안의 각 노드는 커널 한 번 이상을 호출한다. 이 3단 호출 구조를 머리에 두면 어떤 파일을 열어야 할지 바로 보인다:

| 궁금한 것 | 열 파일 |
|---|---|
| 프롬프트가 어떻게 토큰이 되나 | `cactus-engine/src/tokenizer.cpp`, `bpe.cpp`, `sp.cpp` |
| 프리필/디코드가 어떤 그래프를 몇 번 부르나 | `cactus-engine/src/model.cpp` (`run_chunked_prefill`, `run_step`, `decode`) |
| 그래프 노드 하나가 어떻게 실행되나 | `cactus-graph/src/execute.cpp` (`execute`, `dispatch_node`) |
| matmul이 실제로 어떤 명령어를 쓰나 | `cactus-kernels/src/matmul.cpp` |
| GPU로 가는 기준은 | `cactus-graph/src/execute.cpp` (`try_encode_metal`), `metal_plan.cpp` |
| 가중치 파일 헤더 | `cactus-graph/src/io.cpp` (`MappedFile::parse_header`), `python/cactus/convert/quantization/cq.py` (`write_cq_tensor`) |

## 6. 설계 원칙 (코드에서 읽히는 것)

1. **디코드는 대역폭, 프리필은 연산.** `GemmThreading`이 M=1이면 스레드를 1~5개로 제한하고 M>1이면 풀 전체를 쓴다(`threading.h:593-624`).
2. **파일을 메모리에 복사하지 않는다.** `mmap(PROT_READ, MAP_SHARED)` + `MADV_SEQUENTIAL`, Metal은 `newBufferWithBytesNoCopy`로 같은 페이지를 GPU에 노출한다.
3. **모델별 C++ 코드를 늘리지 않는다.** 새 모델은 Python 프로필/어댑터로 추가하고, 런타임은 manifest 계약만 지킨다.
4. **실패는 CPU로 떨어진다.** Metal 인코더가 `false`를 반환하면 그 노드는 CPU 커널로 실행되고, 융합 클러스터가 실패하면 해당 앵커를 금지하고 재실행한다(`execute.cpp:1843-1880`).
5. **의존성을 최소화한다.** libcurl(+Android mbedtls), picojson, stb_image 외에는 표준 라이브러리와 플랫폼 프레임워크뿐이다.

다음 장(02)에서는 이 구조가 각 플랫폼에서 어떻게 빌드되고 배포되는지 본다.
