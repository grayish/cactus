# 05. 학습 로드맵: 튜토리얼에서 커널 튜닝까지

> 04장을 마쳤다는 전제로, 6주 정도의 자기주도 커리큘럼을 제안한다. 각 주차는 "읽을 것 → 실습 → 확인 질문" 구조다.
> 순서는 사용자 관점(위)에서 하드웨어 관점(아래)으로 내려가되, 3주차 이후는 관심에 따라 병렬로 진행해도 된다.

```
 1주  사용자 관점        C API · 번들 · CLI · 서버                 (08장 §1, 02장)
 2주  엔진 런타임        생성 루프 · 토크나이저 · 하이브리드 · RAG  (08장)
 3주  그래프             노드/버퍼/실행 · 직렬화 · KV 캐시          (07장)
 4주  변환기             convert(CQ) · transpiler(IR/융합)          (09장, 10장)
 5주  커널               NEON · CQ matmul · 어텐션 · 스레딩         (06장)
 6주  가속기와 튜닝      Metal 플래너 · ANE · 벤치마크 · 노브       (11장)
```

---

## 1주차: 사용자 관점으로 전체를 한 바퀴

**읽기**
- `README.md` 전체, `llms.txt`, `docs/quickstart.md`, `docs/choose-bindings.md`
- `cactus-engine/cactus_engine.h` (전부, 240줄 남짓) — 함수 하나하나에 대해 "어떤 소스 파일이 구현하나"를 08장 §1 표와 대조
- `python/cactus/cli/__init__.py`의 argparse 정의 → 명령이 어떤 핸들러로 가는지(`_COMMANDS`)
- `python/cactus/server.py`의 라우트 3개

**실습**
1. `cactus run`을 `--backend cpu`와 기본값으로 각각 돌려 TTFT/decode tps를 기록한다.
2. `cactus serve`를 띄우고 `curl`로 SSE 스트리밍을 받아 본다.
3. 번들 디렉터리를 열어 `config.txt`, `components/manifest.json`, `runtime_plan.json`을 읽고, manifest의 `bound_constant_bindings`에서 `node_id → path` 하나를 골라 그 `.weights` 파일의 84바이트 헤더를 `xxd`로 본다(07장 §3 표와 대조).

**확인 질문**
- `cactus_init`의 세 번째 인자 `cache_index`는 언제 의미가 있나? (`init.cpp:400-414`)
- 옵션 `temperature: 0`과 `temperature: -1`의 차이는? (`model.cpp:4594-4597`)
- 응답 JSON의 `confidence`는 무엇의 함수인가? (08장 §4)

---

## 2주차: 엔진 런타임

**읽기 순서**
1. `cactus-engine/src/init.cpp` (`cactus_init` 356-440, 자동 RAG 청킹 77-214)
2. `cactus-engine/src/complete.cpp`: `prepare_prompt`(535-671) → `do_prefill`(673-760) → `cactus_complete`(814-1240) 순서로 위에서 아래로
3. `cactus-engine/src/utils.h`: `parse_inference_options_json`(1586-1721), `parse_messages_json`(989-1152), `construct_response_json`(1898-1952)
4. `cactus-engine/src/tokenizer.cpp`: `detect_model_type`(383-432), `format_*_style`(471-517), 그리고 `bpe.cpp`/`sp.cpp` 중 하나
5. `cactus-engine/src/model.cpp`: `Model::init`(671-917), 라우트 선택(737-805), `run_step`(1403-1436), `run_chunked_prefill`(2165-2376), `decode`(4590-4655), `sample_component_logits`(3101-3215)
6. `cloud.cpp`, `rag.cpp`, `kv_compress.cpp`, `constraints.cpp`는 관심 순서대로

**실습**
1. `cactus_log_set_level(0)`으로 DEBUG 로그를 켜고 한 번의 `cactus_complete` 동안 어떤 컴포넌트 그래프가 몇 번 `execute` 되는지 센다.
2. `CACTUS_KV_COMPRESS_AT=64 CACTUS_KV_COMPRESS_TO=32`로 KV 축출을 강제로 일찍 발동시켜 긴 대화의 품질 변화를 관찰한다.
3. `handoff_probe.bin`이 있는 Gemma 4 번들과 없는 번들에서 `confidence_threshold`가 어떻게 달라지는지 응답 JSON으로 확인한다.

**확인 질문**
- 왜 첫 토큰은 temperature와 무관하게 argmax인가? (`model.cpp:3217-3302`)
- 정지 시퀀스 매칭은 문자열 기준인가 토큰 id 기준인가? (`init.cpp:340-350`)
- 어떤 조건에서 클라우드 핸드오프가 "영구적으로" 꺼지나? (`complete.cpp:919-925`)

---

## 3주차: 계산 그래프

**읽기 순서**
1. `cactus-graph/cactus_graph.h`: `Precision`/`OpType`(92-151), `BufferDesc`(255-344), `OpParams`(358-442), `GraphNode`(444-451), 빌더 메서드 목록(542-833)
2. `src/core.cpp`: `BufferPool`(6-58), `BufferDesc::get_data` 우선순위(173-185), `resize_from_pool`(215-223)
3. `src/builder.cpp`: `matmul`(140-165), `add_node`(1323-1352), `kv_cache_state`(1558 부근)
4. `src/execute.cpp`: `execute`(1494-2305)를 "CPU 고속 경로(1896-1925)"부터 읽고, 그다음 aliasing(1623-1647)과 liveness(1650-1682), 마지막에 Metal 분기
5. `src/io.cpp`: `parse_header`(1054-1145), 스케일 블롭(529-552), `save_graph`/`load_graph`(694-830), `MappedFileRegistry`(899-930)
6. `src/ops_cache.cpp` 전체(804줄)
7. `tests/test_graph.cpp`, `test_cache.cpp`, `test_dynamic_shapes.cpp`, `test_io.cpp`

**실습**
1. 04장 §6의 그래프에 `retain_outputs` 없이 중간 노드를 읽으면 무슨 일이 생기는지 확인하고, 이유를 `release_after`로 설명한다.
2. `CACTUS_PROFILE=1`(또는 `execute(profile_file)`)로 노드별 시간표를 뽑아 가장 비싼 op 5개를 찾는다.
3. `set_runtime_input_shape`로 배치 1→4→2를 바꿔 가며 같은 그래프를 재실행하고 풀 재할당이 일어나는지 `BufferPool` 통계로 본다.
4. Python `Graph`로 `kv_cache_state` + `kv_cache_append` + `attention_cached`를 손으로 엮어 2토큰 디코드를 흉내 낸다(`test_cache.cpp` 참고).

**확인 질문**
- "zero-copy"가 CPU 경로에서 실제로 성립하는 네 가지 경우는? (07장 §2)
- 노드 실행 순서는 어떻게 보장되나? 위상 정렬이 있나?
- KV 캐시가 INT8이라면 현재 스텝의 새 K/V는 어떤 정밀도로 어텐션에 들어가나?

---

## 4주차: 변환기와 양자화

**읽기 순서**
1. `docs/cactus_quants.md`, `blog/turboquant-h.md`
2. `python/cactus/convert/quantization/cq.py`: `make_codebook`(47-76), Hadamard(79-100), `quantize_hadamard`(245-293), `quantize_orthogonal`(296-311), `write_cq_tensor`(321-410)
3. `python/cactus/convert/cli.py`(408-611)와 `model_adapters/policy.py`(텐서별 정책)
4. `python/cactus/transpiler/CONTRIBUTING.md` → `Converter/convert.py` → `IR/simplify_ir.py` → `Fusions/fusions.py` → `Generator/generate.py` → `RuntimePlan/models.py`
5. `ModelProfiles/profiles.py`의 `MODEL_ID_MAP`과 프로필 하나(예: LFM2-VL)
6. 구 파이프라인이 궁금하면 `python/cactus/transpile/hf_model.py`와 `docs/cactus_transpiler.md`

**실습**
1. `cactus convert Qwen/Qwen2.5-0.5B ./w --bits 4`와 `--bits 2`로 만든 두 번들의 크기, 퍼플렉시티(간단한 프롬프트 로그확률), 속도를 비교한다.
2. `transpiler_ir/output_decode_*.json`과 `*_simplified.json`을 diff 해 어떤 패턴이 어떤 융합 op으로 바뀌었는지 5개 찾는다.
3. `weights_manifest.json`에서 FP16 폴백된 텐서 목록을 뽑고 `policy.py`의 규칙과 대조한다.
4. (심화) `cq.py`의 `GROUP_SIZE`를 64로 바꿔 변환한 뒤 커널이 어떤 경로를 타는지(06장 §2.5) 추적한다.

**확인 질문**
- CQ의 코드북은 데이터로 학습되나? (블로그와 코드가 다른 지점, 09장 §3)
- 임베딩과 LM head가 Hadamard 대신 ORTHOGONAL을 쓰는 이유는?
- 새 모델 패밀리를 추가하려면 어느 파일들을 건드려야 하나? (10장 §6)

---

## 5주차: 커널

**읽기 순서** — 06장 §10의 순서 그대로:
1. `cactus_kernels.h` 훑기
2. `threading.h`: `ParallelConfig`, `Thresholds`, `GemmThreading`, `CoreTopology`
3. `matmul.cpp`: 밀집 FP16 → `CactusQuantMatrix` → Hadamard 변환 → SDOT GEMV → 디스패처 → pair/triple → two-phase 드라이버
4. `attention_hybrid.cpp` 디코드 고속 경로 → `attention.cpp`
5. `quants.cpp`, `fused.cpp`, `norms_rope.cpp`
6. `tests/test_matmul.cpp`의 `SyntheticCQ`

**실습**
1. `tests/test_matmul.cpp`의 벤치마크를 기준으로 `CACTUS_NUM_THREADS=1,2,4,8`에서 GEMV/GEMM 처리량 곡선을 그린다. 디코드(M=1)가 스레드를 늘려도 안 빨라지는 지점을 찾는다.
2. `CACTUS_GEMV_SB_PER_THREAD`를 1~8로 바꿔 인터리브 GEMV 스레드 수와 성능의 관계를 본다.
3. NEON 인트린식 하나(`vdotq_laneq_s32`)를 골라 `TQ_SDOT_PANEL_T` 매크로가 4행×16k 패널을 어떻게 소비하는지 종이에 그려 본다.
4. (심화) 자신의 기기에서 `perf`/Instruments로 `cactus_quant_sdot_gemv_int8`의 IPC와 메모리 대역폭을 측정해 이론 상한(모델 크기 ÷ 대역폭)과 비교한다.

**확인 질문**
- CQ 커널이 가중치가 아니라 활성화를 회전시키는 이유는?
- `gs % 32 != 0`이면 어떤 경로로 떨어지고 왜 느린가?
- Android에서 GEMV 스레드가 항상 1인 근거는?

---

## 6주차: 가속기 백엔드와 튜닝

**읽기 순서** — 11장 전체, 그리고:
1. `cactus-graph/src/execute.cpp`의 Metal 분기(1569-1615, 1760-1887), `try_encode_metal`(638-1403)
2. `cactus-graph/src/metal_plan.cpp`의 규칙 1(어텐션 융합)과 규칙 2(QKV 배치 GEMV)
3. `cactus-kernels/src/metal_backend.mm`: `MetalCtx`(33-146), `wrapHostPtr`(305-354), `resident`(220-265), `session_sync`(406-421)
4. `cactus_kernels.metal`: `cq4_transform`, `cq4_gemv`, `cq4_gemm_mma`, `attn_decode_i8`
5. `cactus-engine/src/npu_ane.mm`, `model_npu.cpp`, `python/cactus/transpile/npu/README.md`
6. `cactus-graph/tests/test_metal_parity.cpp`

**실습**
1. Mac에서 `cactus benchmark`를 `--backend cpu`/`metal`로 실행하고 프리필 vs 디코드에서 GPU 이득이 어디서 나는지 정리한다.
2. `CACTUS_FLUSH_CADENCE`를 8, 48, 256으로 바꿔 디코드 tps 변화를 측정한다.
3. `metal_plan.cpp`의 규칙 하나를 비활성화(예: 조건에 `false &&`)하고 패리티 테스트와 성능이 어떻게 변하는지 본다.
4. (심화) 06장 §9의 상수 하나(예: `GEMV_MIN_N_BLOCKS`)를 자신의 기기에 맞게 조정하고 벤치마크로 검증한 뒤, CONTRIBUTING.md 규칙(벤치마크 첨부, 최소 범위)에 맞는 PR 초안을 써 본다.

**확인 질문**
- "노드 단위 폴백"과 "그래프 파티셔닝"의 차이는? Cactus는 어느 쪽인가?
- ANE 경로가 v2.2.1에서 실제로 실행되지 않는 이유를 코드로 설명하라.
- Android GPU 백엔드를 추가한다면 어떤 인터페이스(함수 시그니처)를 구현해야 하나? (11장 §7)

---

## 학습 중 도움이 되는 습관

- **`파일:줄` 대신 함수명으로 기억한다.** 이 자료의 줄 번호는 v2.2.1 기준이라 곧 어긋난다.
- **테스트를 문서로 읽는다.** `tests/test_*.cpp`는 각 API의 정확한 사용법과 허용 오차를 보여 준다.
- **env 변수로 실험한다.** 코드를 고치기 전에 `CACTUS_*` 환경변수(11장 §6 표)로 동작을 바꿔 가설을 검증한다.
- **CONTRIBUTING.md의 규칙**을 기억한다: 주석 대신 읽히는 코드, 벤치마크 없는 성능 PR 금지, DCO 서명.
