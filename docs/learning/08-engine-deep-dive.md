# 08. Cactus Engine 심화: C API, 런타임 모델, 생성 루프, 하이브리드

> 대상: `cactus-engine/cactus_engine.h`, `src/engine.h`(1475줄), `model.cpp`(5902줄), `init.cpp`, `complete.cpp`(1600줄), `utils.h`(2023줄), `tokenizer.cpp`/`bpe.cpp`/`sp.cpp`, `cloud.cpp`, `telemetry_impl.cpp`, `rag.cpp`/`index.cpp`, `kv_compress.cpp`, `constraints.cpp`, `npu_ane.mm`/`model_npu.cpp`, `tests/`

---

## 0. 가장 중요한 아키텍처 포인트

- **패밀리별 C++ 트랜스포머 코드도, `Model` 클래스 계층도 없다.** 구체 클래스 `cactus::engine::Model` 하나(`engine.h:674-1190`)가 **manifest에 의해 구동되는 런타임**이며, 사전 직렬화된 CactusGraph "컴포넌트"를 실행한다.
- 아키텍처는 **Python 트랜스파일러**가 정의한다(어댑터/프로필). C++은 (1) manifest 읽기 → (2) 존재하는 컴포넌트 이름으로 "디코드 경로" 선택 → (3) `CactusGraph::load` + mmap 바인딩 → (4) 토큰마다 `input_ids/position_ids/attention_mask`를 채우고 `graph->execute()`.
- MoE, 슬라이딩 윈도, PLE, 어텐션 싱크, DeltaNet, AltUp은 **그래프 op**이지 엔진 코드가 아니다.
- 그래서 `create_model()`은 `components/manifest.json` 존재만 확인하고 `make_unique<Model>()`을 돌려준다(`model.cpp:5883-5893`).

---

## 1. 공개 C API (`cactus_engine.h`)

타입: `cactus_model_t`, `cactus_index_t`, `cactus_stream_transcribe_t`는 모두 `void*`. 콜백 `cactus_token_callback(const char* token, uint32_t token_id, void* user_data)`.

| 함수 (헤더 줄) | 목적 | 구현 |
|---|---|---|
| `cactus_init(model_path, corpus_dir, cache_index)` :27 | 번들 로드, 선택적으로 RAG 인덱스 빌드/로드 | `init.cpp:356` |
| `cactus_destroy` :33 / `cactus_reset` :34 / `cactus_stop` :35 | 해제 / KV·처리 토큰 초기화 / `should_stop` 세팅 | `init.cpp:442-458` |
| `cactus_set_backend("cpu"\|"metal")` :37 | 전역 백엔드 선택 | `init.cpp:354` → `cactus-graph/src/execute.cpp:25` |
| `cactus_complete(model, messages, buf, size, options, tools, cb, user, pcm, pcm_size)` :39-50 | 채팅 완성(텍스트·이미지·오디오·툴·스트리밍) | `complete.cpp:814` |
| `cactus_prefill` :52 | 대화로 KV 캐시만 채움 | `complete.cpp:1286` |
| `cactus_tokenize` :63 / `cactus_render_prompt` :71 | 토큰화 / 렌더된 템플릿 문자열 | `complete.cpp:1380, 1406` |
| `cactus_score_window` :80 | 로그확률 점수 — **스텁(항상 0)** | `model.cpp:5895-5899` |
| `cactus_benchmark_tokens` :91 | 토큰 id로 프리필/디코드 벤치 | `complete.cpp:1473` |
| `cactus_transcribe` :100 | Whisper/Parakeet-TDT ASR (그 외 모델은 오디오 첨부 `complete`로 라우팅) | `transcribe.cpp:155` |
| `cactus_preprocess_audio_features` :113 | WAV → 멜 특징 | `transcribe.cpp:58` |
| `cactus_stream_transcribe_start/process/stop` :124-141 | 스트리밍 ASR | `stream.cpp:326/355/383` |
| `cactus_embed` :143 / `cactus_image_embed` :152 / `cactus_audio_embed` :160 | 텍스트/이미지/오디오 임베딩 | `embed.cpp` |
| `cactus_rag_query` :168 | 코퍼스 하이브리드 검색 | `rag.cpp:360` |
| `cactus_index_init/add/delete/get/query/compact/destroy` :176-222 | 독립 벡터 DB | `index_ffi.cpp` |
| `cactus_get_last_error` :224 | thread_local 마지막 오류 | `last_error.cpp:6` |
| `cactus_log_set_level(0..4)` / `cactus_log_set_callback` :227-230 | 로깅 (기본 WARN) | `log.cpp` |
| `cactus_set_telemetry_environment`, `cactus_set_app_id`, `cactus_telemetry_flush`, `cactus_telemetry_shutdown` :232-235 | 텔레메트리 | `telemetry.cpp` |

**없는 것**: `cactus_version()` 함수(버전은 `CACTUS_COMPILE_TIME_VERSION` define), 클라우드 인증 C 함수(키는 env/캐시 파일), 옵션의 `backend` 키, `"npu"` 백엔드 값.

### 1.2 `options_json` 키 (`parse_inference_options_json`, `utils.h:1586-1721`; 기본값 `InferenceOptions`, :498-518)
파서는 진짜 JSON 파서가 아니라 부분 문자열 매칭이다.

| 키 | 기본 | 비고 |
|---|---|---|
| `temperature` | 0.0 | 음수면 모델 기본값(`model.cpp:4594`). **0.0은 greedy**(≤0.011, :4597) |
| `top_p` | 0.0 | ≤0 또는 >1이면 모델 기본값 |
| `top_k` | 0 | 0이면 모델 기본값, 샘플러에서 512 상한 |
| `min_p` | 0.15 | [0,1] 클램프 |
| `repetition_penalty` | 1.1 | 최근 64토큰 대상 |
| `seed` | 0 | 0 = 재시드 안 함 |
| `max_tokens` | 100 | |
| `stop_sequences` | [] | 문자열 배열 |
| `include_stop_sequences` | false | |
| `force_tools` | false | 제약 디코딩 켬 |
| `tool_rag_top_k` | 2 | 툴 가지치기 |
| `confidence_threshold` | -1 | → 해석된 기본값(§4) |
| `auto_handoff` | true | |
| `cloud_timeout_ms` | 15000 | |
| `handoff_with_images` | true | |
| `enable_thinking_if_supported` | false | thinking 스위치 |
| `timestamps` | false | Whisper 세그먼트 |
| `telemetry_enabled` | true | **파싱만 되고 읽히지 않음** |
| `use_vad` | true | **파싱만 되고 읽히지 않음** |
| `language` | – | Whisper 전용, 별도 파싱(`transcribe.cpp:290`) |

패밀리별 샘플링 기본값(`model.cpp:5800-5819`): Gemma 계열 temp 1.0/top_p 0.95/top_k 64(Gemma 4 핸드오프 임계 0.81), LFM2 0.3/0.95/20, Qwen 0.6/0.95/20, Qwen3.5 0.7/0.8/20.

### 1.3 `messages_json` (`parse_messages_json`, `utils.h:989-1152`)
`[{"role","content":"<문자열>"}]` 필수(콘텐츠 파트 배열 미지원). 선택: `"images":[path]`, `"audio":[path.wav]`(절대경로로 변환), `role:"tool"`의 `"name"`, `"tool_calls":[{"function":{"name","arguments"}}]`. 원시 PCM은 `pcm_buffer` 인자로 전달하며 Gemma 4에서는 마지막 user 메시지에 붙는다(`complete.cpp:595-604`).
`tools_json`은 OpenAI 스타일(`parse_tools_json`, `utils.h:1155+`).

### 1.4 응답 JSON (`construct_response_json`, `utils.h:1898-1952`)
`success, error, cloud_handoff, cloud_handoff_reason, response, context_response(비어 있지 않을 때), thinking(비어 있지 않을 때), function_calls, segments[{start,end,text}], confidence, confidence_threshold, time_to_first_token_ms, total_time_ms, prefill_tps, decode_tps, ram_usage_mb, prefill_tokens, decode_tokens, total_tokens`.

---

## 2. `engine.h` / `model.cpp`: 런타임 모델

### 2.1 Config와 패밀리
- `Config::ModelType`(`engine.h:152`): `QWEN=0, GEMMA=1, NOMIC=3, LFM2=5, SIGLIP2=6, WHISPER=7, MOONSHINE=8, PARAKEET=10, QWEN3P5=11, PARAKEET_TDT=12, GEMMA3N=13, YOUTU=14, GEMMA4=15, NEEDLE=18, GENERIC=19`.
- `Config::from_json`(`model.cpp:5556`)은 이름과 달리 **`key=value` 텍스트 `config.txt`** 를 파싱한다. `model_type` 문자열(:5643-5658): `qwen`, `qwen3p5|qwen3_5`, `gemma|gemma3`, `gemma3n`, `lfm2`, `whisper`, `parakeet_tdt|parakeet-tdt`, `youtu`, `needle`, `bert|nomic`, `generic`. **그 외는 GEMMA4로 폴백**(:5657).
- 실제 트랜스파일러 어댑터가 있는 패밀리: qwen3/qwen2_moe/qwen3_5(텍스트·비전), gemma3(텍스트), gemma4(텍스트+비전+오디오), lfm2/lfm2_moe/lfm2_vl, needle, parakeet_tdt, whisper, nomic. Moonshine, Gemma 3n, Youtu, Parakeet-CTC의 enum은 흔적이다.

### 2.2 컴포넌트와 경로
- `Component`(`engine.h:911-924`): 이름, `graph_path`, 런타임 입출력 노드 id, 가중치 `bindings`, `cache_states`, 메타데이터, `unique_ptr<CactusGraph>`, 입력 버퍼.
- 컴포넌트 포인터(:1038-1056): `encoder_, decoder_, decoder_prefill_, decoder_prefill_chunk_, decoder_prefill_cache_chunk_, decoder_prefill_logits_head_, vision_encoder_, vision_projector_, audio_encoder_, lm_encoder_media_step_, source_encoder_, decoder_cross_kv_` 등.
- `DecodeRoute`(:1043): `CACHED_STEP`, `DIRECT_DECODER_STEP`, `FULL_CONTEXT_TEXT`, `ENCODER_CROSS_KV_STEP`.

### 2.3 `Model::init` (`model.cpp:671-917`)
1. `config.txt` → `Config`; `CACTUS_KV_COMPRESS_AT/TO` env 오버라이드
2. `load_manifest()`(919-1099): `family`, 프롬프트/미디어 스타일과 토큰, `suppress_generation_token_ids`, `repetition_penalty_scope`, `npu_*_encoder` 경로, `components[]`
3. `setup_tokenizer()`(1101-1113)
4. 텍스트 임베딩 전용 번들은 조기 반환(722-731)
5. **경로 선택**(737-805) 우선순위: `runtime_route=encoder_cross_kv_decoder_step` 메타 → `ENCODER_CROSS_KV_STEP`(Whisper, Needle); `lm_encoder_step + decoder_media_step + lm_encoder_text_chunk + decoder_prefill_chunk` → `CACHED_STEP`; `decoder_step`(ids/pos 입력) → `DIRECT_DECODER_STEP`; `lm_encoder_step + decoder_step` → `CACHED_STEP`; `decoder_full_context` 또는 `text_lm_encoder + decoder` → `FULL_CONTEXT_TEXT`; `audio_encoder + decoder|decoder_joint` → `DIRECT_DECODER_STEP`(Parakeet)
6. 필요한 컴포넌트마다 `load_component_graph`(1123-1163): `CactusGraph::load` → `bind_mmap_weights` → 입력 버퍼·캐시 상태 바인딩
7. Metal 가중치 프리웜(827-828), 프리필 그래프를 0으로 **워밍업 실행**(872-885)
8. 비전 인코더 출력 형상에서 이미지 소프트 토큰 수 읽기(887-896)
9. `handoff_probe.bin`(매직 `CHP10P6`) 로드(904-913; 로더 521-575)

### 2.4 스텝 실행
```cpp
// model.cpp:1430-1433 (CACHED_STEP)
run_encoder_step(token_id, position);            // lm_encoder_step: ids/pos → embeds (+PLE)
copy_component_outputs_to_inputs(*encoder_, *decoder_);
fill_int_input(*decoder_, "attention_mask", 1, position + 1);
decoder_->graph->execute();                      // decoder_step (KV_CACHE_STATE 노드 포함)
```
- `DIRECT_DECODER_STEP`은 ids/pos를 `decoder_step`에 직접 쓴다(1404-1413).
- Metal에서는 `extract_ple_pathway`로 인코더 그래프의 PLE 경로를 뽑아 디코더만 `FusedEmbedCtx`로 돌리는 융합 경로가 있다(1415-1429).
- 컴포넌트는 사용 후 언로드해 RAM을 아낄 수 있다(Metal에서는 no-op, 1165-1166).

### 2.5 프리필 vs 디코드
- **프리필**은 청크 단위(`run_chunked_prefill`, 2165-2376): `decoder_prefill_chunk`/`decoder_prefill_cache_chunk`(로짓 없는 캐시 전용) + 선택적 `decoder_prefill_logits_head`. 청크 크기는 그래프에 박혀 있다(`prefill_chunk_tokens` 메타 / `inputs_embeds` 형상). 꼬리 토큰은 (a) conv/recurrent/슬라이딩 상태가 없으면 패딩 청크, (b) 슬라이딩 캐시면 snapshot/rollback 패딩, (c) 아니면 `run_step`으로 토큰 단위. KV 상태는 프리필 컴포넌트에서 스텝 디코더로 **이동**(`move_cache_states`, 2362-2370). 재귀 상태 모델(Qwen3.5)은 청크 1개 제한(2244-2246).
- **디코드**는 `Model::decode`(4590-4655): 스텝 → `maybe_roll_compact()` → greedy argmax 또는 `sample_component_logits`.
- **첫 토큰**: 새 텍스트 프롬프트는 `prefill_and_sample_first_token`(3217-3302)이 프리필 로짓의 **argmax**로 뽑는다(temperature와 무관).

### 2.6 비전·오디오
- 이미지 전처리(C++): SigLIP2/LFM2-VL 타일링, Gemma 4 패치+위치 id, Qwen3-VL(`engine_image.cpp:131-923`). `run_vision_encoder`가 `vision_encoder` 그래프를 실행해 "media features"로 게시(`model.cpp:2513-2609`); LFM2-VL은 `vision_projector` + 위치 임베딩 그리드 파일 추가(2629-2670).
- 오디오: 멜 스펙트로그램은 엔진이 계산(`utils.h`의 Whisper/Parakeet/Gemma 4 설정, `transcribe.cpp:241-270`). `run_audio_encoder`가 프레임 용량에 맞춰 청킹(2799+). Parakeet는 `audio_frames` 메타로 **버킷된** `audio_encoder*` 컴포넌트를 고른다(`select_audio_encoder`, 4697-4710).
- LM 주입(`prefill_with_media`, 4425-4588): 청크 미디어 프리필(`token_row_replacement`), "warm media step"(플레이스홀더 토큰마다 `lm_encoder_media_step`으로 특징 행 1개 치환), 텍스트 전용 폴백.
- ASR: Whisper는 cross-KV seq2seq(`transcribe_whisper_seq2seq`, 4665), Parakeet는 TDT greedy(`transcribe_parakeet_tdt`, 4712).

### 2.7 배치
`decode_batch/generate_batch/set_decode_slots`(1507-1625)가 있고 테스트에서 쓰이지만 **C API에는 노출되지 않는다**.

---

## 3. `init.cpp`와 번들, 백엔드 선택

번들 파일 목록은 01장 §4 참고. `cactus_init`(356-440): `CACTUS_NO_CLOUD_TELE`/`CACTUS_DISABLE_CLOUD_HANDOFF` 적용 → `telemetry::init` → `create_model` → `Model::init(path, DEFAULT_CONTEXT_SIZE=512)`(512는 `cache_max_seq_len_`만 정하고 실제 용량은 그래프에 박혀 있음) → 코퍼스가 있으면 `index.bin/data.bin` 로드 또는 빌드 → `recordInit`.

백엔드: 런타임 선택지는 **CPU vs Metal**뿐. `cactus_default_backend()`는 Metal이 가능하면 Metal(`execute.cpp:17-34`). `cactus_set_backend`는 그 외 문자열에 -1.

---

## 4. `complete.cpp`: 생성 루프

### 4.1 프롬프트 준비 (`prepare_prompt`, 535-671)
옵션·메시지 파싱 → 자동 RAG 주입(42-59) → 툴 파싱 및 `tool_rag_top_k`로 가지치기(562-567) → **임계값 해석**(576-585: 프로브 있으면 0.50, 아니면 모델 기본(Gemma 4 0.81), 아니면 0.7) → Gemma 4 오디오 전처리(587-619) → 패밀리별 툴 포맷(Needle JSON / Qwen / LFM2 `chat_tools.h:66-88` / Gemma `gemma_tools.h:428`) → `force_tools`면 `setup_tool_constraints`(temperature 0은 0.01로) → `format_chat_prompt` → `encode`.

### 4.2 채팅 템플릿
**Jinja 인터프리터가 없다.** 패밀리별 C++ 하드코딩(`tokenizer.cpp:471-517`): `format_qwen_style`(ChatML; `chat_template.jinja2`에 `<|im_start|>`가 있는 미지 모델에도 적용), `format_lfm2_style`, `format_needle_style`, `format_gemma_style`, `format_gemma4_style`(기본). 템플릿이 없으면 "System:/User:/Assistant:"(519-540).

### 4.3 프리필과 KV 재사용
`do_prefill`(673-760): `processed_tokens`가 새 프롬프트의 접두사면 캐시 재사용(`prompt_context_matches`, 511-533). 마지막 토큰을 제외하고 프리필한 뒤 마지막 토큰을 디코드.

### 4.4 샘플링 (`Model::sample_component_logits`, `model.cpp:3101-3215`)
top-k 후보(≤512) → temperature softmax → top-p → min-p → RNG. 반복 패널티는 최근 64토큰(3061-3099). 툴 제약과 어휘 bias는 가산. Metal에서는 argmax/패널티가 그래프 안에서 돈다(`cactus_graph_metal_argmax`).

### 4.5 정지 시퀀스
`build_stop_sequences`(356-391): EOS + 사용자 지정 또는 패밀리 기본(`<|im_end|>`, `<end_of_turn>`, `<turn|>`; `tokenizer.cpp:434-453`) + 템플릿 없으면 `"\nUser:"`류 + Gemma 4는 `<turn|>`, 툴 있으면 `<|tool_response>`. 매칭은 **토큰 id 접미사**(`init.cpp:340-350`), 매치된 접미사는 잘라낸다(393-405).

### 4.6 스트리밍 콜백 계약
- 토큰마다 `callback(tokenizer->decode({id}).c_str(), id, user_data)`(1071-1074, 1104-1107). 토큰을 개별 디코드하므로 멀티바이트 문자가 쪼개질 수 있다.
- 정지 토큰은 절대 방출되지 않는다.
- 클라우드 응답이거나 프로브가 스트리밍을 보류한 경우 **전체 텍스트를 한 번에 `token_id=0`으로** 호출(935-937, 1193-1195).
- `cactus_stop`이 루프를 끊는다(1077). `cactus_complete`는 `model_mutex`를 잡지 않는다(transcribe/stream은 잡음).

### 4.7 툴콜 파싱과 제약 디코딩
- `parse_function_calls_from_response`(`utils.h:1723-1831`) 순서: Needle `<tool_call>[...]` → Gemma `<|tool_call>`/`<start_function_call>`(`gemma_tools.h:763-858`) → Qwen `<tool_call>` JSON·LFM2 pythonic(`chat_tools.h:160-258`) → `"function_call"` 마커 → `name`+`arguments`를 가진 아무 JSON.
- `constraints.cpp`의 `ToolCallConstrainer`는 **범용 JSON/문법 엔진이 아니다.** Needle과 Gemma 계열에서만(437-438) 호출 태그를 강제하고, 툴 이름·파라미터 키·enum 값 트라이로 토큰에 `-1e9` bias를 준다(82-133). 필수 파라미터 추적(164).

### 4.8 Thinking
`enable_thinking_if_supported`가 템플릿에 전달되고, 생성 후 `partition_thinking_response`가 `<|channel>…<channel|>`(Gemma 4) 또는 `<think>…</think>`를 분리(`utils.h:1867-1890`). Gemma 4는 항상 잘라내되 플래그가 켜져 있을 때만 `thinking` 필드로 반환(1143-1151).

### 4.9 confidence와 클라우드 핸드오프 결정
- **"entropy"는 섀넌 엔트로피가 아니다.** `1 - sigmoid(best_logit - second_logit)`(`uncertainty_from_margin`, `model.cpp:2905-2912`). 일반 경로에서 `confidence = 1 - first_token_uncertainty`(1033).
- 트리거: (a) 임계값 ≥ 1.0이면 로컬 생성 전 클라우드, 실패 시 오류(965-984); (b) 프로브 없고 첫 토큰 confidence < 임계값이면 스트리밍 전 클라우드, 실패 시 로컬 폴백(1046-1069); (c) 프로브가 있으면(Gemma 4/Parakeet) 스트리밍 없이 로컬 생성 후 `confidence = 1 - p_wrong`(소형 MLP over `probe_hidden`, `model.cpp:602-669`), confidence가 낮거나 로컬 툴콜에 필수 인자가 빠지면 핸드오프(1164-1191).
- 자격 조건(876-878): `auto_handoff` true, `CACTUS_DISABLE_CLOUD_HANDOFF` 미설정, 이 핸들에서 이전 401/403 없음(있으면 영구 비활성, 919-925), 이미지가 없거나 `handoff_with_images` true.
- `cloud_handoff_reason`: "handoff off", "disabled (env)", "images kept local", "low confidence", "above threshold", "kept local", "handoff failed: …".

### 4.10 지표
`ttft`는 `cactus_complete` 진입부터. `prefill_tps = prompt_tokens/ttft`, `decode_tps = (n-1)/(total - ttft)`(1129-1135). `ram_usage_mb`는 프로세스 footprint.

---

## 5. NPU(Apple Neural Engine) 경로: 존재하지만 v2.2.1에서는 비활성

- **CoreML `.mlpackage` 인코더만** `MLModel`로 실행한다. MPSGraph나 커스텀 ANE 코드는 없다. 범위는 **오디오 인코더, 비전 인코더, Needle 소스 인코더**. 텍스트 디코더 프리필/디코드는 NPU에 올리지 않는다(`python/cactus/transpile/npu/README.md:9-10`; 프리필 NPU가 제거된 이유는 :32-47 — 변환 시 OOM, CPU 청크 프리필이 충분히 빠름).
- 빌드: Apple에서 `src/npu_ane.mm`(ARC, CoreML 프레임워크), 그 외 `src/npu.cpp` 스텁(`is_npu_available()` false). `CACTUS_HAS_ANE = __APPLE__`.
- 아티팩트 생성(Python): `python -m cactus.transpile.hf_model --npu [--npu-quantize 0|4|8] [--npu-audio-quantize] [--npu-vision-quantize]` → `run_encoder_pipeline` → 같은 PyTorch 어댑터 모듈을 `torch.export` → `coremltools.convert(mlprogram, FLOAT16, iOS17/18)`, 기본 양자화 오디오 int8 / 비전 fp16. manifest에 `npu_audio_encoder`/`npu_vision_encoder`/`npu_source_encoder` 경로와 `npu_audio_compute_units="CPU_AND_NE"` 기록. **공개 `cactus convert` CLI에는 `--npu` 플래그가 없다.**
- 런타임(`npu_ane.mm`): `.mlpackage`를 **런타임에 컴파일**해 `.mlmodelc` 캐시(`CACTUS_ANE_FORCE_RECOMPILE`로 강제). compute units 기본 `CPUAndNeuralEngine`, 단 경로에 `audio_encoder`/`vision_encoder`가 있으면 기본 **`CPUAndGPU`**(:96-101). 오버라이드: manifest 힌트 → `CACTUS_ANE_COMPUTE_UNITS` → `_ENCODER_` → `_AUDIO_`/`_VISION_` (값 all/cpu_and_ne/cpu_and_gpu/cpu_only). I/O는 fp16 `MLMultiArray`.
- 엔진 접착(`model_npu.cpp`): `load_npu_{audio,vision,source}_encoder`, `audio/vision/source_encode_via_npu`(불일치 시 false → CPU 그래프).
- **HEAD 상태**: `load_npu_*_encoder`에 **호출자가 없다**(선언 `engine.h:796-800`과 정의뿐). manifest의 `npu_*_mlpackage_` 경로는 파싱만 되고(`model.cpp:1031-1042`) 쓰이지 않는다. 따라서 `has_npu_vision_encoder()`는 항상 false이고 `vision_encode_via_npu` 호출 지점(2570, 4060)은 실행되지 않는다. 히스토리: c4c9cfa에서 추가 → "Remove coreml (#754)" 68cb6a9에서 제거 → 커스텀 트랜스파일러 브랜치 338b5a5에서 재추가 → 병합 c90a1af 이후 부재.
- Android/기타 NPU(NNAPI, QNN/Hexagon, NeuroPilot), Vulkan, OpenCL 경로는 **없다**.

---

## 6. 클라우드 핸드오프와 텔레메트리

### `cloud.cpp`
- 베이스 URL `CACTUS_CLOUD_API_BASE`(기본 `https://104.198.76.3/api/v1`), 모델 `CACTUS_CLOUD_MODEL`(기본 `gemini-2.5-flash`).
- 엔드포인트: `/text`, `/vlm`(첫 이미지 base64), `/omni`(텍스트+이미지+16kHz WAV), `/transcribe`(Parakeet).
- 전송 내용(`build_cloud_text_prompt`, 126-163): 출력 계약 프리앰블, 전체 대화(툴콜·결과 포함), 툴 JSON, **로컬 모델 초안**.
- 헤더 `X-API-Key`, `CACTUS_CLOUD_HEADERS` 추가 가능. 연결 타임아웃 `min(timeout, 2000)`ms.
- **TLS 피어/호스트 검증이 기본 꺼져 있고** `CACTUS_CLOUD_STRICT_SSL`로만 켜진다(298-305). 보안상 인지하고 사용해야 한다.
- 키 해석 순서: `CACTUS_CLOUD_KEY` → `CACTUS_CLOUD_API_KEY` → `<telemetry_dir>/cloud_api_key` 캐시. `cactus auth`는 `~/.cactus/config.json`(0600)에 저장하고 CLI가 `CACTUS_CLOUD_KEY`로 export. 키가 없으면 네트워크 대기 없이 `missing_api_key`.

### 텔레메트리 (`telemetry_impl.cpp`)
- **기본 ON(옵트아웃).** `init()`이 `enabled=true`, 모든 `cactus_init`에서 호출. 옵션 `telemetry_enabled`는 무시된다.
- 끄기: `CACTUS_NO_CLOUD_TELE` 또는 `CACTUS_DISABLE_CLOUD_HANDOFF`.
- 목적지: Supabase(`CACTUS_SUPABASE_URL/KEY`로 변경 가능), 테이블 `projects/devices/logs`.
- 이벤트 필드: 이벤트 종류, 모델명, 성공 여부, cloud_handoff, ttft/tps/응답 시간/RAM/confidence/토큰 수, project_id(`CACTUS_PROJECT_ID` 또는 git remote URL 기반 UUID 또는 device id), `key_hash`(**이름과 달리 raw `CACTUS_CLOUD_KEY` 값이 들어감**, 1050-1052), 프레임워크/버전, 랜덤 device_id, app_id, 메시지, 오류. **프롬프트/응답 내용은 보내지 않는다.**
- 저장 위치: `$HOME/Library/Caches/cactus/telemetry`(Linux에서도), Android는 앱 캐시.

---

## 7. 토크나이저 (`tokenizer.cpp`, `bpe.cpp`, `sp.cpp`)
- 타입 `{UNKNOWN, BPE, SENTENCEPIECE}`(`engine.h:288`). `tokenizer_type=BPE`이거나 미지정 + `merges.txt` 존재 → BPE, 아니면 SentencePiece(`model.cpp:1106-1111`). **WordPiece 없음, `tokenizer.json` 직접 로딩 없음**(added_tokens만 채굴).
- BPE: `vocab.txt`/`merges.txt` mmap, GPT-2 byte-level 유니코드 매핑, 우선순위 병합, 공백 전용 토큰 병합 합성(170-186).
- SentencePiece: `ID\tTOKEN\tSCORE`, `sp_model_type=bpe`면 BPE, 아니면 **트라이 기반 greedy 최장 일치**(Viterbi unigram 아님, 275-360), `▁` 처리, `sp_add_dummy_prefix`, `<0xNN>` 바이트 폴백.
- 특수 토큰으로 텍스트를 먼저 분할한 뒤 인코딩(292-371). 템플릿용 패밀리 감지는 `config.txt` 부분 문자열 매칭(383-432).

---

## 8. RAG, 인덱스, 임베딩
- 자동 RAG(`init.cpp`): `*.txt/*.md` 스캔(77-98) → 마크다운 헤더/빈 줄로 분할(100-141) → ≤128토큰 청크, 큰 문단은 stride 96(overlap 32), 24토큰 미만 꼬리는 병합(143-214).
- 임베딩 모델은 **같은 로드된 모델**: `text_embedding` 컴포넌트(Nomic)가 있으면 BOS+text+EOS 평균 풀링 L2 정규화(5478-5554), 아니면 LM의 `decoder_embed_chunk` `last_hidden_state` 평균 풀링(5372-5476).
- 인덱스 포맷(`engine.h:1375-1472`): mmap `index.bin`(엔트리 `{int32 doc_id, u64 data_offset, u8 flags(tombstone)}` + fp16 L2 정규화 임베딩)과 `data.bin`(`{u16 content_len, u16 metadata_len}` + 바이트), 매직 `CACT` v1. 삭제는 tombstone, `compact()`가 재작성.
- 검색: 전수 fp16 내적(=코사인) + 최소 힙 top-k(`index.cpp:313-395`) → 상위 20에 BM25(k1 1.5, b 0.75) → RRF(k=60, 0.8/0.2) → 상위 5(`rag.cpp:116-208`). 컨텍스트는 "use ONLY this information" 지시와 함께 시스템 메시지 앞에 붙는다. Tool RAG도 같은 RRF.

---

## 9. KV 압축 (`kv_compress.cpp`)
- 양자화가 아니라 **KeyDiff식 롤링 토큰 축출**. 점수 `s_i = -cos(k_i, mean(k))`를 **RoPE 제거한 키**로 계산(`kv_compress.h:33-34`).
- (layer, KV head)별 keep-set: 싱크 4 + 최근 `recent_frac`(0.30) 예산 + 상위 점수 중간 토큰. 특수 토큰 행 보호(`SpecialRowTracker`).
- 선택 후 생존자를 모으고 K를 새 위치 0..B-1로 재RoPE(`compact_fp16/int8`), 비압축 층의 최근 K는 −Δ로 재RoPE(`model.cpp:5285-5317`), 캐시 버퍼 축소.
- 참여 층: `physical_compressible_layers`(KV 공유 Gemma 층 제외, 예: Gemma는 {4,9}), MLA(V 차원 ≠ K)는 건너뜀.
- 발동: `maybe_roll_compact()`(5335-5344)가 프리필 후·디코드 스텝마다 `cache_total_seq_len_ ≥ trigger`이면. 기본 ON: 4096에서 2048로(`engine.h:180-187`). `config.txt`의 `kv_compress*` 또는 `CACTUS_KV_COMPRESS_AT/TO`(AT=0이면 비활성), `CACTUS_KV_PRESERVE_SPECIAL`.

---

## 10. 테스트
- `test.sh`: `--ios/--android/--suite/--model/--transcription-model/--backend`. `components/manifest.json`이 있는 번들 필요(`cactus test`가 준비). env `CACTUS_TEST_MODEL`, `CACTUS_TEST_TRANSCRIPTION_MODEL`, `CACTUS_TEST_ASSETS`, `CACTUS_INDEX_PATH`, `CACTUS_TEST_BACKEND`.
- `run.cpp`는 테스트가 아니라 **대화형 REPL**(SDL2가 있으면 빌드, `cactus run`의 실체). `transcribe.cpp`는 별도 CLI.
- 스위트: `test_benchmark, test_bucket, test_curl, test_embed, test_index, test_kv_compress, test_llm, test_model_loading, test_needle, test_rag, test_stream_transcribe, test_stt, test_telemetry, test_vlm`.
- `test_llm.cpp`: 스트리밍+2턴 회상("Henry", KV 접두사 재사용), prefill, 메시지 변경 시 prefill 무효화, 청크 프리필 패딩, 툴콜(단일/다중, `force_tools`), 제약 bias 해제, thinking 분리/유지, 배치 생성(ragged, 처리량, distinct-4 = single).
- 온디바이스 러너는 02장 §3 참고.

---

## 11. `cactus_complete()`에서 첫 스트리밍 토큰까지 12단계

1. 진입·검증: Metal 프리필 캐시 트림 가드, 핸들 확인, ASR 모델이면 거부(826-864)
2. 요청 파싱: 옵션 → `InferenceOptions`, 메시지 → `ChatMessage`(`utils.h:1586, 989`)
3. 자동 RAG: 코퍼스가 있으면 같은 모델로 쿼리 임베딩 → 코사인 top-20 → BM25+RRF → top-5를 시스템 메시지에 prepend
4. 툴: 파싱 → Tool-RAG 가지치기 → 패밀리 포맷 → `force_tools`면 `ToolCallConstrainer` 무장
5. 임계값 해석(프로브 0.5 / Gemma 4 0.81 / 0.7)과 핸드오프 자격 판단(576-585, 876-892)
6. 렌더·토큰화: 하드코딩 템플릿 → BPE/SP encode(658-668)
7. 강제 클라우드: 임계값 ≥ 1.0이면 `/text|/vlm|/omni` POST 후 반환
8. 정지 시퀀스 구성(356-391, 986)
9. 프리필: 새 텍스트 프롬프트는 `prefill_and_sample_first_token`(청크 프리필 → 꼬리 처리 → KV 이동 → argmax); 아니면 `do_prefill`(캐시 접두사 재사용, 미디어 인코더·주입, 마지막 토큰 제외 프리필)
10. 첫 토큰(비고속 경로): `Model::decode` → `run_step`(lm_encoder_step → 복사 → decoder_step `execute`) → 샘플링; `maybe_roll_compact`
11. 조기 클라우드 검사: `confidence = 1 - 마진 불확실성`, TTFT 계산; 프로브 없고 임계값 미만이면 스트리밍 전 클라우드 시도, 실패 시 로컬(1030-1069)
12. 스트리밍: 첫 토큰이 정지 시퀀스가 아니고 프로브가 보류하지 않으면 `callback(decode(token), id, user)`(1071-1074) → 디코드 루프(1076-1108) → 툴/thinking 파싱, 사후 프로브 핸드오프, 응답 JSON, 텔레메트리(1113-1237)
