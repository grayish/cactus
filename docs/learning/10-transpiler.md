# 10. 변환기와 트랜스파일러: HuggingFace 모델이 Cactus 번들이 되기까지

> 근거: `python/cactus/cli/{convert,model,transpiler,transpile}.py`, `python/cactus/convert/`, `python/cactus/transpiler/`(신), `python/cactus/transpile/`(구), `python/cactus/transpiler/CONTRIBUTING.md`, `docs/cactus_transpiler.md`

---

## 0. 두 개의 파이프라인이 공존한다

| | 구 `python/cactus/transpile/` | 신 `python/cactus/transpiler/` |
|---|---|---|
| 문서 | `docs/cactus_transpiler.md`(6단계 파이프라인, `raw_ir.json`/`optimized_ir.json`/`graph_bindings.json`) | `python/cactus/transpiler/CONTRIBUTING.md`("삼두정치": Crassus/Pompey/Caesar) |
| 산출물 | `graph.cactus`, `components/manifest.json`, IR JSON | `components/<name>/*.cactus`, `components/manifest.json`, **`runtime_plan.json`**, `transpiler_ir/output_<mode>[_simplified].json` |
| 기본 경로 | 아니오 | **예** (`cactus convert`/`run`의 기본) |
| 아직 쓰이는 경우 | Parakeet TDT(`--task tdt_transcription --component-pipeline on` 서브프로세스, `cli/transpiler.py:285-330`), 임베딩 모델(`run_transpile`, `cli/model.py:247-262`), `--task` 비일반값/`--prompt`/`--image-file` 등 특정 옵션(`cli/convert.py:16-35`), NPU CoreML 방출(`transpile/npu/`), JAX 사용자 그래프 | 나머지 전부 |

`docs/cactus_transpiler.md`는 구 파이프라인 기준이며, `docs/cactus_quants.md:186-188`과 `docs/finetuning.md:68-74`의 "그래프 빌더 재작성 중이라 로컬 번들 생성 불가" 문구는 현재 코드(`cli/model.py:234-275`가 번들을 만든다)와 맞지 않는다.

---

## 1. `cactus convert`의 4단계 (`cli/convert.py:136-203`)

```
HF 체크포인트
   │ (1) 선택적 LoRA 병합 (PEFT merge_and_unload → 임시 디렉터리)
   ▼
convert/cli.py  ── (2) ensure_weights: CQ 가중치 + 토크나이저 + config.txt
   ▼
transpiler/    ── (3) build_transpiled_bundle: Crassus → Pompey → Caesar
   ▼
convert/handoff_probe.py ── (4) handoff_probe.json 패키징
   ▼
weights/<model>-cq<bits>/
```

### (2) 가중치 변환 (`convert/cli.py:408-611`)
1. `AutoConfig`(실패 시 raw `config.json`) → `detect_family` → `adapter.load_processor` → 모델 클래스 로드(`torch_dtype=float16, low_cpu_mem_usage=True`, 폴백 `AutoModel`). Parakeet `.nemo` 처리. HF 캐시는 `HF_HUB_CACHE`/`HF_HOME`.
2. 원시 텐서 로드: `model.safetensors`, 샤드 index, 로컬 `*.safetensors`, `pytorch_model*.bin`. 작은 버퍼 추가.
3. 아키텍처별 이름 정규화: `adapter.normalize_state_dict`, `name_tensor`, `expand_tensor`(MoE expert 분리 등), `transform_tensor`.
4. `config.txt`, `hf_config.json` 작성, HF 토크나이저/프로세서 파일 복사(`export/files.py`).
5. 토크나이저 export: `vocab.txt`, `merges.txt`, `special_tokens.json`, `tokenizer_config.txt`(`cactus_adapters/tokenizer.py`).
6. 선택적 캘리브레이션: GPTQ Hessian 수집(`calibration/hessian.py`), `--hessian-cache-in/out`.
7. 텐서별 정책(`policy_for_tensor`: convert / fallback / ignore) → `quantize_hadamard` 또는 `quantize_orthogonal` → `write_cq_tensor`; 폴백은 `save_tensor_with_header`로 FP16/INT8.
8. 리포트: `conversion_manifest.json`, `weights_manifest.json`, `conversion_summary.json`.

지원 패밀리 레지스트리: `SUPPORTED_FAMILIES = {auto, gemma3, gemma4, qwen, lfm2, whisper, parakeet, parakeet_tdt, moonshine, nomic, needle, generic}`(`model_adapters/detection.py:6`), 어댑터 클래스 `Gemma4Adapter, QwenAdapter, WhisperAdapter, ParakeetAdapter, NomicAdapter, ParakeetTDTAdapter, Lfm2Adapter, Gemma3Adapter` + 일반 `FamilyAdapter`(`adapters.py:874-887`).

### (3) 트랜스파일 (`cli/transpiler.py:49-165`)
- **Crassus (Converter + ModelProfiles)**: 소스 모델을 로드하고 대표 입력을 만들어 `torch.export.export(..., strict=False).run_decompositions()`(`Converter/models.py:446`)로 캡처, 추론 모드(`prefill_with_cache`, `decode_with_cache` 기본; 또는 `prefill_no_cache`)마다 `LayerMap` JSON(`transpiler_ir/output_<mode>.json`)을 쓴다. `ModelProfiles/`가 모달리티, 캐시 스타일·정밀도, 융합 그룹, 컴포넌트 정의와 실행 경로, 프롬프트/미디어 계약을 선언한다.
- **Pompey (IR + Fusions)**: IR 단순화와 패턴 융합 → `output_<mode>_simplified.json`.
- **Caesar (Generator + RuntimePlan)**: 각 컴포넌트를 `components/<name>/*.cactus`로 하강하고 `RuntimePlan.write`가 `components/manifest.json`과 `runtime_plan.json`을 만든다(`Generator/generate.py:62-95`, `RuntimePlan/models.py:579-586`).
- 세 단계의 버전은 `Converter/version.py`, `IR/version.py`, `Generator/version.py`에 있고 CLI가 "Running converter (Crassus vX.Y.Z)" 식으로 출력한다.

`MODEL_ID_MAP`(`ModelProfiles/profiles.py:583-594`)에 등록된 최적화 프로필: `google/gemma-4-E2B(-it)`, `openai/whisper-tiny/-small`, `nvidia/parakeet-tdt-0.6b-v3/-v2`, `LiquidAI/LFM2-VL-450M/-3B`, `Qwen/Qwen2.5-0.5B`, `LiquidAI/LFM2.5-8B-A1B`(MoE). 미등록 모델은 `--task/--modalities/--cache-style/--fusion-groups`로 만든 일반 프로필(`generic_text`, `generic_vlm`, `generic_speech_seq2seq`)을 쓴다. 일반 causal-LM 오디오는 "not supported yet".

설계 규칙(`CONTRIBUTING.md`): 모델별 정책은 `ModelProfiles/` 또는 좁은 구조 매처에만 두고, **공유 변환/IR/융합/하강 코드는 HF 리포 이름 부분 문자열로 분기하면 안 된다.** 등록된 모델 id는 프로필 전체를 쓰며 일반 CLI 플래그가 이를 덮어쓰지 않는다.

---

## 2. 구 파이프라인의 6단계 (`docs/cactus_transpiler.md`, 개념은 신 파이프라인에도 그대로 유효)

1. **모델 정규화**: 태스크 어댑터로 감싸 안정된 인터페이스 노출(`causal_lm_logits`, `multimodal_causal_lm_logits`, `encoder_hidden_states`, `ctc_logits`, `seq2seq_transcription`, `tdt_transcription`). `use_cache=False`, `return_dict=False` 주입.
2. **`torch.export`**: 모든 연산이 ATen 호출(`aten.linear.default` 등)인 FX 그래프. 이후로는 **레이어 이름이 아니라 ATen op 패턴**으로만 동작한다.
3. **IR 임포트**: FX → `IRGraph`(`IRNode`/`IRValue`). 상수는 `weights_manifest.json`을 통해 변환된 가중치 파일에 바인딩된다.
4. **IR 정규화**: 이름 정규화, transpose/movedim → permute, 상수 접기, ones/zeros/arange 구체화, no-op 제거, FP16 캐스트 삽입, DCE.
5. **패턴 융합**: `variance→rsqrt→mul→mul` → `rms_norm`, cos/sin 인터리브 → `rope`, `Q@Kᵀ→scale→mask→softmax→@V` → `attention`, `gate*up→act→down` → `dense_mlp_tq_fused`, `conv→bn→act` → `conv_module`, 게이트 4개 → `lstm_cell`. `--no-fuse-rms-norm/--no-fuse-rope/--no-fuse-attention`으로 끌 수 있다.
6. **CactusGraph 하강**: `IRNode` → 그래프 op. 상수는 `graph.mmap_weights(path)`(대부분), 인라인 스칼라, 또는 그래프에 박힌 소형 텐서. **하강은 Python이 ctypes로 `graph_ffi.cpp`의 빌더를 호출해 실제 `CactusGraph`를 만든 뒤 `save()`하는 것**이다(07장 §6).

멀티모달은 `vision_encoder / audio_encoder / lm_encoder / decoder`로 컴포넌트를 분리해 각각 캡처·최적화한다(torch.export 부담을 줄이고 런타임에서 개별 로드/언로드 가능). 디코더 그래프는 동적 배치 축과 KV 슬롯 1개로 방출되며 런타임이 `Model::set_decode_slots(N)`으로 슬롯 풀을 조정한다.

---

## 3. 번들 레이아웃 (`weights/<model>-cq<bits>/`)

| 그룹 | 파일 |
|---|---|
| 변환기 | `config.txt`, `hf_config.json`, HF 토크나이저/프로세서 JSON, `vocab.txt`, `merges.txt`, `special_tokens.json`, `tokenizer_config.txt`, 텐서당 `*.weights`, `conversion_manifest.json`, `weights_manifest.json`, `conversion_summary.json`, `hessian_metadata.json`(+`hessians.safetensors`), `handoff_probe.json` |
| 트랜스파일러 | `transpiler_ir/output_{prefill,decode}_*.json`, `components/<name>/*.cactus`, `components/manifest.json`, `runtime_plan.json` |
| 다운로드 | `.cactus_cq_download.json` |

"유효한 v2 번들" = `config.txt` + (`components/manifest.json` 또는 `runtime_plan.json`)(`cli/common.py:98-104`). `runtime_plan.json`만 있으면 manifest를 생성한다(`cli/model.py:104-158`).

---

## 4. `cactus run`이 번들을 찾는 순서 (`cli/model.py:161-210`)
1. 로컬 경로(`components/manifest.json` 또는 `runtime_plan.json` 포함)
2. 캐시 디렉터리 `weights/<name-lower>-cq<bits>/`(체크아웃) 또는 `~/.cache/cactus/weights/`
3. HF `Cactus-Compute/<name>`에서 **런타임 버전 이하의 최신 태그**(`resolve_weight_revision`, `cli/utils.py:144-158`; `docs/compatibility.md`의 규칙)의 `*-cq<bits>.{zip,tar.gz}` 다운로드 → SHA256 검증 → 안전 추출(zip-slip/symlink 검사) → `validate_extracted_bundle`(config/vocab/manifest/weights/토크나이저 사이드카 필수)
4. 실패 시 로컬 빌드(`ensure_bundle`: CQ 변환 + 트랜스파일). 균일 1~4bit만 가능.

---

## 5. NPU 아티팩트 방출 (`python/cactus/transpile/npu/`)
`python -m cactus.transpile.hf_model --npu`가 `run_encoder_pipeline`으로 `audio_encoder`/`vision_encoder`/`source_encoder`를 CoreML `.mlpackage`로 만든다(같은 어댑터 모듈, 보조 입력은 상수로 베이크, `coremltools.convert(mlprogram, FLOAT16, iOS17+)`, 기본 양자화 오디오 int8/비전 fp16, coremltools 9.0 패치 모음). manifest에 `npu_*_encoder` 경로를 남기지만 **런타임이 이를 로드하는 코드 경로는 v2.2.1에서 비활성**이다(08장 §5, 11장 §4).

---

## 6. 새 모델 패밀리를 추가하려면
1. `convert/model_adapters/`: 패밀리 감지(`detection.py`), 텐서 이름 정규화·확장 어댑터(`adapters.py`), 정책 예외(`policy.py`), 토크나이저 export 확인.
2. `transpiler/ModelProfiles/`: 프로필(모달리티, 캐시 스타일, 융합 그룹, 컴포넌트·경로 계약)과 `MODEL_ID_MAP` 등록. 필요하면 `Converter/overrides.py`에 export 호환 패치.
3. 새 연산 패턴이 있으면 `transpiler/Fusions/`에 매처 추가, 대응 op이 그래프에 없으면 `cactus-graph`(빌더 + `ops_*.cpp` + `param_io.cpp` 스키마 + `dispatch_flat`)와 `cactus-kernels`에 커널 추가.
4. 엔진 쪽은 보통 손댈 필요가 없다. 단 새 채팅 템플릿(`tokenizer.cpp`), 툴콜 포맷(`chat_tools.h`), `config.txt` 필수 키(`model.cpp:5821-5837` 같은 검증)가 필요하면 추가한다.
5. `cactus convert <id>` → `transpiler_ir/*.json` diff로 융합 확인 → `cactus run` → `cactus test --component engine --model <id>`.

---

## 7. 읽기 순서 제안
1. `python/cactus/cli/convert.py`(200줄) → `cli/model.py`의 `ensure_weights`/`ensure_bundle`
2. `convert/cli.py:408-611` 한 번 통독 → `policy.py`
3. `transpiler/CONTRIBUTING.md` → `Converter/convert.py` → `ModelProfiles/profiles.py`의 LFM2-VL 프로필
4. 작은 모델(`Qwen/Qwen2.5-0.5B`)을 변환하면서 `transpiler_ir/output_decode_with_cache.json`과 `_simplified.json`을 열어 노드 수와 op 종류 변화를 센다
5. `Generator/lowerings/`에서 `rms_norm`이 어떤 FFI 호출로 바뀌는지 추적하고, 그 FFI가 `graph_ffi.cpp`의 어느 함수인지 확인한다
