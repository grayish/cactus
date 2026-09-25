# 11. 가속기 백엔드와 튜닝: CPU, Metal GPU, ANE, 그리고 없는 것들

> 근거: `cactus-kernels/{CMakeLists.txt, metal_backend.h, src/metal_backend.mm, src/metal_backend_stub.cpp, src/cactus_kernels.metal, src/threading.h, src/matmul.cpp}`,
> `cactus-graph/src/{execute.cpp, metal_plan.cpp, metal_runtime.cpp, ops_cache.cpp}`, `cactus-engine/src/{init.cpp, npu_ane.mm, model_npu.cpp}`, `python/cactus/transpile/npu/`, `blog/lfm2.5_350m.md`

---

## 0. 결론부터

| 백엔드 | 상태 (v2.2.1) | 플랫폼 | 무엇을 실행하나 |
|---|---|---|---|
| **CPU (ARM NEON + DOTPROD)** | 주력, 유일한 보편 경로 | Apple, Android, Linux aarch64 | 모든 op. Apple에서는 큰 형상에 Accelerate(cblas/vDSP) 위임 |
| **Apple Metal GPU** | 활성. Apple에서 기본 백엔드 | macOS, iOS(시뮬레이터 포함) | 노드 단위로 GPU 인코딩, 실패 시 CPU 폴백. 120개 컴퓨트 커널 + 18개 융합 규칙 |
| **Apple Neural Engine (CoreML)** | **인프라만 존재, 런타임 호출 경로 비활성** | Apple | 설계상 오디오/비전/소스 인코더 `.mlpackage`만. 텍스트 디코더는 대상 아님 |
| Android GPU (Vulkan/OpenCL) | 없음 | – | – |
| Android NPU (NNAPI, QNN/Hexagon, MediaTek APU, Exynos NPU) | 없음 | – | – |
| NVIDIA CUDA / AMD ROCm / Intel XPU·OpenVINO / WebGPU / TensorRT | 없음 | – | – |
| x86 SIMD (SSE/AVX), ARM SVE/SME | 없음 | – | – |

사용자가 물은 "NPU, XPU 등 가속기 백엔드"에 대해 정직하게 말하면: **Cactus에는 Intel XPU 백엔드가 없고, NPU는 Apple ANE 한 종류만 코드가 있으며 그마저 현재 릴리스에서는 연결이 끊겨 있다.** 이 장은 (1) 실제로 존재하는 세 경로의 구조와 선택 로직, (2) 각 경로의 튜닝 노브, (3) 왜 다른 가속기가 없는지와 추가하려면 어디를 건드려야 하는지를 다룬다.

---

## 1. 백엔드 선택 로직 (끝에서 끝까지)

```cpp
// cactus-graph/src/execute.cpp:17-33
static int g_selected_backend = -1;
ComputeBackend cactus_default_backend() {
    if (g_selected_backend >= 0) return static_cast<ComputeBackend>(g_selected_backend);
    if (cactus_metal_available()) return ComputeBackend::METAL;   // "auto"
    return ComputeBackend::CPU;
}
int cactus_backend_select(const char* backend);   // "cpu" | "metal"(가능할 때만) | 그 외 -1
```
- enum은 `ComputeBackend { CPU = 0, METAL = 1 }` 두 값뿐이다(`cactus_graph.h:86`). `"npu"` 값은 없다.
- **엔진 C API**: `cactus_set_backend(const char*)`(`cactus_engine.h:37`) → `cactus_backend_select`(`init.cpp:354`). `cactus_init`이나 옵션 JSON에는 backend 필드가 없다. **전역 설정**이므로 모델 로드 전에 호출해야 한다.
- **노드 단위 고정**: `cactus_graph_set_node_backend(graph, node, 0|1)`(`graph_ffi.cpp:632-653`), Python은 모든 op의 `backend=` 인자(`Graph.CPU`/`Graph.METAL`). 노드 핀이 전역 기본값을 이긴다(`test_metal_parity.cpp`의 `parity_per_op_pins`).
- **CLI**: `--backend cpu|metal`(기본 None=auto). `run`은 네이티브 `bin/run`에 전달해 `cactus_set_backend` 호출, 실패 시 "Metal not available; using CPU"(`tests/run.cpp:863-904`). `serve`는 바인딩 직접 호출. `test/benchmark`는 `CACTUS_TEST_BACKEND` env로 테스트 하네스에 전달.
- **환경변수로 백엔드를 고르는 기능은 라이브러리에 없다.** `CACTUS_TEST_BACKEND`는 테스트 전용.
- **"auto"의 실제 의미**: `__APPLE__` 빌드(macOS·iOS 모두)에서 `MTLCreateSystemDefaultDevice()`가 성공하고 핵심 파이프라인 ~34개가 컴파일되면 Metal. iOS도 제외되지 않는다(`apple/CMakeLists.txt:29-30`이 Metal/MPS 링크). Android/Linux는 스텁이라 항상 CPU.
- 직렬화된 그래프에는 backend가 저장되지 않으므로(`param_io.cpp:394`) 로드 시점의 전역 기본값이 노드 backend가 된다.
- 그래프 실행 시 Metal 모드 진입 조건: Metal 가능 + 비INPUT 노드 중 METAL 태그가 하나라도 있음(`execute.cpp:1569-1575`). 예외: 100노드 미만 + LSTM 포함 → CPU(1576-1584); 프로파일/트레이스 env가 켜지면 Metal 세션을 열지 않음(1505-1515, 1684).

---

## 2. CPU 백엔드

### 2.1 ISA와 빌드
- 플래그는 플랫폼 구분 없이 항상 `-march=armv8.2-a+fp16+simd+dotprod+i8mm`(`cactus-kernels/CMakeLists.txt:36`).
- 실제 사용 명령: NEON FP16 FMA(`vfmaq_f16`), **SDOT**(`vdotq_laneq_s32`, `vdotq_s32`), 테이블 룩업(`vqtbl1q/2q`), 비시간적 저장(`stnp`).
- **i8mm(SMMLA)은 플래그에만 있고 사용되지 않는다.** `cpu_has_i8mm()`(`threading.h:51-76`)에 호출자가 없다. 블로그(`lfm2.5_350m.md`)가 설명하는 "I8MM → DOTPROD 폴백"은 현 코드에서는 "DOTPROD 필수"로 봐야 한다. `__ARM_FEATURE_DOTPROD` 없을 때의 에뮬레이션(`matmul.cpp:27-71`)은 인터리브 커널과 하이브리드 어텐션이 `vdotq_*`를 직접 호출하므로 불완전하다.
- SVE/SME, x86 없음. 스칼라 전용 빌드 없음.

### 2.2 Apple에서의 Accelerate 위임 (GPU가 아닌 CPU 경로)
| 커널 | 조건 | 위치 |
|---|---|---|
| `cactus_matmul_f16` | `M ≥ 4 && K ≥ 256` → FP32 변환 후 `cblas_sgemm` | `matmul.cpp:18-19, 637-661` |
| `cactus_attention_f16` | 마스크 있음 + `seq_len ≥ 64` + window 0 + 차원 %8 + cap 없음 (env `CACTUS_DISABLE_TILED_MASKED_ATTENTION`으로 해제); 마스크 없음 + `seq_len ≥ 64` | `attention.cpp:17-242, 266-277, 435` |
| conv1d/conv2d | `K ≥ 32 && L ≥ 128`, im2col + sgemm | `conv.cpp:17-21`, `conv2d.cpp:10-51` |
| LSTM | `cblas_sgemv` | `lstm.cpp:285-351` |
| FFT | `vDSP_fft_zrip` | `dsp.cpp:146-202` |

### 2.3 스레딩 정책 (06장 §3 요약)
- 풀: 조건변수 기반, `CACTUS_NUM_THREADS`(≤16). Android는 성능 코어 수로 상한 + affinity 고정 + 호출 스레드를 가장 한가한 성능 코어에 고정(`prepare_current_thread_for_cactus_work`, 엔진이 생성 진입 시 호출).
- GEMV(M=1) 스레드: Android 1, iOS ≤3(N블록 ≥512), macOS/Linux ≤5(N블록 ≥512). GEMM(M>1): Android 전체, iOS ≤2, 그 외 ≤4 → 이것이 "단일 코어 디코드, 멀티 코어 프리필"의 구현이다.
- matmul의 2단계 스핀 드라이버는 cv sleep 비용(5~10µs)을 피하려고 busy-wait를 쓴다(`matmul.cpp:348-378`).

### 2.4 CPU 튜닝 노브
**환경변수**

| 변수 | 효과 | 위치 |
|---|---|---|
| `CACTUS_NUM_THREADS` | 풀 크기(≤16). Android는 성능 코어 수 상한 | `threading.h:430` |
| `CACTUS_GEMV_SB_PER_THREAD` | 인터리브 GEMV: 스레드당 슈퍼블록(64행) 수, 기본 4 → 스레드 예산 결정 | `matmul.cpp:326-333` |
| `CACTUS_INTERLEAVED_GEMV_THREADS` | 인터리브 GEMV(4bit, pair/triple) 스레드 상한 | `matmul.cpp:335-346` |
| `CACTUS_DISABLE_TILED_MASKED_ATTENTION` | Apple Accelerate 마스크 어텐션 끄기 | `attention.cpp:435` |
| `CACTUS_GATED_DELTANET_CHUNK_SIZE` / `_PREFILL_OLD` | DeltaNet 프리필 청크(기본 64→16 클램프) / 순차 경로 | `fused.cpp:242, 834` |
| `CACTUS_KV_CACHE_FP16` | KV 캐시를 INT8 대신 FP16으로 (Metal 미가속) | `builder.cpp:12` |

런타임 API: `CactusThreading::set_gemm_threads(n)/reset_gemm_threads()`(`threading.h:626-637`, 현재 호출자 없음).

**소스 상수 (새 칩에 맞출 때 우선 볼 것)**

| 상수 | 값 | 의미 | 위치 |
|---|---|---|---|
| `Thresholds::*` `{gate, per_thread}` | ATTENTION {64,32}/{32,16}, ELEMENT_WISE {5000,2500}, AXIS_REDUCE {1000,500}, ALL_REDUCE {10000,5000}, SCALAR_BASIC {30000,15000}/{5000,2500}, SCALAR_EXPENSIVE {10000,5000}/{2500,1250} (Android/그 외) | 병렬화 시작 임계와 스레드당 작업량 | `threading.h:575-591` |
| `GEMV_MIN_N_BLOCKS` | iOS 512, 그 외 256 | GEMV 멀티스레드 시작 N블록 | `threading.h:602-622` |
| GEMV/GEMM 스레드 상한 | Android 1/전체, iOS 3/2, 기타 5/4 | | 같은 곳 |
| `ACCELERATE_M_THRESHOLD`, `_K_THRESHOLD` | 4, 256 | cblas 위임 시작 | `matmul.cpp:18-19` |
| 밀집 `TILE_M/TILE_N` | 4/4 | FP16 matmul 레지스터 타일 | `matmul.cpp:552-553` |
| SDOT GEMV `INT8_TILE_N` | 16 (16블록/스레드) | | `matmul.cpp:719, 782` |
| LUT GEMV `TILE_N` | 4/3/1bit 12, 2bit 16 | | `matmul.cpp:906, 981, 1106, 1232` |
| 프리필 INT8 GEMM `TILE_M` | 8; 인터리브 GEMM 64블록/스레드 | | `matmul.cpp:1764, 2398, 2469` |
| `group_size` 제한 | ≤256, 2의 거듭제곱; SDOT 경로는 `gs%32==0` | | `matmul.cpp:516-524, 920, 1581` |
| 어텐션 `BLOCK_SIZE` | Accelerate 64, NEON 32, 하이브리드 디코드 64, `SINK_SIZE` 4, `MAX_HEAD_DIM` 512 | | `attention.cpp:36, 262, 460`, `attention_hybrid.cpp:33-37, 374, 435` |
| `MAX_WORKERS`, `MAX_CPUS`, 성능 코어 컷오프 | 16, 16, 0.70 | | `threading.h:388, 206, 237` |
| `ANDROID_DYNAMIC_CHUNK_MULTIPLIER` | 16 | Android 동적 밸런싱 과분해 배수 | `threading.h:181` |
| `STREAMING_STORE_THRESHOLD` | 32768 원소 | 비시간적 저장 시작 | `threading.h:34` |
| conv 단일 스레드 컷 | `total_compute < 100000` | | `conv.cpp:217`, `conv2d.cpp:126` |

**튜닝 방법론(CPU)**
1. 이론 상한을 먼저 계산한다: 디코드 tok/s ≤ 메모리 대역폭 ÷ (모델 바이트/토큰). 블로그 예시: 355MB 모델, 60GB/s → 169 tok/s, 실측 140(83%).
2. `cactus benchmark`(`test_benchmark.cpp`)로 프리필/디코드를 분리해 측정한다. 디코드가 상한의 70% 미만이면 스레딩·프리페치·레이아웃, 프리필이 낮으면 타일·Accelerate 임계를 본다.
3. env 노브로 가설 검증 → 상수 수정 → `cactus test --component kernels`로 정확성 → 벤치 첨부 PR(CONTRIBUTING.md 요구).
4. 새 Android SoC: `CoreTopology::detect`가 성능 코어를 올바르게 잡는지(`cpu_capacity` 파일 존재 여부), 0.70 컷오프가 중간 코어(예: X4/A720/A520 3계층)를 어떻게 분류하는지 확인한다.

---

## 3. Apple Metal GPU 백엔드

### 3.1 구조
- **컨텍스트**: 프로세스당 하나의 `MetalCtx`(함수 지역 static, `metal_backend.mm:146`). `MTLCreateSystemDefaultDevice` → 커맨드 큐 1개 → **임베디드 MSL 소스 문자열을 런타임 컴파일**(`newLibraryWithSource`, :80; `cmake/embed_msl.cmake`가 `.metal`을 `kCactusMSL[]` C 배열로 변환). `.metallib` 오프라인 컴파일 없음. 120개 파이프라인 생성(:75-143), GQA 디코드 변형 `attn_decode_i8_gqa_{b}_{u}`는 루프로 생성.
- **메모리는 전부 통합·zero-copy**: `MTLResourceStorageModeShared`만 사용. `owned_shared`는 16KiB 정렬 anonymous mmap을 `newBufferWithBytesNoCopy`로 감싸고(:148-161), `wrapHostPtr`는 **임의 호스트 포인터(mmap 가중치, std::vector)를 페이지 정렬해 no-copy 버퍼로 래핑**하고 캐시한다(:305-354; macOS에서는 `mach_vm_region`으로 512MiB 이하 RW 영역 전체로 확장). `bufForPtrOff`는 공유 할당 → 래핑 → 최후에 memcpy 순(:356-373). MoE expert는 `vm_remap`으로 하나의 가상 아레나로 이어 붙인다(:954-1004). **`MTLHeap`, private storage, residency set, `MTLEvent/Fence`는 없다.**
- **배칭·동기화**: 전역 커맨드 버퍼 + 컴퓨트 인코더 하나에 디스패치를 누적, `CACTUS_FLUSH_CADENCE`(기본 48) op마다 대기 없이 commit(`session_flush`), CPU가 GPU 데이터를 만질 때만 `waitUntilCompleted`(`session_sync`, :406-421). completion handler는 오류 로깅용뿐. MPS 호출 전후로 인코더를 끊는다.
- **가중치 상주**(`resident(W)`, :220-265): CQ 컴포넌트(코드북·부호·순열·스케일·패킹·norm)를 zero-copy 바인딩하고 FNV 해시로 캐시. 패킹 8B/norm 2B 정렬 필수. 2bit는 int8 코드북 사본 추가 생성.
- **풀**: 16KiB 버킷 free list, 32MiB 슬랩 할당기(256B 정렬, 4MiB 초과는 개별 할당), `session_sync`에서 회수.

### 3.2 GPU 커널 목록 (`cactus_kernels.metal`, 120 엔트리)
| 영역 | 커널 |
|---|---|
| CQ 활성화 변환 | `cq4_transform`, `_simd`, `_batch`, `_m` — 128포인트 Hadamard를 `simd_shuffle_xor` 버터플라이로 레지스터 안에서 수행(`cq4_hada128`, :20-32) |
| CQ GEMV(디코드) | `cq4_gemv`, `cq4_gemv_mr/mr4`(매크로 NR=2/4), 융합 `cq4_transform_gemv`, `cq4_swiglu_transform`, `cq4_gemv_cat`(최대 3행렬), 저비트 `cq_gemv_mr_lowbit(4)`, `cq_act_quant_i8` + `cq2_gemv_i8`(INT8 내적) |
| CQ GEMM(프리필) | `cq4_gemm_mma`, `cq4_gemm_dense_f16` — `simdgroup_matrix<half,8,8>` MMA, 타일 M32×N64×K32, 128 스레드 |
| LM head/임베딩 | `lmhead_rotate_wide`, `emb_ortho_wide/_m`, `emb_hadamard/_m` |
| MoE | `cq4_moe_transform(2)`, `cq4_moe_gemv_up`, `cq4_moe_gemv_down_acc` |
| 어텐션 | 밀집 `attn_f16`, `attn_f16_d64`, `attn_flash_f16`(simdgroup MMA); INT8 캐시 디코드 `attn_decode_i8`(split-K) + `attn_decode_combine`, `attn_decode_fused_i8`(RMSNorm+RoPE+KV append+어텐션), GQA 템플릿 12종; 프리필 `attn_prefill_i8`, `attn_prefill_mma2`(hd 512), `attn_prefill_mma_hd256` |
| KV 캐시 | `kv_append_i8(_m)`, `kv_append_ring_i8_m`, `kv_slide_save/restore(_m)`, `conv_cache_append_f16` |
| 정규화 | `rms_norm(_add/_add_scale/_add_rms/_scale/_simd)_f16`, `rms2_add_clip_f16`, `layer_norm`, `batchnorm`, `groupnorm` |
| 원소별 | `binary/scalar/unary_{f16,f32}`, `bcast_binary(_rows)_f16`, `clamp`, `glu`, `swiglu`, `softcap`, `cast_*`, `copy_bytes`, `elemwise_chain_f16`(최대 12단 융합) |
| 레이아웃 | `strided_copy(_rows)`, `strided_scatter(_rows)`, `transpose2d`, `concat2`, `gather_f16`, `gather_f32idx` |
| RoPE | `rope_full`, `rope_pair`, `rope_pair_rms` |
| 리덕션·샘플링 | `reduce_axis_{f16,f32}`, `cumsum`, `softmax_rows`, `softmax_topk`, `topk_rows`, `argmax_part/final/logits`, `adjust_logits` |
| conv/비전/오디오 | `conv1d_k3/gen/dw/nlc_dw`, `conv2d`, `maxpool1d`, `bilinear`, `rel_pos_bias` |
| 재귀 | `gated_deltanet_decode/prefill_f16` |

특이점: function constant 없음(템플릿·매크로로 특수화), atomics 없음, bfloat 없음. GPU에 **없는 것**: INT8 밀집 가중치 matmul(INT8 선형은 CPU 폴백, conv 가중치만 CPU에서 FP16 변환 후 캐시), **CQ1 경로**, FP16 KV 캐시 가속. CQ4는 `gs ≥128 && %128==0`, `N%4==0`, 인터리브 플래그 필수; CQ2/3은 `gs==128`, `N%16==0`(`.mm:707-720`).

### 3.3 그래프 레벨 배치: 파티셔닝이 아니라 "노드 단위 시도 + 폴백"
- 노드마다 (1) 융합 플랜의 앵커면 `cactus_metal_plan_encode`, (2) 아니면 `try_encode_metal`(op별 인코더, `execute.cpp:638-1403`), (3) 둘 다 실패하고 입력 중 GPU-live가 있으면 `session_sync` 후 CPU 커널(1862-1882). 강제 CPU 예외: ≤8원소 `PRECISION_CAST`(입력이 GPU에 없을 때), 입력이 GPU-live인 `EMBEDDING`, 인덱스 4096 초과 임베딩.
- 통합 메모리라 **폴백 비용은 복사가 아니라 동기화**다. `metal_live[]`가 GPU 산출 텐서를 추적해 CPU op이 소비할 때만 sync한다.
- FP32 원소별 "섬"은 FP16으로 retype 해 GPU에서 돌린다(`build_metal_retype_plan`, 1422-1492; 스칼라 |값| < 60000 조건).
- 디코드형 플랜은 선형 스캔 아레나(텐서당 8MiB, 총 256MiB 상한). 1500노드 이상 그래프는 256KiB 이상 출력에 transient 버퍼(256 인코드마다 회수·sync).
- **GPU 샘플링 꼬리**: LM head → softcap → `adjust_logits`(반복 패널티·억제) → `argmax`를 GPU에서 수행하고 float 3개(best, second, index)만 읽어 온다(`metal_runtime.cpp:76-114`, `model.cpp:2927, 3136`).
- **융합 임베딩 프롤로그**: 임베딩+PLE+프로젝션을 디코드 시작 시 GPU에서(`cactus_graph_metal_fold_prologue`, `model.cpp:1416-1429`).
- 실패 복구: 융합 클러스터 실패 → 앵커 금지 후 `execute()` 재귀 재실행; retype 실패 → retype 비활성 후 재실행; `MetalExecGuard` 소멸자가 sync, KV 길이 워드 복원, 세션 종료.

### 3.4 융합 플래너 규칙 (`metal_plan.cpp`, 그래프 서명 FNV 해시로 캐시)
| 규칙 | 패턴 | 인코딩 |
|---|---|---|
| 1 | 디코드 `ATTENTION_CACHED`(쿼리 1, `num_kv_heads==1`) + Q/K RMSNorm + RoPE + KV append | `attention_fused_i8` |
| 2 | 같은 입력을 공유하는 2~3개 M=1 CQ4 gs=128 matmul(QKV) | `transform_batch` + `gemv_cat` |
| 3 / 12 / 18 | `ADD_CLIPPED(RMS_NORM(x), res)` 변형, `RMS_NORM(ADD)` 다행, `ADD_CLIPPED(RMS(a),RMS(b))` | `rms_norm_add*`, `rms_norm_add_rows`, `rms2_add_clip` |
| 4 / 7 | GELU 게이트 MLP down-proj / gate→down | `swiglu_transform`+`gemv_precoded` / `transform_gemv`×2 |
| 5 | LM head softcap + orthogonal CQ | `quant_matmul_ortho` + softcap + GPU argmax 꼬리 |
| 8 / 9 / 10 | 연속 슬라이스의 N개 MATMUL / N개 CONV1D / 동일 TRANSPOSE의 CAT | `gemm_batch` / `conv1d_dw` / `strided_copy` |
| 11 | 2~12개 원소별 op 체인(사이드 입력 ≤3) | `elemwise_chain_f16` |
| 13 | `ADD(MATMUL(fp16 vec, W), bias)` | `gemv_bias` |
| 14 / 17 / 15 | RoPE rotate-half [+선행 RMSNorm, D≤1024] / `SCALAR_MULTIPLY(RMS_NORM)` | `rope_pair`, `rope_pair_rms`, `rms_norm_scale` |
| 16 | `TOPK(SCALAR_MULTIPLY(logits))`+SOFTMAX (MoE 라우터, k≤16, E≤4096) | `softmax_topk` |

### 3.5 KV 캐시와 프리필/디코드
- 기본 백엔드가 METAL이면 KV 버퍼를 `cactus_metal_alloc_shared`로 할당(`ops_cache.cpp:104-106`), 초기 256 엔트리, 2배 성장(성장 전 sync + 복사). 슬롯 헤더 64B(u64: `current_len, max_len, kv_heads, hdim, sink, num_slots`) + int8 데이터 + 32그룹 float 스케일. append는 GPU 커널(일반/슬라이딩/링). CPU가 인코드 시 길이 워드를 갱신하고 실행 전 값을 스냅샷해 중단 시 복원.
- matmul: M==1 → GEMV, M>1 또는 `cactus_graph_prefill_consistent()` → `cq4_transform_m` + `cq4_gemm_mma`. "prefill-consistent" 모드는 M==1도 M 경로로 강제해 수치 일관성을 맞춘다(`model.cpp:1408-1434`).
- 어텐션: `seq>1` → `attention_i8_prefill`(hd 512/256 + nqh 8 + nkv 1이면 MMA 특수화, 그 외 일반 커널 `maxsc ≤ 7936`); 디코드 → split-K `attention_i8` + combine 또는 규칙 1 융합.
- 매 `complete/embed/transcribe` 후 `cactus_metal_trim_prefill_cache()`(RAII `MetalTrimGuard`)로 MPS 행렬·free list·스코어/코드 버퍼 정리. 모델 로드 시 `prewarm_metal_quant_weights()`.

### 3.6 MPS 사용
- `gemm_f16`: `MPSMatrixMultiplication`(16B 정렬, `K%8==0 && N%8==0`), 아니면 `cq4_gemm_dense_f16`.
- 밀집 어텐션: `B==1, HQ==HKV`, 비인과, window/cap 없음, `S≥512 && S%8==0`이면 MPS. `S≥256`, D=64면 `attn_flash_f16`.
- MPS 커널 객체·행렬 캐시(4096 초과 시 비움).

### 3.7 정확성 검증 (`test_metal_parity.cpp`)
같은 케이스를 `cpu` → `metal`로 두 번 실행해 `|cpu-metal| > tol·max(1,|cpu|)`면 실패. 허용 오차: 기본 5e-2, softmax 5e-3, rms_norm/elemwise_chain 1e-2, flash/causal 어텐션 2e-2, 동등 비교 1e-3. 커버: 단항·이항·브로드캐스트·리덕션·cumsum·concat·gather·rope·pool·conv1d/2d·norm·softmax·융합 norm·원소별 체인·flash/causal 어텐션·**슬라이딩 윈도 링 KV 캐시**(ceiling 64, window 8, sink 2)·**캐시 성장**(600토큰, ceiling 2048)·노드 핀. CQ matmul은 `cactus-kernels/tests/test_matmul.cpp:606-676`에서 정확 일치로 검증.

### 3.8 GPU 튜닝 노브
| 노브 | 값 | 위치 |
|---|---|---|
| `CACTUS_FLUSH_CADENCE`(env) | 48 op마다 비대기 commit | `execute.cpp:1506, 1839, 1885` |
| transient 모드 임계 / 최소 크기 / 회수 주기 | 1500노드 / 256KiB / 256 인코드 | `execute.cpp:1685, 1723, 1752` |
| 소형 LSTM 그래프 CPU 강제 | `n < 100` | `execute.cpp:1576` |
| 소형 cast CPU / 임베딩 최대 인덱스 / retype 스칼라 한계 | ≤8 / 4096 / <60000 | `execute.cpp:1862, 1383, 1446` |
| 디코드 아레나 텐서/총 상한, 정렬 | 8MiB / 256MiB / 256B | `metal_plan.cpp:1424, 1441, 1423` |
| 원소별 체인 길이/사이드 | 12 / 3 | `metal_plan.cpp:1248, 1315` |
| MoE top-k / 라우터 E | ≤16 / ≤4096 | `metal_plan.cpp:1197, 1206` |
| 페이지/버킷 단위, 슬랩, pooled 컷, 정렬 | 16KiB, 32MiB, 4MiB, 256B | `.mm:149, 276, 479, 471, 472` |
| macOS 래핑 영역 최대 | 512MiB | `.mm:330` |
| GEMV 스레드그룹 | `ROWS=8` simdgroup×32 = 256 스레드; mr NR=2 → 16행/TG, mr4 → 32행/TG; `CQ4_VPL=16` | `.metal:3, 172`, `.mm:278, 791-794` |
| 변환 TG 크기 | 32(gs=128) 또는 `min(gs,1024)` | `.mm:755-761` |
| 융합 transform-GEMV 적합 조건 | `K*2 + 8*128*4 + 64 ≤ maxThreadgroupMemoryLength` | `.mm:844-846` |
| 프리필 GEMM 타일 | M32×N64, 128 스레드, K 32 | `.metal:648-658` |
| 디코드 어텐션 split-K | 비GQA T=256, `nwg=clamp(R/24,1,32)`; GQA T=32, `nwg=min(ceil(256/units), R/32, 256)`, 헤드 그룹 4/3/2(hd 256이면 2) | `.mm:1537-1556, 1600` |
| 프리필 어텐션 | T=128, `maxsc` 기본 256, >7936이면 포기 | `.mm:1643-1650` |
| flash / MPS 어텐션 자격 | `S≥256`, D=64 / `S≥512`, `S%8==0` | `.mm:1892-1915` |
| argmax 분할 | `V≥32768` → 128 파티션×256 / 그 외 1TG×1024 | `.mm:619-655` |
| norm/원소별 TG | RMS 256(simd 버전 128, 4행), layer/softmax 1024(dim≥1024) 또는 256; 원소별 256스레드×4원소 | `.mm:562-575, 524, 534, 1753, 1767` |
| KV 양자화 그룹 / 초기 용량 | 32 / 256(2배 성장) | `execute.cpp:1278`, `ops_cache.cpp:103` |

CPU env 노브(`CACTUS_NUM_THREADS` 등)는 Metal 동작에 영향이 없다.

**튜닝 방법론(GPU)**
1. `cactus benchmark --backend cpu` vs `metal`로 프리필·디코드를 나눠 비교한다. 이득은 프리필(MMA GEMM)과 긴 컨텍스트 어텐션에서 크고, 짧은 디코드는 커맨드 제출 오버헤드가 지배할 수 있다.
2. `CACTUS_FLUSH_CADENCE`를 조정해 제출 지연 vs GPU 유휴의 균형을 본다.
3. `CACTUS_TRACE_EXECUTE`/프로파일 env는 Metal 세션을 끄므로 GPU 프로파일링은 Xcode Instruments(Metal System Trace)를 쓴다.
4. 폴백이 얼마나 일어나는지 확인하려면 `try_encode_metal`이 false를 반환하는 op을 로그로 세어 본다(정렬 조건, `gs`/`N` 배수 조건이 흔한 원인).
5. 새 융합 규칙을 추가하면 `test_metal_parity.cpp`에 케이스를 추가하고 CQ 경로는 정확 일치 테스트를 유지한다.

---

## 4. Apple Neural Engine (CoreML) 경로

### 4.1 무엇이 있나
- **Python 방출기**(`python/cactus/transpile/npu/`): `run_encoder_pipeline`이 `audio_encoder`/`vision_encoder`/`source_encoder` 컴포넌트를 CPU 그래프와 같은 어댑터 모듈로 `torch.export` → `coremltools.convert(mlprogram, FLOAT16, iOS17/18)` → `.mlpackage`. 보조 입력(마스크·위치 id)은 상수로 베이크(ANE는 정적 단일 입력 서명 요구). 기본 양자화 오디오 int8(선형 대칭 채널별), 비전 fp16(int4는 Gemma 4 비전 출력을 눈에 띄게 손상). `coremltools_patches.py`가 coremltools 9.0의 op 공백(`__and__`, `new_ones`, `one_hot`, `unfold`, layer_norm eps dtype 등)을 패치.
- **범위 결정**(`npu/README.md`): 텍스트 디코더 프리필은 의도적으로 제외. 이전에 있던 `prefill.py`와 C++ `NPUPrefill`은 제거됐다. 이유: 2B 디코더의 coremltools 변환이 15~20GB 피크 메모리, PR #659 이후 CPU 청크 프리필이 충분히 빠름.
- **검증 상태(README 표)**: Parakeet TDT 오디오 인코더 "end-to-end 검증, M시리즈에서 warm 4.5× 빠름(~155ms vs ~693ms TTFT)", LFM-VL 비전 인코더 "검증, 인코더 그래프 단독 ~4×", Gemma 4 오디오/비전 "방출 검증(int8, cos_sim 0.99+)".
- **런타임**(`npu_ane.mm`, Apple 전용, `CACTUS_HAS_ANE=__APPLE__`): `.mlpackage`를 런타임에 `compileModelAtURL`로 컴파일해 `.mlmodelc` 캐시(mtime 비교, `CACTUS_ANE_FORCE_RECOMPILE`). compute units 기본 `CPUAndNeuralEngine`이나 경로에 `audio_encoder`/`vision_encoder`가 있으면 **`CPUAndGPU`가 기본**(:96-101); 오버라이드는 manifest 힌트 → `CACTUS_ANE_COMPUTE_UNITS` → `CACTUS_ANE_ENCODER_COMPUTE_UNITS` → `CACTUS_ANE_AUDIO_COMPUTE_UNITS`/`CACTUS_ANE_VISION_COMPUTE_UNITS`(값 `all / cpu_and_ne / cpu_and_gpu / cpu_only`). I/O는 fp16 `MLMultiArray`(stride 인식 복사), 단일 입력 `encode()`와 다중 입력 `encode_multimodal_input()`.
- **엔진 접착**(`model_npu.cpp`): `load_npu_{audio,vision,source}_encoder`, `audio/vision/source_encode_via_npu`(출력 형상 불일치면 false → CPU 그래프), LFM2-VL 타일 헬퍼.

### 4.2 왜 "비활성"이라고 하는가
- `load_npu_audio_encoder / load_npu_vision_encoder / load_npu_source_encoder`는 선언(`engine.h:796-800`)과 정의만 있고 **호출자가 없다**. manifest의 `npu_*` 경로는 파싱(`model.cpp:1031-1042`)만 된다. 따라서 `has_npu_vision_encoder()`는 항상 false, `vision_encode_via_npu` 호출 지점(`model.cpp:2570, 4060`)은 도달하지 않는다.
- 공개 `cactus convert` CLI에 `--npu` 플래그가 없다(`python -m cactus.transpile.hf_model --npu`로만 방출 가능).
- 히스토리: c4c9cfa 추가 → 68cb6a9 "Remove coreml (#754)" 제거 → 338b5a5 재추가 → c90a1af 병합 후 부재. 즉 ANE는 "언제든 다시 연결할 수 있게 남겨 둔 인프라"다.

### 4.3 다시 켜려면 (학습 과제)
1. `Model::init`의 manifest 처리 직후 `npu_*_mlpackage_`가 비어 있지 않으면 `load_npu_*_encoder(full_path, compute_units)`를 호출하는 3~6줄을 복원한다(커밋 c4c9cfa diff 참고).
2. `cactus convert`에 `--npu` 계열 플래그를 노출해 `run_encoder_pipeline`을 호출한다.
3. `CACTUS_ANE_AUDIO_COMPUTE_UNITS=ALL`로 Parakeet TTFT를 CPU 그래프와 비교한다.
4. 텍스트 디코더를 ANE에 올리고 싶다면 README의 세 가지 제약(변환 메모리, 정적 서명, 양자화 스킴 제한)을 먼저 해결해야 하며, 이는 블로그가 "NPU를 1급 타깃으로 삼지 않는 이유"로 든 것과 같다.

---

## 5. 왜 다른 가속기가 없는가, 그리고 추가하려면

### 5.1 프로젝트의 공식 입장 (`blog/lfm2.5_350m.md:53-62`)
- 디코드는 대역폭 바운드라 GPU FLOPS가 남아돈다. 모바일 GPU는 UI 파이프라인과 경쟁하고 배터리를 더 쓴다.
- Android GPU는 Vulkan/OpenCL 두 셰이더 경로를 벤더별 특성과 함께 유지해야 한다.
- ANE/Hexagon(QNN)은 벤더 포맷·제한된 양자화 스킴·실행 스케줄 통제 불가라는 제약이 있고, 저가 기기에는 아예 없다.
- "CPU는 보편 타깃이며 ARM이 DOTPROD → I8MM → SME2로 CPU 자체에 행렬 효율을 흡수하고 있다"는 데 베팅한다.

### 5.2 코드 근거
리포 전체 grep 결과: NNAPI, QNN, Hexagon, Vulkan, OpenCL, ROCm, HIP, OpenVINO, XPU, WebGPU, TensorRT, ExecuTorch, LiteRT, TFLite 관련 코드 0건(블로그의 언급과 Python 변환 도구의 `torch.cuda` 사용만 존재).

### 5.3 새 백엔드를 추가한다면 어디를 건드리나
현 구조는 "백엔드 추상 인터페이스"가 아니라 **Metal 전용 인코더 함수군**이므로, 두 번째 GPU 백엔드를 넣으려면 다음이 필요하다.

1. **커널 레이어**: `metal_backend.h`의 `cactus_metal_*` 시그니처(약 75개 `encode_*` + 세션/메모리 함수)를 본떠 `cactus_<backend>_*` 계열을 만들거나, 이를 함수 포인터 테이블로 추상화한다. 최소 집합: `available()`, `session_begin/sync/flush/end`, `alloc_shared/free_shared`, `encode_quant_matmul`(CQ4 GEMV), `encode_quant_matmul_m`(GEMM), `encode_rms_norm`, `encode_rope_*`, `encode_kv_append_i8`, `encode_attention_i8`, `encode_binary/unary`, `encode_softmax_rows`, `encode_argmax`.
2. **메모리 모델**: Metal 경로는 통합 메모리를 전제로 `wrapHostPtr`/`newBufferWithBytesNoCopy`에 의존한다. 이산 메모리(Vulkan on Mali/Adreno는 대개 host-visible이지만 캐시 일관성 차이 있음)에서는 `metal_live[]` 추적을 "sync"가 아닌 "복사"로 바꿔야 한다.
3. **그래프 레이어**: `ComputeBackend` enum에 값 추가(`cactus_graph.h:86`), `cactus_backend_select` 문자열, `execute.cpp`의 `try_encode_metal`(638-1403) 분기와 `MetalExecGuard`, KV 캐시 할당(`ops_cache.cpp:104-106`)의 백엔드 분기. `param_io.cpp`는 backend를 저장하지 않으므로 포맷 변경은 불필요.
4. **셰이더**: `cactus_kernels.metal`의 CQ 디코드 수학(`cq_idx`, `cq4_il_base/nib`, Hadamard 버터플라이, INT8 KV 디양자화)을 GLSL/SPIR-V로 이식. `simdgroup_matrix` 대신 `VK_KHR_cooperative_matrix` 또는 서브그룹 연산.
5. **빌드**: `cactus-kernels/CMakeLists.txt`의 `if(APPLE)` 블록에 대응하는 Android 분기, 스텁 파일, `android/CMakeLists.txt` 링크.
6. **검증**: `test_metal_parity.cpp`를 백엔드 파라미터화하고, `test_matmul.cpp`의 CQ 정확 일치 테스트를 추가한다.
7. **정책**: 블로그의 배터리·백그라운드 논리에 따라 "프리필만 GPU, 디코드는 CPU" 같은 분할이 현실적일 수 있다. 현 그래프는 노드 단위 backend 태그를 지원하므로 트랜스파일러가 프리필 컴포넌트에만 GPU 태그를 붙이는 식으로 실험 가능하다.

Android NPU(QNN/NNAPI)나 Intel XPU 같은 "그래프 위임형" 가속기는 노드 단위 인코더 모델과 맞지 않는다. 그 경우 현 ANE 경로처럼 **컴포넌트 단위 위임**(인코더 전체를 벤더 포맷으로 내보내고 `model_npu.cpp`처럼 입출력만 이어 붙이기)이 더 적합하며, 트랜스파일러의 컴포넌트 분할이 그 접점이 된다.

---

## 6. 환경변수 총정리 (런타임 동작에 영향을 주는 것)

| 변수 | 레이어 | 효과 |
|---|---|---|
| `CACTUS_NUM_THREADS` | kernels | 스레드 풀 크기 |
| `CACTUS_GEMV_SB_PER_THREAD`, `CACTUS_INTERLEAVED_GEMV_THREADS` | kernels | 인터리브 GEMV 스레딩 |
| `CACTUS_DISABLE_TILED_MASKED_ATTENTION` | kernels(Apple) | Accelerate 마스크 어텐션 끄기 |
| `CACTUS_GATED_DELTANET_CHUNK_SIZE`, `CACTUS_GATED_DELTANET_PREFILL_OLD` | kernels | DeltaNet 프리필 |
| `CACTUS_KV_CACHE_FP16` | graph | KV를 FP16으로 |
| `CACTUS_FLUSH_CADENCE` | graph(Metal) | 커맨드 버퍼 commit 주기 |
| `CACTUS_PROFILE`, `CACTUS_PROFILE_FILE`, `CACTUS_CAPTURE_ENABLE/STDOUT/FILE/DIR`, `CACTUS_TRACE_EXECUTE` | graph | 프로파일/캡처(Metal 세션 비활성) |
| `CACTUS_KV_COMPRESS_AT`, `CACTUS_KV_COMPRESS_TO`, `CACTUS_KV_PRESERVE_SPECIAL` | engine | KeyDiff 토큰 축출 |
| `CACTUS_NO_CLOUD_TELE`, `CACTUS_DISABLE_CLOUD_HANDOFF` | engine | 텔레메트리/핸드오프 끄기 |
| `CACTUS_CLOUD_KEY`, `CACTUS_CLOUD_API_KEY`, `CACTUS_CLOUD_API_BASE`, `CACTUS_CLOUD_MODEL`, `CACTUS_CLOUD_HEADERS`, `CACTUS_CLOUD_STRICT_SSL` | engine | 클라우드 핸드오프 |
| `CACTUS_SUPABASE_URL/KEY`, `CACTUS_PROJECT_ID` | engine | 텔레메트리 목적지 |
| `CACTUS_ANE_COMPUTE_UNITS`, `CACTUS_ANE_{PREFILL,ENCODER,AUDIO,VISION}_COMPUTE_UNITS`, `CACTUS_ANE_FORCE_RECOMPILE` | engine(ANE, 현재 비활성 경로) | CoreML compute units |
| `CACTUS_TEST_BACKEND`, `CACTUS_TEST_MODEL`, `CACTUS_TEST_TRANSCRIPTION_MODEL`, `CACTUS_TEST_ASSETS`, `CACTUS_INDEX_PATH`, `CACTUS_PARITY_ALLOW_SKIP` | tests | 테스트 하네스 |
| `CACTUS_LIB_PATH` | python | ctypes 라이브러리 경로 |

---

## 7. 한 페이지 요약: 어떤 워크로드가 어디서 도는가 (Mac, 기본 auto)

| 단계 | 실행 위치 | 비고 |
|---|---|---|
| 토큰화, 템플릿, RAG 검색 | CPU(엔진) | |
| 임베딩 gather + PLE | GPU(융합 프롤로그) 또는 CPU | 인덱스 >4096이면 CPU |
| 프리필 CQ4 matmul | GPU `cq4_gemm_mma` | Metal 불가 시 CPU SDOT GEMM(또는 Accelerate는 FP16 밀집만) |
| 프리필 어텐션 | GPU `attn_prefill_*` | hd 512/256 특수화 |
| 디코드 QKV/MLP GEMV | GPU `gemv_cat`/`transform_gemv` | CPU면 `matmul_triple/pair` + SDOT |
| 디코드 어텐션 | GPU `attention_fused_i8`(kv_heads=1) 또는 split-K | CPU면 `attention_hybrid` 디코드 고속 경로 |
| KV append | GPU 커널 | INT8 32그룹 |
| LM head + softcap + argmax | GPU, float 3개만 회수 | 샘플링(온도>0)은 CPU에서 top-k 후보로 |
| Whisper/Parakeet 오디오 인코더 | CPU 그래프 (또는 GPU 노드) | ANE 경로는 비활성 |
| 비전 인코더 | CPU 그래프 (또는 GPU 노드) | conv2d는 Apple에서 im2col+sgemm |
| KV 축출, 툴콜 파싱, 클라우드 | CPU(엔진) | |

Android/Linux에서는 위 표의 GPU 칸이 전부 CPU가 되며, 디코드 GEMV는 Android에서 단일 스레드다.
