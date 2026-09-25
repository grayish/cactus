# 06. Cactus Kernels 심화: CPU 커널 레이어

> 대상 디렉터리: `cactus-kernels/`
> 공개 헤더: `cactus-kernels/cactus_kernels.h` (881줄)
> 핵심 파일: `src/matmul.cpp`(3031줄), `src/threading.h`(833줄), `src/attention.cpp`, `src/attention_hybrid.cpp`, `src/quants.cpp`, `src/fused.cpp`
> 이 장의 `파일:줄` 표기는 v2.2.1 기준이며, 코드가 바뀌면 줄 번호는 어긋날 수 있다.

---

## 0. 먼저 알아둘 사실 다섯 가지

1. **ARM64 전용이다.** 헤더가 `<arm_neon.h>`를 조건 없이 include 하고(`cactus_kernels.h:6`), 컴파일 플래그는 항상
   `-march=armv8.2-a+fp16+simd+dotprod+i8mm` 이다(`CMakeLists.txt:36`). Android는 arm64-v8a 외 ABI를 거부한다(`android/CMakeLists.txt:7-8`).
2. **x86 SIMD, SVE, SME 경로는 없다.** `cpu_has_i8mm()` 프로브는 있지만(`threading.h:51-76`) 아무도 호출하지 않고, `vmmlaq_s32`(SMMLA)도 쓰이지 않는다.
   실제로 쓰는 ISA 확장은 **NEON FP16 + DOTPROD(`sdot`)** 이다.
3. **"INT4 가중치 포맷"은 없다.** 저비트 가중치는 전부 **CQ(코드북) 포맷**이다. `Precision::INT8`은 KV 캐시와 커널 내부의 임시 활성화 양자화에만 쓰인다.
4. **Apple에서는 큰 형상에 Accelerate(cblas/vDSP)를 호출**한다. matmul, attention, conv, lstm, FFT가 해당된다.
5. CPU 쪽에는 RMSNorm+matmul 같은 **융합 커널이 거의 없다.** 그런 이름은 Metal 인코더 쪽에만 있다(`metal_backend_stub.cpp:34-39,116-119`). CPU 융합은 "여러 프로젝션이 활성화 변환을 공유"하는 `matmul_pair/triple` 정도다.

---

## 1. 공개 API 지도 (`cactus_kernels.h`)

| 계열 | 대표 함수 (헤더 줄) | 정밀도 |
|---|---|---|
| 원소별 연산 / 브로드캐스트 | `cactus_add_f16`, `_subtract_`, `_multiply_`, `_divide_`, `_add_scaled_f16`, `*_broadcast_f16` (36-114) | FP16 |
| 스칼라 op | `cactus_scalar_op_f16(..., ScalarOpType)` (73) | FP16 |
| 리덕션 | `cactus_{sum,mean,variance}_all_f16`(double 반환), `{min,max}_all_f16`, `*_axis_f16` (116-155) | FP16 → double/FP16 |
| 레이아웃 | `transpose_2d_f16`, `transpose_f16`, `concat_f16`, `cat_f16` (157-191) | FP16 |
| 밀집 matmul | `cactus_matmul_f16(a, b_transposed, c, M, K, N)` (193) — B는 N×K로 저장 | FP16 |
| **CQ 코드북 matmul** | `CactusQuantMatrix` 구조체 (206-224), `cactus_quant_{1,2,3,4}bit_gemv/gemm` (228-270), `cactus_quant_matmul` (272), `_matmul_pair/_triple` (299-307), `_orthogonal_matmul` (317), `_gemv_interleaved` (323-349), 임베딩 행 디양자화 (351, 366) | CQ1~CQ4 인덱스 + FP16 코드북 + FP16 그룹 norm |
| 정규화 / RoPE | `rms_norm_f16`, `layer_norm_f16`, `batchnorm_f16/f32`, `softmax_f16`, `rope_f16`, `gpt_j_rope_f16` (378-436) | FP16 |
| 활성화 | `relu`, `silu`, `gelu`(tanh 근사), `gelu_f16_erf`, `sigmoid`, `tanh`, `softcap_f16`, `gelu_scaled_multiply_f16`, `glu_f16/f32` (448-469, 279, 288) | FP16 |
| 어텐션 | `cactus_attention_f16` (472-491), `cactus_attention_hybrid_int8_fp16` (493-514) | FP16 / INT8 KV + FP32 스케일 |
| 컨볼루션 | `conv1d_*` 6종, `conv2d_*` 4종, `maxpool1d_f16`, `bilinear_interpolation_f16` (517-653) | FP16 |
| DSP | `cactus_stft_f16`, `rfft/irfft_f32_1d`, mel 필터뱅크, `compute_spectrogram_f32` (631, 820-844) | FP32 |
| 재귀 / 특수 | `lstm_cell_f16`, `bilstm_sequence_f16`, `gated_deltanet_decode/prefill_f16`, `gaussian_topk_f16`, `altup_predict/correct_f16` (662-747) | FP16 I/O, FP32 상태 |
| 샘플링 | `sample_f32/_f16` 및 `_ex` (temperature, top_p, top_k, min_p, 반복 패널티, logit bias) (750-800) | FP32/FP16 로짓 |
| 변환 / 양자화 | `int8↔fp32`, `fp16↔fp32`, `int8↔fp16`, `fp16_max_abs`, `quantize_kv_fp16_to_int8`, `kv_scales_count` (803-818, 872-879) | INT8/FP16/FP32 |
| 이미지 | `image_load/info/free`, `resize_uint8/float`, `normalize`, `to_patches`, `convert_to_rgb` (846-870) | uint8/FP32 |

상수: `Precision {INT8, FP16, FP32, CQ1, CQ2, CQ3, CQ4}` (10-18), `KV_QUANT_GROUP_SIZE = 32` (34).

각 함수의 구현 파일: `blas.cpp`(원소별·transpose·concat), `scalar.cpp`, `reduce.cpp`, `nn.cpp`(활성화·softmax·샘플링·batchnorm), `norms_rope.cpp`,
`matmul.cpp`, `attention*.cpp`, `quants.cpp`, `fused.cpp`, `conv*.cpp`, `dsp.cpp`, `lstm.cpp`, `image.cpp`.

---

## 2. `matmul.cpp`: 엔진 성능의 심장

### 2.1 밀집 FP16 matmul (`cactus_matmul_f16`, :627-682)

- **Apple**: `K >= ACCELERATE_K_THRESHOLD(256)` 이고 `M >= ACCELERATE_M_THRESHOLD(4)` 이면 (:18-19, :637)
  A, B를 FP32로 변환해 `cblas_sgemm`을 부르고 결과를 ±65504로 클램프한다(:642-661).
- **그 외(NEON)**: `cactus_matmul_f16_worker`(:542-625)가 `TILE_M=4 × TILE_N=4` 레지스터 타일에 `float16x8_t` 누산기를 두고,
  K 루프를 16단위로 언롤해 `vfmaq_f16`을 돌린다. 누산과 수평합(`hsum_f16x8`, :73-80) 모두 **FP16**이다.
  4행 블록 단위로 `Thresholds::SCALAR_EXPENSIVE`에 따라 병렬화한다(:668). 따라서 M=1이면 단일 스레드다.
- B는 패킹하지 않고 `b_transposed[n*K + k]`로 그대로 읽는다.

### 2.2 CQ 가중치 포맷 (`CactusQuantMatrix`)

```cpp
struct CactusQuantMatrix {
    uint32_t bits, K, N, group_size, num_groups, flags;
    const __fp16* codebook;          // 2^bits 개 FP16 (행렬당 1개, 그룹 공유)
    const __fp16* input_scale, *input_scale_recip;   // 열별 (선택)
    const __fp16* norms;             // (row, group) 별 스케일
    const uint8_t* packed_indices;   // bits 비트 인덱스 패킹
    const int8_t* left_signs, *right_signs;  // ±1, Hadamard 앞뒤
    const uint32_t* permutation;
    const __fp16* rotation;          // ORTHOGONAL 변형: 밀집 K×K
    const int8_t* expanded; const float* norm_f32;   // 사전 확장 INT8 레이아웃
};
```

- **패킹 주소**: 행·그룹당 `pgb = (group_size*bits + 7)/8` 바이트(:684-687), 위치는 `packed_indices + ((row*num_groups + group)*pgb) + k*bits/8` (:109-118).
- **검증**(`cactus_quant_valid_common`, :516-524): `group_size`는 2의 거듭제곱 ≤ 256, `K == group_size*num_groups`.
- **활성화 회전 (TurboQuant-H의 런타임 절반)** `cactus_quant_transform_hadamard_group`(:239-285), 그룹마다:
  1. `x *= input_scale_recip * left_signs`
  2. 정규화(1/√n) Walsh-Hadamard 변환. `gs==128`은 완전 언롤된 NEON FWHT(`cactus_quant_fwht128_f16`, :163-205)
  3. `*= right_signs`
  4. 선택적 `permutation`

  전체 활성화는 `parallel_for(M*num_groups, {16,1})`로 병렬 처리된다(:287-307).
  **가중치는 회전된 기저에 그대로 두고 활성화 쪽을 회전**시킨다는 점이 핵심이다(`docs/cactus_quants.md` 마지막 절과 일치).
- **ORTHOGONAL 변형**: 밀집 K×K FP16 `rotation`을 FP32로 적용(:2136-2284). 단일 그룹(`num_groups==1`)에서만 쓰이며, LM head/임베딩용 고속 경로는
  `cactus_quant_orthogonal_interleaved_lmhead_matmul`(:2034-2134)이다.
- **플래그 주의**: 파일 포맷의 플래그 비트(`FLAG_ORTHOGONAL_ROTATION = 1<<1`, `FLAG_INTERLEAVED_4ROW = 1<<2`)와 런타임 플래그(`1<<2`, `1<<3`)가 다르며,
  `cactus-graph/src/io.cpp:26-27,476-486`에서 변환한다.

### 2.3 코드북 디양자화가 내적에 융합되는 두 가지 방식

**(a) FP16 LUT 경로** — `gs % 32 != 0`일 때의 폴백
- FP16 코드북을 바이트 테이블로 두고, 인덱스 니블을 바이트 오프셋으로 바꿔 `vqtbl2q_u8`(4bit) / `vqtbl1q_u8`(1~3bit)로 곧바로 `float16x8_t` 가중치를 얻는다.
  이어서 `vfmaq_f16`. (:133-161, :423-463, :1183-1223)
- GEMV 타일: 4bit `TILE_N=12`(:925-971), 2bit `TILE_N=16`(:998-1034). 그룹 norm은 그룹당 한 번 FP32로 곱한다.

**(b) INT8 SDOT 경로** — `gs % 32 == 0 && gs <= 256`일 때의 주 경로
- 회전된 활성화를 그룹별 절대최대/127 대칭 INT8로 양자화(`tq_quantize_group_i8`, :1306-1332).
- 코드북(≤16 엔트리) 자체도 하나의 `cb_scale`로 INT8 양자화(`tq_quantize_codebook_i8`, :1292-1304) → 16바이트 `vqtbl1q_s8` LUT.
- 인덱스는 레지스터 안에서 바로 INT8 가중치로 확장(`tq_expand_i8_16`, :1349-1384).
- 내적은 `vdotq_laneq_s32`(SDOT by element). 출력 스케일 = `norm * cb_scale * act_scale` (:847-855).
- 가중치는 **4행 인터리브 패널**: 4개 출력행 × 16 k 마다 64바이트, 4바이트 워드 하나가 한 행의 연속 4 k값 (`tq_interleave_4x_s8`, :120-131).
  `vdotq_laneq` 한 번에 4개 출력이 나온다.

```cpp
// matmul.cpp:823-833 — 이중 누산기로 SDOT 의존 체인을 끊는다
#define TQ_SDOT_PANEL_T(DOT_A, DOT_B, BASE) do { \
    const int8_t* bk = (BASE) + k * 4; \
    DOT_A = CACTUS_DOTQ_LANE(DOT_A, vld1q_s8(bk),       a_lo, 0); \
    DOT_B = CACTUS_DOTQ_LANE(DOT_B, vld1q_s8(bk + 16),  a_lo, 1); \
    DOT_A = CACTUS_DOTQ_LANE(DOT_A, vld1q_s8(bk + 32),  a_lo, 2); \
    DOT_B = CACTUS_DOTQ_LANE(DOT_B, vld1q_s8(bk + 48),  a_lo, 3); \
```
주석(:810-813)은 "Apple Silicon (4 sdot/cycle, ~3 cycle latency)"에서 8-deep SDOT 체인을 끊기 위한 설계라고 설명한다.

### 2.4 가중치 레이아웃 세 가지

| 레이아웃 | 설명 | 위치 |
|---|---|---|
| 평문 row-major 패킹 | §2.2의 주소식. 호출 시 4행마다 256×4 INT8 스택 버퍼로 확장(`expand_group4`) | :738-780 |
| 사전 확장 INT8 (`expanded` + `norm_f32`) | 호출자가 만들어 둠. 크기 `N_blocks*num_groups*gs*4` | :1499-1536, :1580-1585 |
| `INTERLEAVED_4ROW` (디스크 포맷) | 4행 패널, `panel_bytes = 4*pgb`. 4bit: 16바이트 청크가 8 k × 4행, 바이트 `r*4+j`의 하위 니블이 k=j, 상위가 k=j+4. 3bit: 6바이트 세그먼트에 4행×4k | :2019-2032, :1463-1476 |

인터리브 4bit 커널은 사전 확장 없이 `vandq/vshrq + vqtbl1q_s8 + vdotq_laneq_s32`만으로 디코드한다(:2319-2332).

### 2.5 GEMV(디코드, M=1) vs GEMM(프리필, M>1) 디스패치

`cactus_quant_matmul`(:1541-1888)의 결정 순서:

1. `FLAG_ORTHOGONAL` → `cactus_quant_orthogonal_matmul`
2. `FLAG_INTERLEAVED_4ROW`:
   - M==1 → `cactus_quant_{4,3,2,1}bit_gemv_interleaved`
   - M>1, 4bit, gs%32==0, N%4==0 → `cactus_quant_4bit_gemm_interleaved`(:2442-2474). `TILE_M=8`행이 디코드된 가중치 레지스터를 공유, N블록 64개/스레드로 병렬.
3. 비인터리브 M==1 (:1587-1737): 사전 확장 없으면 `cactus_quant_Nbit_gemv`(gs%32==0이면 SDOT, 아니면 LUT), 있으면 인라인 SDOT 루프(그룹 2개/반복 + 프리페치).
   스레드 수는 `GemmThreading::get_gemv_threads`(:1641).
4. 비인터리브 M>1, gs%32==0 (:1744-1887): 활성화 전체를 INT8로 변환한 뒤, 가중치를 **INT8로 한 번 확장해 프로세스 전역 캐시**
   (`static std::unordered_map<const void*, ExpandEntry>`, `packed_indices` 포인터 키, FNV-1a 지문으로 유효성 검사, :1766-1797)에 보관. `TILE_M=8 × 4열` 타일, `parallel_gemm_tiles`.
5. gs%32 != 0 → `cactus_quant_dispatch_group_gemm`(:528-540): `TILE_N=16` 행 타일을 FP16으로 디코드, 그룹 내부 FP16 FMA + 그룹 간 FP32 누산.

**멀티 프로젝션 융합** `cactus_quant_matmul_pair/_triple`(:2639-2759): M==1이고 모든 행렬이 인터리브 4bit이며 활성화 변환이 동일하면
(`memcmp`로 스케일·부호·순열 비교, :2532-2555) Hadamard + INT8 양자화를 **한 번만** 수행하고 2~3개 행렬의 N 청크를 하나의 작업 큐로 처리한다. Q/K/V, gate/up에 해당.

**2단계 스핀 드라이버** `cactus_quant_two_phase_run`(:348-378):
- Phase A: 워커들이 원자 카운터로 활성화 그룹(Hadamard + INT8) 처리
- `groups_done`을 **busy-wait**
- Phase B: N 청크를 동적으로 가져감(남은 것이 `4*nt`보다 많으면 4개씩, 아니면 1개씩)
- 메인 스레드는 워커 0으로 참여. 주석: "a cv sleep costs ~5-10us per call, material at decode rates"(:371)

### 2.6 폴백의 실체

- `CACTUS_DOTQ_LANE`은 `__ARM_FEATURE_DOTPROD`가 있으면 `vdotq_laneq_s32`, 없으면 `vmull_s8 + vpaddlq_s16 + vpadd_s32` 에뮬레이션(:27-71).
  하지만 인터리브 커널과 `attention_hybrid.cpp`는 `vdotq_*`를 직접 호출하므로 **사실상 dotprod는 필수**다.
- 순수 스칼라 빌드는 없다. 스칼라 코드는 루프 꼬리와 참조 경로뿐이다.

---

## 3. `threading.h`: 스레드 풀과 플랫폼별 정책

### 3.1 풀 구조 (`CactusThreading::ThreadPool`, :386-552)
- mutex + `std::deque<std::function<void()>>` + 두 조건변수(`work_available`, `work_done`) + 원자 `pending_tasks`.
- 워커는 조건변수에서 **잠든다**. 스핀은 matmul의 2단계 드라이버에서만 일어난다.
- 지연 생성 싱글턴 `get_thread_pool()`(:554-557).
- **스레드 수**: 기본 `hardware_concurrency()`, 환경변수 **`CACTUS_NUM_THREADS`**로 재정의(:428-436), `MAX_WORKERS = 16` 상한. Android에서는 성능 코어 수로 추가 상한(:440-445).

### 3.2 분할 API
`enqueue`, `enqueue_batch`, `enqueue_n_threads`(Android는 `n*16` 작업으로 과분해해 동적 밸런싱, :524-549), `parallel_for`, `parallel_for_2d`, `parallel_reduce`,
`parallel_gemm_tiles`. `ParallelConfig{min_work_gate, work_per_thread}`로 "일이 gate보다 작으면 1스레드, 아니면 `min(pool, ceil(work/per_thread))`"(:559-573).

### 3.3 플랫폼별 휴리스틱
- **Android big.LITTLE**(`CoreTopology::detect`, :204-246): `/sys/devices/system/cpu/cpu%d/cpu_capacity`(없으면 `cpuinfo_max_freq`)를 읽어 최대의 **70% 이상**을 성능 코어로 분류(:237).
  워커 i는 `sched_setaffinity`로 `perf[i % perf.size()]`에 고정(:449-455).
  `prepare_current_thread_for_cactus_work()`(:367-381)는 `/proc/stat`을 20ms 간격으로 두 번 샘플링해 가장 한가한 성능 코어에 **호출 스레드**를 고정하며,
  엔진이 생성 진입 시 호출한다(`cactus-engine/src/complete.cpp:845,1321,1443,1487`).
- **Apple**: QoS나 affinity 호출 없음.
- **Linux**: macOS와 같은 `#else` 분기. affinity 없음.

### 3.4 임계값 표 (`Thresholds`, :575-591; `GemmThreading`, :593-624)

| 설정 | Android `{gate, per_thread}` | Apple/기타 |
|---|---|---|
| ATTENTION | {64, 32} | {32, 16} |
| ELEMENT_WISE | {5000, 2500} | {5000, 2500} |
| AXIS_REDUCE | {1000, 500} | {1000, 500} |
| ALL_REDUCE | {10000, 5000} | {10000, 5000} |
| SCALAR_BASIC | {30000, 15000} | {5000, 2500} |
| SCALAR_EXPENSIVE | {10000, 5000} | {2500, 1250} |

`GemmThreading`:
- **Android**: M>1이면 전 스레드, GEMV는 항상 **1 스레드**(:594-601) — 블로그의 "단일 코어 디코드" 정책이 코드로 구현된 곳.
- **iOS**: M≤1 → min(pool, 2). GEMV는 N블록 < 512이면 1, 아니면 min(pool, 3)(:602-611).
- **macOS/Linux**: M≤1 → min(pool, 4). GEMV: <256 → 1, <512 → 2, 그 외 min(pool, 5)(:612-622).

같은 헤더에 NEON 수학 헬퍼 `fast_exp_f32x4`(5차 다항, :78-115), `fast_tanh_f32x4`(:117-149), `fast_sigmoid_f32x4`, 비시간적 저장 `stream_store_f16x8`(`stnp`, :36-49),
그리고 언롤·프리페치·스트리밍 저장이 들어간 범용 `elementwise_op_f16`(:785-831)이 있다.

---

## 4. 어텐션

### 4.1 `cactus_attention_f16` 디스패치 (`attention.cpp:408-457`)
1. **Apple + 마스크 있음**: `seq_len ≥ 64`, `window_size == 0`, 차원 %8, logit cap 없음, 환경변수 `CACTUS_DISABLE_TILED_MASKED_ATTENTION` 미설정이면 Accelerate 경로(:17-242).
2. **마스크 없음**: `cactus_attention_f16_fast`(:245-406). Apple에서 `seq_len ≥ 64`면 역시 Accelerate로.
3. **일반 경로**(:459-729): 마스크, softcap `cap*tanh(s/cap)`(:610-612), 8의 배수가 아닌 꼬리 처리.

### 4.2 알고리즘
모든 경로가 **flash 스타일 온라인 softmax**(running max/sum, 최대값 갱신 시 누산기 재스케일)다.
- Accelerate 경로: `BLOCK_SIZE=64`, (batch, head)마다 Q·K 블록을 FP32로 바꿔 `cblas_sgemm`, 다항 exp2, P·V는 β=1 `cblas_sgemm`.
- Fast NEON 경로: `BLOCK_SIZE=32`, 작업 단위는 (batch, q_head, q_pos). 디코드(`seq_len==1`)는 QK를 **FP16**으로 누산 후 FP32 리덕션(:324-334), 프리필은 FP32 누산.
- **GQA**: `kv_head = q_head / (num_q_heads/num_kv_heads)`(:279, :300). K/V 복제 없음.
- **슬라이딩 윈도**: `kv_start = abs_q - window_size`(:315).
- **인과 마스크**: `kv_end = min(kv_len, position_offset + q_pos + 1)`.

### 4.3 "하이브리드" INT8/FP16 어텐션 (`attention_hybrid.cpp`)
"하이브리드"는 **혼합 정밀도 KV 캐시**를 뜻한다.
- 이미 캐시된 prefix(`cache_len`)는 **INT8** + (token, kv_head, 32원소 그룹)별 **FP32 스케일**.
- 현재 스텝의 `new_len` 토큰은 아직 양자화되지 않은 **FP16**.
- 캐시는 `cactus_quantize_kv_fp16_to_int8`(그룹별 absmax/127, `quants.cpp:224-294`)가 만들며 `cactus-graph/src/ops_cache.cpp`가 호출한다.

**디코드 고속 경로**(:10-327, 조건: `seq_len==1`, `head_dim ≤ 512`, `head_dim%32==0`, `group==32`):
Q를 32 그룹 INT8로 양자화(:72-92) → INT8 K와 `vdotq_s32`로 4키씩 QKᵀ(:118-173) → V는 `vmovl_s8 + vcvtq_f16_s16`으로 FP16 FMA(:236-278) → 블록 끝마다 FP32 누산기로 합산.
상수 `BLOCK_SIZE=64`, `QGROUP=32`, `MAX_HEAD_DIM=512`.

**일반 경로**(:371-617): `BLOCK_SIZE=32`, K를 INT8→FP16 변환. `k_scales == nullptr`이면 캐시를 FP16으로 취급(:491-504).
슬라이딩 윈도에서는 캐시 앞 `SINK_SIZE = 4` 토큰을 **어텐션 싱크**로 항상 남긴다(:435-459).

---

## 5. 런타임 양자화 포맷 정리

| 포맷 | 어디에 | 스케일 | 파일 |
|---|---|---|---|
| INT8 텐서 변환 (전역 스케일) | 유틸 | 1개 | `quants.cpp:8-222` |
| INT8 KV 캐시 | K/V 캐시 | 32원소 그룹, FP32, 대칭, zero-point 없음 | `quants.cpp:224-294` |
| CQ1~CQ4 코드북 가중치 | 모든 선형/임베딩 | (row, group) FP16 norm + 행렬당 FP16 코드북 | `matmul.cpp` |
| 커널 내부 임시 INT8 | 활성화·코드북 | 그룹별 absmax/127, 코드북 1개 스케일 | `matmul.cpp:1292-1332` |

임베딩 행 디양자화: `cactus_quant_dequantize_hadamard_embedding_row`(:1906-1952, 스칼라 FP32 역 FWHT),
`cactus_quant_dequantize_orthogonal_embedding_row`(:1954-2017, `EMBEDDING_ROTATION{128,256}`로 병렬).

---

## 6. 융합·특수 커널 (`fused.cpp`, `nn.cpp`)

- `cactus_gaussian_topk_f16`(:12-92): 행별 평균/σ, `MAX_SAFE_ABS=240` 프리스케일, `relu(x - (mu + ppf*sigma))` — Gemma 3n/4의 sparse activation.
- `cactus_altup_predict/correct_f16`(:94-181): AltUp 스트림 혼합(n ≤ 8) — Gemma E 시리즈.
- `cactus_gated_deltanet_decode/prefill_f16`(:751-850): Gated DeltaNet 선형 어텐션(FP32 상태). 프리필 청크 기본 64이나 **16으로 클램프**, 환경변수
  `CACTUS_GATED_DELTANET_CHUNK_SIZE`로 조정, `CACTUS_GATED_DELTANET_PREFILL_OLD`로 순차 경로 강제(:239-255, :834-839) — Qwen3.5 계열.
- `nn.cpp`: `gelu_scaled_multiply_f16`(GeGLU, :116-156), `softcap_f16`(:263-288), `glu_f16/f32`.
- `matmul.cpp`: `matmul_pair/_triple`(§2.5).

---

## 7. 주변 커널 한 줄 요약

| 파일 | 역할 | 특이점 |
|---|---|---|
| `dsp.cpp` | STFT / log-mel 스펙트로그램 (Whisper·Parakeet·Gemma 오디오 프런트엔드) | radix-2 FFT, Apple은 `vDSP_fft_zrip` + mutex 캐시. `compute_spectrogram_f32`는 HF feature extractor 옵션(center/pad/dither/preemphasis/dB range) 재현 |
| `conv.cpp` | 1D conv (LFM2 short conv, Parakeet subsampling 등) | `conv1d_f16_k7s3_oc8`는 8출력채널×4타임스텝 마이크로커널, Apple은 `vDSP_conv`/`cblas_sgemm` |
| `conv2d.cpp` | NCHW 3×3 / depthwise / 1×1 GEMM conv (비전 타워) | Apple은 im2col + sgemm, NEON은 `OW_VEC=4` |
| `lstm.cpp` | LSTM 셀, BiLSTM 시퀀스 (Parakeet TDT 디코더 등) | Apple은 [W_ih|W_hh] 연결 후 `cblas_sgemv` |
| `image.cpp` | stb_image 로드/리사이즈, normalize, to_patches, RGB 변환 (VLM 전처리) | SIMD·스레딩 없음 |

---

## 8. 테스트 구조

- `test.sh`: `rm -rf build && cmake .. -DCACTUS_BUILD_TESTS=ON && make -j` 후 모든 `./test_*` 실행, `--suite <name>`으로 하나만.
- 스위트: `test_attention`, `test_conv`, `test_dsp`, `test_elementwise`, `test_matmul`, `test_quant`, `test_reduce` (파일당 실행 파일 1개).
- 하네스 `tests/test_utils.h`: `TestRunner::run_test(name, bool)`, `run_bench`, `compare_arrays(tol=1e-2)`, `fill_random_fp16`(seed 42 고정).
- `test_matmul.cpp`는 `SyntheticCQ`로 임의 코드북/스케일/부호/패킹을 만들고, 스칼라 FP32 FWHT 참조와 MSE 0.1 임계값으로 비교한다(:292-319).
  CQ1~4, 인터리브, pair/triple, Metal 패리티(`#if !CACTUS_HAS_METAL` 가드)를 커버한다.

최소 테스트 예시(`tests/test_attention.cpp:8-20`):
```cpp
bool test_rms_norm() {
    const size_t batch = 4, dim = 128;
    std::vector<__fp16> input(batch * dim), weight(dim), output(batch * dim);
    fill_random_fp16(input, -1.0f, 1.0f);
    for (size_t i = 0; i < dim; i++) weight[i] = static_cast<__fp16>(1.0f);
    cactus_rms_norm_f16(input.data(), weight.data(), output.data(), batch, dim, 1e-6f);
    for (size_t b = 0; b < batch; b++) {
        float sum_sq = 0.0f;
        for (size_t d = 0; d < dim; d++) { float v = output[b*dim + d]; sum_sq += v*v; }
        if (std::abs(std::sqrt(sum_sq / dim) - 1.0f) > 0.05f) return false;
    }
    return true;
}
```

---

## 9. 새 칩에 맞춰 손댈 수 있는 노브 (요약, 상세는 11장)

**환경변수**

| 변수 | 효과 | 위치 |
|---|---|---|
| `CACTUS_NUM_THREADS` | 풀 크기 (≤16) | `threading.h:430` |
| `CACTUS_GEMV_SB_PER_THREAD` | 인터리브 GEMV에서 스레드당 슈퍼블록 수(기본 4) | `matmul.cpp:326-333` |
| `CACTUS_INTERLEAVED_GEMV_THREADS` | 인터리브 GEMV 스레드 상한 | `matmul.cpp:335-346` |
| `CACTUS_DISABLE_TILED_MASKED_ATTENTION` | Apple Accelerate 마스크 어텐션 끄기 | `attention.cpp:435` |
| `CACTUS_GATED_DELTANET_CHUNK_SIZE` / `_PREFILL_OLD` | DeltaNet 프리필 청크/경로 | `fused.cpp:242, 834` |

**상수**
- `threading.h`: `MAX_WORKERS=16`, `STREAMING_STORE_THRESHOLD=32768`, `ANDROID_DYNAMIC_CHUNK_MULTIPLIER=16`, 성능 코어 컷오프 0.70, `Thresholds` 표, `GEMV_MIN_N_BLOCKS`(iOS 512 / 기타 256).
- `matmul.cpp`: `ACCELERATE_M/K_THRESHOLD = 4/256`, 밀집 `TILE_M/N = 4/4`, SDOT GEMV `INT8_TILE_N=16`(16블록/스레드), LUT GEMV `TILE_N=12/16`, 프리필 `TILE_M=8`, 인터리브 GEMM 64블록/스레드, `group_size ≤ 256`, SDOT는 `gs%32==0`.
- 어텐션: `BLOCK_SIZE` 64(Accelerate)/32(NEON), 하이브리드 디코드 64, `SINK_SIZE=4`.
- conv: `total_compute < 100000`이면 단일 스레드, Apple `ACCELERATE_K/L_THRESHOLD = 32/128`.
- RoPE cos/sin 테이블은 thread_local로 (head_dim, theta) 키 캐시.

---

## 10. 읽기 순서 제안

1. `cactus_kernels.h` 전체를 훑어 함수 계열 파악 (30분)
2. `threading.h`의 `ParallelConfig`, `Thresholds`, `GemmThreading` (1시간) — 이후 모든 커널의 병렬화 결정을 이해하는 열쇠
3. `matmul.cpp`: `cactus_matmul_f16` → `CactusQuantMatrix` → `cactus_quant_transform_hadamard_group` → `cactus_quant_sdot_gemv_int8` → `cactus_quant_matmul` 디스패처 (반나절)
4. `attention_hybrid.cpp`의 디코드 고속 경로 (2시간)
5. `tests/test_matmul.cpp`의 `SyntheticCQ`로 직접 CQ 행렬을 만들어 커널을 호출해 보기 (실습)
