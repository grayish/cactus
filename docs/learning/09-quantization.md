# 09. 양자화: Cactus Quants(CQ)와 TurboQuant-H

> 근거: `docs/cactus_quants.md`, `blog/turboquant-h.md`, `python/cactus/convert/quantization/cq.py`, `python/cactus/convert/cli.py`, `python/cactus/convert/model_adapters/policy.py`, `cactus-kernels/src/matmul.cpp`, `cactus-graph/src/io.cpp`

---

## 1. 왜 또 하나의 양자화인가 (`docs/cactus_quants.md:15-34`)

기존 PTQ(GPTQ, AWQ, HQQ)의 세 가지 한계를 CQ가 겨냥한다.
1. **선형 레이어만 양자화한다.** Gemma 4 E2B는 per-layer 임베딩(PLI)만으로 모델의 60%다. 임베딩·모달리티 타워·브리지가 FP16으로 남으면 절반 이상이 손대지 않은 채 남는다.
2. **4bit 아래에서 무너진다.** 3bit에서 15~30pt, 2bit에서 생성 태스크가 거의 무작위.
3. **임베딩을 다룰 수 없다.** 임베딩은 토큰 id로 gather될 뿐 활성화와 곱해지지 않으므로 Hessian 기반 방법에 신호가 없다.

CQ는 "회전 + 코드북" 한 가지 레시피를 트랜스포머 선형, 비전/오디오 인코더, 크로스모달 브리지, PLI, 공유 임베딩 전부에 1~4bit로 적용한다.

## 2. 결과 요약

| 설정 (Gemma 4 E2B) | CQ | 최고 베이스라인 / FP16 |
|---|---|---|
| 4bit GSM8K | 71.20 | AWQ 70.53 / 73.67 |
| 4bit MMLU | 59.45 | AWQ 59.66 / 62.33 |
| 3bit GSM8K | **52.67** | AWQ 21.87 |
| 3bit BFCL Parallel | **81.17** | AWQ 58.00 |
| 2bit ARC-E | **50.80** | AWQ 27.40 |
| 2bit MMLU | **33.18** | HQQ 25.06 |
| Parakeet-1.1B 2bit WER | **0.147** | GPTQ 1.043 |
| ChartQA, 2bit 타워 + 4bit LLM | **48.44** | AWQ 14.22 |

- 4bit에서는 접전, 3bit부터 격차(최대 +32pt), 2bit에서는 CQ만 생존.
- 혼합 정밀도: 68개 고민감 유닛 4bit + 207개 3bit = **평균 3.26bit**가 스위트스폿, CQ는 균일 4bit 대비 ~2pt 손해인 반면 베이스라인은 7~15pt.
- **프로덕션 Gemma 4 E2B 번들**: LLM 선형 4bit, PLI 2bit, 공유 임베딩 4bit, 비전/오디오 타워 2bit → **4.79GB → ~0.9GB (5.3×)**, ~3.0bpw. MMLU 59.14, GSM8K 74.00, BFCL Parallel-Multi 78.00 유지. 최대 양보는 LibriSpeech WER 5.9 → 10.42.
- 약점: 2bit 오디오 타워의 WER, 2bit 툴콜링(모든 방법 0 근처), LFM 4bit(HQQ가 우세).

README의 표(F16/CQ4/CQ3.26/CQ2.54/CQ2)는 같은 실험의 12개 태스크 버전이다.

## 3. 알고리즘: 블로그와 코드의 차이를 포함해서

### 3.1 TurboQuant-H (블로그가 설명하는 개념)
TurboQuant(Zandieh et al., ICLR 2026)의 오프라인 변형:

| | TurboQuant | TurboQuant-H |
|---|---|---|
| 대상 | KV 캐시(런타임 활성화) | 임베딩 가중치 테이블(오프라인) |
| 회전 | 가우시안 QR 랜덤 직교, O(d²) | 정규화 Hadamard, O(N log N), 대칭=자기역행렬 |
| 양자화기 | Beta 분포 기반 좌표별 스칼라 | 위치별 Lloyd-Max 코드북(실제 가중치 분포로 학습) |
| 편향 보정 | QJL 1bit 잔차 | 없음 |
| 비트 | 2.5/3.5 | 2 + 0.125 코드북 오버헤드 = 2.125 |

수식: 행을 G=128 그룹으로 나눠 `x̂ = H̄_G x` (H̄ = H/√G), 위치 p마다 4개 centroid 코드북 C_p를 Lloyd-Max로 학습, `q = argmin |x̂_i − c_j|`, 디양자화는 `H̄_G · scatter(C_p, q)` (H̄가 대칭이라 전치 불필요). Gemma 4 E2B PLI: 2,496MB → 624MB(4×), LLM 전체 4,790 → 2,918MB(−40%), PPL 1.85 → 1.91.

그룹 크기 스윕: 32(오버헤드 0.5bit, 저하) / 64 / **128(0.125bit, 최적)** / 256 / 512(분포가 얇아져 저하).

### 3.2 코드가 실제로 하는 것 (`cq.py`)
- 상수: `GROUP_SIZE=128`, `HEADER_SIZE=84`, `PRECISION_CQ={1:3, 2:4, 3:5, 4:6}`, `FLAG_ORTHOGONAL_ROTATION=1<<1`, `FLAG_INTERLEAVED_4ROW=1<<2`(:22-26).
- **코드북(`make_codebook`, :47-76)은 가중치로 학습하지 않는다.** d차원 단위 구면 위 균등 벡터의 한 좌표의 해석적 밀도 `∝(1−x²)^((d−3)/2)`에 Lloyd-Max(200회)를 맞춘 `2^bits`개 centroid이며 `(group_dim, bits)`마다 하나다. 블로그의 "위치별 학습 코드북"과 다르다.
- **Hadamard 회전**(:79-100): `scipy.linalg.hadamard/√G` + 랜덤 ±1 좌/우 부호 벡터 + 열 순열(시드 `seed + 17·G`, 기본 1234).
- **`quantize_hadamard`**(:245-293, 선형용): K를 128 배수로 패딩 → AWQ식 `input_scale` 곱 → 128 그룹마다 행을 L2 norm으로 나누고 회전 후 최근접 코드북 인덱스 → (row, group) norm을 FP16으로 저장 → 선택적으로 역Hessian을 이용한 GPTQ식 오차 보정(`_gptq_correct_group`, :233-242).
- **`quantize_orthogonal`**(:296-311, 임베딩·출력 헤드용): K×K QR 랜덤 직교 회전 1회, 차원 K용 코드북, 행당 norm 1개.
- **input scale**(`convert/cli.py:269-295`): `ALPHA=0.25`, `x_abs^α / w_abs^(1−α)`, 기하평균 정규화, [1/8, 8] 클립. 활성화 통계는 Hessian 대각 또는 임베딩은 캘리브레이션 토큰 id에서.
- **텐서별 정책**(`policy.py`): norm/bias/비2D/위치 임베딩/K가 128 배수가 아닌 텐서 → FP16; depthwise·pointwise conv → INT8; Whisper 인코더, Parakeet LSTM/TDT/rel-pos-bias, Gemma 4 오디오·비전 타워 → FP16; 임베딩·tied 출력 헤드 → ORTHOGONAL CQ4(가능하면 인터리브); Gemma 4 `embed_tokens_per_layer` → Hadamard CQ(사용자 비트); 오디오/전사 컴포넌트 → Hadamard(GPTQ 없음); 나머지 → Hadamard + GPTQ.
- 컴포넌트별 비트: `--language-bits/--vision-bits/--audio-bits/--embedding-bits`(`python -m cactus.convert`).
- **혼합 2.54/3.26을 만드는 Python 코드는 없다.** 허용 비트 목록에만 있고(`cli/utils.py:22`) 로컬 변환은 거부하며(`cli/model.py:41-45`) 프리빌트 `-cq2.54`/`-cq3.26` 아카이브만 받는다.

> 정리: 코드의 CQ는 "TurboQuant-H 블로그"보다 QuIP#/GPTQ/AWQ 아이디어를 더 섞은 형태(부호·순열이 있는 Hadamard, 그룹 norm, AWQ 스케일, GPTQ 보정, 해석적 코드북)다. 블로그는 PLI에 국한한 연구 프리뷰이고 "TurboQuant-H PLI 가중치는 향후 릴리스에 포함 예정"이라고 적혀 있다.

## 4. 파일 포맷 (`write_cq_tensor`, `cq.py:321-410`; 읽는 쪽은 `io.cpp:1054-1145`)
84바이트 헤더(`CACT`, flags, alignment 32, ndim 2, shape (N,K,0,0), precision 3~6, data_bytes, scales_bytes, group_size, num_groups, N) → 32B 패딩 → 스케일 블롭(코드북 fp16 `2^bits`, `input_scale[K]`, `input_scale_recip[K]`, norms(인터리브면 4행 블록, 코드북은 int8 범위로 재스케일), 회전 데이터: Hadamard `left int8[G]`, `right int8[G]`, `perm u32[G]` 또는 전체 fp16 직교 행렬) → 패딩 → 패킹 인덱스(LSB 8원소 청크 `pack_indices_lsb` 또는 SIMD용 `INTERLEAVED_4ROW` 1/2/3/4bit 변형).

FP16/INT8 폴백 텐서도 같은 84바이트 `CACT` 헤더를 쓴다(precision 0 = INT8, `tensor_io.py:15-21`). 07장 §3.1과 대조해 보면 좋다.

## 5. 런타임에서 벌어지는 일 (06장 §2와 연결)
1. 그래프가 `.weights`를 mmap하고 `BufferDesc`에 코드북·스케일·norm·부호·순열 포인터를 연결(`io.cpp:529-552`), 파일 플래그를 커널 플래그로 재매핑.
2. matmul 시 `to_cq_matrix()` → `cactus_quant_matmul`.
3. 커널은 **활성화**에 `input_scale_recip`·좌부호 → FWHT(gs=128 언롤) → 우부호 → 순열을 적용하고(`cactus_quant_transform_hadamard_group`), 그룹별 INT8로 양자화한 뒤, 코드북도 INT8로 양자화해 16바이트 `vqtbl1q_s8` LUT로 만들고, `vdotq_laneq_s32`로 내적, 출력에 `norm × cb_scale × act_scale`을 곱한다.
4. 임베딩은 `cactus_quant_dequantize_hadamard_embedding_row`/`_orthogonal_embedding_row`로 행 단위 즉석 디양자화(`ops_tensor.cpp:313-411`).
5. Metal에서는 `cq4_transform` → `cq4_gemv`/`cq4_gemm_mma`가 같은 수학을 GPU에서 수행하며 CQ1은 GPU 경로가 없다(11장).

## 6. INT8 KV 캐시와의 관계
- 블로그(`turboquant-h.md:24`)는 "소형 모델은 KV가 INT4 아래로 가면 크게 열화되어 Cactus는 KV를 INT8로 유지한다"고 밝힌다.
- 구현: 32원소 그룹 FP32 스케일, 대칭 absmax/127(`quants.cpp:224-294`), 어텐션은 `cactus_attention_hybrid_int8_fp16`.
- 그래서 TurboQuant-H는 (원논문과 달리) KV가 아니라 임베딩 가중치에 적용된다.

## 7. 실습 아이디어
1. `cactus convert <model> --bits 4`와 `--bits 3` 번들의 `conversion_summary.json`을 비교해 어떤 텐서가 FP16/INT8 폴백인지 확인한다.
2. `cq.py`의 `make_codebook`을 numpy로 재현해 centroid 4개(2bit)가 어디에 놓이는지 그려 본다. 그룹 차원 128과 8190(orthogonal)에서 분포가 어떻게 다른지 본다.
3. `SyntheticCQ`(`cactus-kernels/tests/test_matmul.cpp:13-55`)를 이용해 `group_size`를 64/128/256으로 바꾸며 커널 처리량을 측정한다.
4. `docs/cactus_quants.md`의 혼합 정밀도 논리(민감도 순위)를 자신의 모델에 적용하려면 어떤 측정이 필요할지 설계해 본다.
