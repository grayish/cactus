# 07. Cactus Graph 심화: 노드, 메모리, 파일 포맷, KV 캐시

> 대상: `cactus-graph/cactus_graph.h`(1307줄), `src/core.cpp`, `builder.cpp`(1924줄), `execute.cpp`(2419줄), `io.cpp`(1178줄), `param_io.cpp`, `ops_*.cpp`, `graph_ffi.cpp`(1918줄), `metal_plan.cpp`(1825줄)

---

## 0. 흔한 오해 다섯 가지를 먼저 바로잡기

1. **C++에는 JSON 그래프 포맷이 없다.** 엔진이 읽는 것은 바이너리 `graph.cactus`(매직 `CGRF`)다. C++이 읽는 유일한 JSON은 `components/manifest.json`이며 그것도 엔진(picojson)이 파싱한다. `raw_ir.json`/`optimized_ir.json`/`transpiler_ir/*.json`은 Python 디버깅·재하강용이다.
2. **CPU 경로에는 아레나가 없다.** 2의 거듭제곱 버킷 풀 + liveness 기반 반환이다. 오프셋 패킹 아레나는 Metal 디코드 플랜에만 있다(`metal_plan.cpp:1359-1444`).
3. **"zero-copy"의 주 대상은 mmap 가중치다.** CPU에서 reshape/view/flatten은 `memcpy`다(`ops_math.cpp:828`). 입력을 별칭(alias)하는 CPU op은 outer extent가 1인 slice와 `index(dim=0)`뿐이다.
4. **CPU에는 자동 융합이 없다.** `DENSE_MLP_TQ_FUSED` 같은 융합 op은 Python 트랜스파일러가 명시적으로 방출한다. 자동 패턴 융합은 Metal 플래너에만 있다.
5. **KV "압축"은 그래프에 없다.** 그래프는 32원소 그룹 스케일의 INT8 KV를 저장할 뿐이고, 토큰 축출(KeyDiff)은 엔진(`cactus-engine/src/kv_compress.*`)에 있다.

---

## 1. 자료구조

### 1.1 노드 = 텐서
```cpp
// cactus_graph.h:444-451
struct GraphNode {
    size_t id;                        // 1부터 시작 (core.cpp:294)
    OpType op_type;                   // ~125종, enum 순서가 곧 직렬화 op 코드
    std::vector<size_t> input_ids;
    BufferDesc output_buffer;         // 이 노드의 출력 텐서
    OpParams params;                  // 모든 op이 공유하는 '뚱뚱한' 파라미터 구조체
};
```
별도의 Tensor 객체가 없다. `CactusGraph::nodes_`는 `vector<unique_ptr<GraphNode>>`, `node_index_map_`이 id → 인덱스다(:856-857).

### 1.2 `BufferDesc` (:255-344)
- `shape`, `total_size`(원소 수), `byte_size`, `precision`, `dynamic_dims`(차원별 마스크)
- 저장 모드 4가지. `get_data()` 우선순위: `external_data` → `pooled_data` → `shared_data` → `data` (`core.cpp:173-185`)
  - `external_data`: mmap 가중치, `set_external_input`
  - `pooled_data`: 버킷 풀에서 빌린 버퍼
  - `shared_data`: 그래프 간 `TensorStorage` 공유
  - `data`: `retain_outputs`된 노드의 전용 할당
- CQ 사이드 포인터 `cq_codebook, cq_input_scale(_recip), cq_norms, cq_left_signs, cq_right_signs, cq_permutation, cq_rotation, cq_flags`(:291-299)와 이를 커널 구조체로 묶는 `to_cq_matrix()`(:301-321)
- `PrecisionTraits`(:153-205): CQ는 "packed"라 `packed_size_of = ceil(count*bits/8)`

### 1.3 `OpParams` (:358-442)
공통(`scalar, scale, theta, epsilon, axis, pretransposed_rhs, position_offset, slice_*, window_size, is_causal, logit_cap, new_shape, permutation, output_precision, broadcast_info, backend`), 샘플링(`temperature, top_p, min_p, repetition_penalty, top_k, random_seed, bias_*`), MoE(`num_experts, num_experts_per_tok, moe_gated, activation`), 캐시(`max_cache_seq_len, cache_sink_size, cache_slot, cache_num_slots`), DSP/이미지 필드.

`backend`의 기본값은 `cactus_default_backend()`(:376)이며, 직렬화 시 **기록되지 않고 로드 시 버려진다**(`param_io.cpp:331-333, 394`). 즉 저장된 그래프의 노드 백엔드는 로드 시점의 전역 기본값으로 결정된다.

### 1.4 빌더 메서드 분류 (:542-833)
| 범주 | 메서드 |
|---|---|
| I/O·가중치 | `input, set_input, set_external_input, export/bind_tensor_storage, get_output, mmap_weights, mmap_embeddings, bind_mmap_weights, mark_embedded_input, release/prefetch_weight_pages, retain_outputs, persistent` |
| 원소별·수학 | `add, add_clipped, subtract, multiply, divide, 비교 6종, bitwise_and/or, where, masked_select_prefix, scalar_* 16종, abs, pow, precision_cast` |
| 활성화 | `relu, leaky_relu, clamp, silu, gelu, gelu_erf, sigmoid, tanh, glu` |
| 리덕션 | `sum, mean, variance, min, max, cumsum, softmax, topk` |
| 텐서/형상 | `reshape, view, expand, flatten, transpose, transposeN, slice, strided_slice, index, unfold, pad, concat, cat, gather, embedding` |
| NN | `matmul, rms_norm, layernorm, groupnorm, batchnorm, rope, rope_gptj` |
| 어텐션 | `attention`(3 오버로드), `attention_masked, rel_pos_bias, attention_int8_hybrid` |
| 캐시 | `kv_cache_state(max_seq, kv_heads, head_dim, window=0, sink=4, slots=1), kv_cache_append, attention_cached, conv_cache_state/append/initialize, recurrent_cache_state/write` |
| 컨볼루션 | `conv1d_*` 8종, `conv2d_*` 4종, `stft, maxpool1d` |
| 재귀·특수 | `lstm_cell, bilstm_sequence, gated_deltanet_decode/prefill, altup_predict/correct, gaussian_topk, moe_layer(gated/ungated), dense_mlp_tq_fused, qkv_tq_fused, projection_pair_tq_fused, logits_tq_softcap, stats_pool, weighted_stats_pool` |
| DSP·이미지 | `rfft, irfft, mel_filter_bank, spectrogram, image_preprocess, bilinear_interpolation` |
| 샘플링 | `sample, sample_with_options, scatter_topk` |
| 실행 | `execute(profile_file=""), hard_reset, soft_reset, soft_reset_keep_pool, set_prefill_mode` |

---

## 2. 생명주기: 한 번 만들고 여러 번 실행

### 2.1 빌드: 형상은 즉시, 메모리는 나중에
- 각 빌더는 입력을 검증하고 출력 형상을 바로 계산한다. `matmul`은 2D 입력과 K 일치를 확인하고 `{M, N}`을 만든다(`builder.cpp:140-165`).
- `add_node`(`builder.cpp:1323-1352`): 출력 형상이 비어 있으면 `inputs[0]` 형상을 복사. **출력 정밀도는 `PRECISION_CAST, EMBEDDING, TOPK, SCATTER_TOPK, SAMPLE, GAUSSIAN_TOPK`만 `params.output_precision`을 따르고 나머지는 `inputs[0]`의 정밀도를 상속**한다(:1333-1343). `BufferDesc` 생성은 크기 계산뿐이다(`core.cpp:64-69`).
- 실행 순서는 **삽입 순서**다(`execute.cpp:1898`). 위상 정렬이 따로 없지만 빌더는 이미 존재하는 id만 참조할 수 있고 `from_serialized`도 "입력이 먼저 복원됨"을 강제하므로(`io.cpp:371-387`) 구조적으로 위상 순서다.

### 2.2 `execute()` (`execute.cpp:1494-2305`)
1. `has_dynamic_shapes_ && runtime_shapes_dirty_`이면 `infer_shapes()`(572-599)
2. Metal 사용 가능 + METAL 태그 노드가 하나라도 있으면 Metal 모드(1569-1575). 노드 100개 미만이면서 LSTM을 포함하면 CPU로 강제(1576-1584)
3. **별칭 규칙**(`aliases_input`, 1623-1647): outer extent 1인 SLICE, axis 0 INDEX, (Metal 모드에서만) VIEW/RESHAPE/FLATTEN/동일 정밀도 CAST는 출력 버퍼를 갖지 않고 입력을 가리킨다
4. **liveness 분석**(1650-1682): `last_use`, `use_count`, `release_after[i]` 계산. 별칭 노드는 원본의 수명을 연장한다. INPUT, 캐시 상태, persistent, retained, 그래프 출력(`use_count==0`)은 절대 반환되지 않는다
5. **CPU 고속 경로**(1896-1925): 노드마다 `resize_from_pool(pool)`(retained면 `allocate()`) → `dispatch_node` → 마지막 소비자가 지나간 입력을 `release_memory(pool)`
6. 디버그/프로파일 경로(1927-2305): `CACTUS_PROFILE`, `CACTUS_PROFILE_FILE`, `CACTUS_CAPTURE_*`, `CACTUS_TRACE_EXECUTE`(:433-438)이 켜지면 중간 버퍼를 반환하지 않고 op별 시간표를 출력한다

### 2.3 메모리
- **CPU `BufferPool`**(`core.cpp:6-58`): 최소 1KiB, 64바이트 정렬, 2의 거듭제곱 버킷, 버킷별 free list, `active/pool/peak` 바이트 추적.
- `resize_from_pool`(`core.cpp:215-223`)은 `byte_size`가 바뀔 때만 재획득 → 동적 형상이 싸다(`test_dynamic_shapes.cpp:61-75`).
- **zero-copy가 실제로 성립하는 경우**: (a) `mmap_weights` → INPUT 노드의 `external_data`가 파일을 직접 가리킴(`io.cpp:513-514`), (b) `set_external_input`(`execute.cpp:349-362`; `set_input`은 memcpy), (c) 별칭 slice/index(+Metal의 view/reshape), (d) `export/bind_tensor_storage`로 컴포넌트 그래프 간 `shared_ptr` 전달.
- **Metal 메모리**: 노드별 persistent 공유 버퍼(1734-1747), 1500노드 이상 그래프의 transient 버퍼(1685-1733), 디코드형 플랜의 진짜 아레나(first-fit, 256B 단위, 텐서당 8MiB·전체 256MiB 상한, `metal_plan.cpp:1408-1443`).

### 2.4 동적 형상
- `set_runtime_input_shape(node, shape)`가 모든 차원을 dynamic으로 표시하고 `runtime_shapes_dirty_`를 세운다(557-564).
- `infer_output_shape`(444-543)에 MATMUL, 브로드캐스트, WHERE, EXPAND, UNFOLD, STRIDED_SLICE, PAD, 어텐션, TRANSPOSE, RESHAPE/VIEW/FLATTEN(0번 차원 재유도), CONCAT/CAT, SLICE 규칙이 있고 **기본값은 "빌드 형상을 유지하되 0번 차원만 동적 입력의 0번 차원으로"** (530-540). 즉 동적 지원의 실체는 선행(batch/seq) 차원이다.
- 동적 마스크는 그래프 파일 v6에 직렬화된다(`io.cpp:167-175, 309-319`).

### 2.5 디스패치와 스레딩
- 정적 함수 포인터 표 `dispatch_flat[OP_TYPE_COUNT]`을 `init_dispatch()`가 채운다(`execute.cpp:120-249`). 빈 슬롯은 "Unknown operation type" 예외(251-259).
- 각 `compute_*_node(GraphNode&, nodes, index_map)`는 `ops_*.cpp`에 있으며 `cactus_*` 커널을 호출한다.
- **그래프는 호출 스레드에서 노드를 하나씩 실행한다. op 간 병렬성은 없다.** 병렬성은 커널 내부 `CactusThreading::parallel_for/parallel_reduce`뿐이다. 일부 compute 함수는 thread_local 스크래치를 쓰며 `shrink_thread_local_buffers()`로 해제한다(`ops_nn.cpp:16-27`).
- Metal 모드에서는 노드마다 `cactus_metal_plan_encode` → `try_encode_metal` → 실패 시 `session_sync` 후 CPU `dispatch_node`(1874-1880). 융합 클러스터 실패는 앵커를 `metal_plan_banned_`에 넣고 `execute()`를 재귀 재실행(1843-1849), FP32→FP16 retype 실패는 retype을 끄고 재실행(1867-1872).

### 2.6 리셋 의미론
- `hard_reset()`(2316-2325): 노드, 인덱스 맵, mmap 파일, 가중치 캐시, 디버그 목록, 풀을 모두 비우고 `next_node_id_=1`.
- `soft_reset()`(2327-2367): `external_data`가 있는 INPUT(mmap 가중치·외부 입력), `weight_cache_`의 값, `persistent_node_ids_`(KV/conv/recurrent 캐시 상태)만 남기고 삭제. 번호는 `max_preserved_id+1`부터 이어진다. `prefill_mode_`가 아니면 풀을 비우고 thread_local 버퍼를 줄인다.
- `soft_reset_keep_pool()`(2369-2404): 풀을 비우지 않는 변형.

---

## 3. 디스크 포맷

### 3.1 가중치 파일 `*.weights` (매직 `CACT` = 0x54434143)
헤더 84바이트(`io.cpp:21-29`, `MappedFile::parse_header` 1054-1145; Python 작성자 `python/cactus/transpile/weight_compat.py:233-262`, `convert/quantization/cq.py:321-410`):

| offset | 필드 |
|---|---|
| 0 | u32 magic `0x54434143` |
| 4 | u32 flags: bit1 `ORTHOGONAL_ROTATION`, bit2 `INTERLEAVED_4ROW`, bit4 `EXTENDED_SHAPE` |
| 8 | u32 alignment (32) |
| 12 | u32 ndim |
| 16 | u64 dims[4] (+`EXTENDED_SHAPE`면 헤더 뒤에 u64 4개 추가) |
| 48 | u32 precision (`Precision` enum: INT8=0, FP16=1, FP32=2, CQ1..CQ4=3..6) |
| 52 | u64 data byte_size |
| 60 | u64 scales_bytes |
| 68 | u32 group_size |
| 72 | u32 num_groups |
| 76 | u64 original_N |

레이아웃: `align(헤더)` → 스케일 블롭(`scales_bytes>0`이면) → `align` → 패킹 데이터(1130-1139).

**CQ 스케일 블롭 순서**(`io.cpp:529-552`):
1. 코드북 `2^bits × fp16`
2. `input_scale[K]` fp16
3. `input_scale_recip[K]` fp16
4. `norms[N × num_groups]` fp16 (인터리브 레이아웃이면 4행 블록 단위)
5. ORTHOGONAL이면 `rotation`(fp16 K×K), 아니면 Hadamard `left_signs[gs] int8`, `right_signs[gs] int8`, `permutation[gs] u32`

- 파일 플래그 → 커널 플래그 재매핑: `CACTUS_QUANT_FLAG_ORTHOGONAL = 1<<2`, `INTERLEAVED_4ROW = 1<<3`(`io.cpp:541-552`).
- INT8 파일에 `group_size>0`이면 스케일 영역이 `activation_scales_data`가 된다(636-640).
- 파일명 해석: `X.cq4.weights` → `X.cq2.weights` → `X.weights` 순(`resolve_quantized_weight_file`, 37-54).

**mmap 정책**(934-1171): `open` → `mmap(PROT_READ, MAP_SHARED)` → fd 닫기. 스케일에 `MADV_WILLNEED`, 데이터에 `MADV_SEQUENTIAL`. `release_pages`는 `MADV_DONTNEED`, `prefetch_pages`는 `MADV_WILLNEED`.

**중복 매핑 제거(커밋 09cb35a)**: `mapped_files_`가 `shared_ptr`로 바뀌고 프로세스 전역 `MappedFileRegistry`(mutex + 정규 절대경로 → `weak_ptr<MappedFile>`, `cactus_graph.h:924-931`, `io.cpp:899-930`)가 생겼다. 여러 컴포넌트 그래프가 같은 파일을 요청하면 살아 있는 매핑을 공유한다. 테스트 `test_shared_mmap_weights_dedup`(`tests/test_io.cpp`).

### 3.2 그래프 파일 `graph.cactus` (매직 `CGRF`, 버전 4/5/6)
`save_graph`(`io.cpp:694-766`), `load_graph`(768-830):
- 헤더 `u32 magic, version, node_count, flags` → `u32[] graph_inputs`(임베드되지 않은 INPUT) → `u32[] graph_outputs`(리프)
- 노드마다: `u32 index`(런타임 id 보존), `u32 op_type`, `u32[] inputs`, `u64[] output_shape`, `u32 precision`, 파라미터 블록, v≥5: 임베디드 상수(`u32 has_embedded, u64 size, bytes`), v≥6: `dynamic_mask`
- 파라미터 블록(`param_io.cpp:389-423`): `u32 count` + `(u32 ParamField, value)` 쌍. op별 스키마(`op_schema` 161-265)가 허용 필드를 정하고, `Backend`는 기록하지 않으며 `ATTENTION_INT8_HYBRID`의 raw 포인터 필드는 `RuntimeOnly`로 직렬화 시 예외.
- 로드는 `from_serialized`(346-431)가 `input()`/`add_node()`를 재생하며 원래 id를 유지하고(`next_node_id_ = node_entry.index`), `broadcast_info`를 재계산하고, persistent/캐시 상태 노드를 다시 표시한다.
- **가중치는 그래프에 없다.** 평범한 INPUT 노드로 저장되고 로드 후 `bind_mmap_weights`로 연결된다.

### 3.3 번들 디렉터리
엔진이 읽는다(`cactus-engine/src/model.cpp:676, 919-1163`). manifest 컴포넌트는 `graph`, `runtime_input_node_ids`, `output_node_ids`, `bound_constant_bindings[{node_id,path}]`, `cache_state_node_ids[{layer_key,key,value}]`, `metadata`를 가진다. `load_component_graph`는 `CactusGraph::load` → `retain_outputs(output_node_ids)` → 바인딩마다 `bind_mmap_weights`.

---

## 4. KV 캐시 (`ops_cache.cpp`)

**레이아웃.** K와 V는 별개의 `KV_CACHE_STATE` 노드. 각각 `num_slots`개의 동일 크기 슬롯(`kv_slot_off`, 52-59)이 있고 슬롯은:
- `CacheMetadata` 64B: `current_seq_len, max_seq_len, num_kv_heads, head_dim, sink_size, num_slots, reserved[2]`(31-41)
- INT8 값 `max_seq × kv_heads × head_dim`, `[token][head][dim]`
- FP32 스케일 `max_seq × kv_heads × ceil(head_dim/32)`(43-46, 69-85). 그룹 크기는 `KV_QUANT_GROUP_SIZE = 32`

**정밀도.** 기본 INT8 + 32원소 그룹 FP32 스케일. `CACTUS_KV_CACHE_FP16=1`이면 FP16(`builder.cpp:11-14`, `ops_cache.cpp:95-101`). FP16 캐시는 Metal 가속 대상이 아니다.

**할당·성장.** 첫 실행 때 지연 할당(166-209), 빌드 시 persistent로 표시(`builder.cpp:1558`). 초기 용량: 슬라이딩 윈도면 `min(ceiling, window+sink+1)`, 멀티슬롯이면 `ceiling`, 그 외 `min(ceiling, 256)`(`kInitialCacheEntries`). 성장은 2배씩 `max_cache_seq_len`까지(150-158), 재할당 시 헤더·데이터·스케일 복사(109-148). Metal이면 공유 Metal 메모리에(105-107, 191-196). `steal_cache_buffer`(649-658)로 컴포넌트 간 이동, `shrink_cache_buffer`(797-804), `resize_cache_slots`.

**append.** `KV_CACHE_APPEND` 하나가 프리필/디코드 모두 담당. `new_seq_len = new_kv.total_size/(heads×dim)`. `kv_append_one_slot`이 `cactus_quantize_kv_fp16_to_int8`로 양자화해 `current_len` 위치에 쓴다(211-299). 멀티슬롯 배치는 행 i → 슬롯 i(317-326). 출력은 새 길이(float)(330).

**슬라이딩 윈도·축출.** CPU: `new_total > window`이면 앞 `sink`(기본 4) 토큰을 남기고 최근 꼬리를 `memmove`(StreamingLLM식, 237-257/269-293). Metal: 링 버퍼 `W = max_len - sink - 1`(`execute.cpp:1273-1281`). 버킷 프리필의 패딩 토큰은 `snapshot/rollback_cache_padded_append`(696-795)로 되돌린다.

**캐시 어텐션**(`compute_attention_cached_node`, 333-503): history 길이 = `cache_len - new_seq_len`이므로 **append가 attention보다 먼저** 와야 한다. INT8 경로는 `cactus_attention_hybrid_int8_fp16`(history는 INT8 캐시, 현재 토큰은 FP16 `key_new/value_new`, 486-502). FP16 경로는 `cactus_attention_f16`. 마스크는 FP16 캐시에서만. `position_offset` 센티널: `SIZE_MAX` = history_len, `SIZE_MAX-1` = 캐시 전용 어텐션(403-411).

**그 밖의 캐시.** `CONV_CACHE_STATE`(64B 헤더 + FP16 링 `window × hidden`, LFM2 short conv용), `CONV_CACHE_APPEND/INITIALIZE`, `RECURRENT_CACHE_STATE/WRITE`(DeltaNet 상태, memcpy).

---

## 5. op 파일별 요약

| 파일 | 내용 | 눈여겨볼 점 |
|---|---|---|
| `ops_nn.cpp` | `matmul`(CQ면 `to_cq_matrix()` → `cactus_quant_matmul`/orthogonal/Metal, FP16이면 thread_local 전치 후 `cactus_matmul_f16`, 29-78), MoE(`compute_moe_layer_node` 153-403: 토큰별 expert 카운팅 정렬, expert별 compact GEMM, 가중 scatter-add, 단일 토큰이면 `matmul_pair`), 융합 op `DENSE_MLP_TQ_FUSED`(405-598, **GeGLU**), `QKV_TQ_FUSED`(622-668, M=1 전용), `PROJECTION_PAIR_TQ_FUSED`, `LOGITS_TQ_SOFTCAP`, rms_norm, rope, softmax, attention(839-906), layernorm, glu, groupnorm | CPU에는 RMSNorm→matmul, SiLU×mul, RoPE→cache 융합 없음 |
| `ops_math.cpp` | 브로드캐스트 이항(233-368), 단항/스칼라/where/활성화(508-611), 리덕션·cumsum(613-812), reshape(memcpy 또는 Metal 별칭, 814-830), precision_cast(832-863) | |
| `ops_tensor.cpp` | transpose, gather, slice(별칭), strided_slice, unfold, expand, pad, concat/cat(FP16만), index(dim 0 별칭), bilinear, **embedding**(313-411: CQ 행을 즉석 디양자화, 인덱스 16개 초과면 행 캐시), persistent | |
| `ops_conv.cpp` | depthwise causal conv1d(LFM2 short conv, `CONV_CACHE_*`와 짝), conv1d k3/generic/k9/pointwise, conv2d 4종(NCHW), batchnorm, stft, maxpool1d | INT8 그룹 가중치는 호출마다 FP16으로 디양자화(10-41) |
| `ops_recurrent.cpp` | Gated DeltaNet 디코드/청크 프리필(Qwen3.5 선형 어텐션; 출력이 `{B, T+K, Hv, V}`로 어텐션 출력과 새 상태를 함께 담음), GPT-J rope, lstm_cell, bilstm_sequence, AltUp, gaussian_topk(Gemma3n/4), stats_pool | |
| `ops_dsp.cpp` | rfft/irfft, mel_filter_bank(htk/kaldi/slaney), spectrogram | 155줄 |
| `ops_image.cpp` | `image_preprocess`: float 변환 → 리사이즈 → 정규화 → 패치 `[num_patches, ps·ps·C]` | 52줄 |
| `ops_sample.cpp` | `sample`(마지막 행: temperature, top-p, min-p, top-k, 반복 패널티, bias; 시드는 빌드 시 시계), `topk`(`[2,B,k]` FP32), `scatter_topk` | |
| `metal_plan.cpp` | Metal 전용 자동 융합 규칙 1~18 (11장 §3) | |

---

## 6. `graph_ffi.cpp`: C FFI와 트랜스파일러의 접점
- 약 147개 `extern "C"` 진입점. 핸들은 `struct GraphHandle { CactusGraph graph; }`. 모든 함수는 "검증 → try 안에서 메서드 호출 → `*out=node_id; return 0` / 예외면 `last_error_message` 설정 후 -1"의 얇은 래퍼.
- 인트로스펙션: `cactus_graph_get_output_info`(`cactus_tensor_info_t{precision, rank, shape[8], num_elements, byte_size}`), `get_node_op_type`, `get_node_inputs`, `get_output_ptr`.
- **트랜스파일러 연결**: Python이 IR을 만든 뒤 ctypes로 이 FFI 빌더들을 호출해 그래프를 구성하고 `Graph.save()` → `cactus_graph_save`로 `graph.cactus`를 쓴다(`python/cactus/bindings/cactus.py`). 즉 "IR → 그래프 하강"은 Python 프로세스 안에서 실제 `CactusGraph`를 만드는 것이다.
- 오류 모델(`last_error.cpp`): C++은 예외(`runtime_error`, `invalid_argument`, `out_of_range`), FFI 경계에서 `thread_local std::string last_error_message`로 변환, `cactus_get_last_error()`로 읽음. `Logger`(레벨 기본 WARN, 콜백, 마지막 ERROR 보관)는 별도.

---

## 7. 테스트
- `test_utils.h`: `TestRunner`, `TestFixture<T>`(`create_input, set_input_data, execute, get_output, verify_output`, 소멸자에서 `hard_reset`), `compare_arrays(tol=1e-2)`.
- 케이스 수: `test_cache` 35, `test_metal_parity` 46, `test_ops` 24, `test_nn` 14, `test_io` 8, `test_precision` 8, `test_dsp_ops` 5, `test_graph` 4, `test_dynamic_shapes` 3, `test_image` 3.
- `test_precision.cpp`는 FP16 덧셈, 브로드캐스트 4종, `PrecisionTraits::size_of`, 입력 버퍼 바이트 수, FP16→FP32 캐스트를 확인한다. CQ/INT8 수학은 다루지 않는다.

최소 예시(`tests/test_graph.cpp:9-31`):
```cpp
TestUtils::FP16TestFixture fixture("Complex Graph Structure");
size_t a = fixture.create_input({2, 2}), b = fixture.create_input({2, 2}), c = fixture.create_input({2, 2});
size_t add_ab = fixture.graph().add(a, b);
size_t mul = fixture.graph().multiply(add_ab, c);
size_t out = fixture.graph().scalar_add(mul, 1.0f);
fixture.set_input_data(a, {1,2,3,4}); fixture.set_input_data(b, {2,3,4,5}); fixture.set_input_data(c, {2,2,2,2});
fixture.execute();
return fixture.verify_output(out, {7, 11, 15, 19});
```

---

## 8. 멘탈 모델: `graph.matmul(a, b)`에서 바이트까지 10단계

1. **빌드.** `CactusGraph::matmul`이 2D·K 일치를 검사하고 `{M,N}`, `pretransposed_rhs`, backend를 기록(`builder.cpp:140-165`).
2. **노드 생성.** `add_node`가 `GraphNode{id, MATMUL, {a,b}, BufferDesc({M,N}, prec(a)), params}`를 `nodes_`에 추가. 메모리 없음.
3. **가중치.** `b`가 `mmap_weights`/`bind_mmap_weights`에서 왔다면 레지스트리 공유 mmap을 가리키는 INPUT 노드이며 CQ 코드북·스케일·norm 포인터가 `BufferDesc`에 연결됨(`io.cpp:501-560`).
4. **입력.** `set_input`은 memcpy, `set_external_input`/`bind_tensor_storage`는 별칭(`execute.cpp:326-417`). 필요하면 `set_runtime_input_shape`.
5. **`execute()`.** `infer_shapes()`로 동적 선행 차원 전파 → CPU/Metal 모드 결정 → Metal이면 구조 해시로 캐시된 융합 플랜 재사용(1569-1615).
6. **liveness.** `last_use`, `use_count`, `release_after` 계산(1650-1682).
7. **출력 버퍼.** MATMUL 노드가 버킷 풀에서 `resize_from_pool`(retained면 전용 할당). Metal이면 아레나 슬롯/공유 버퍼.
8. **디스패치.** `dispatch_flat[MATMUL]` → `compute_matmul_node`. Metal이면 먼저 `cactus_metal_encode_quant_matmul(_m)`/`gemm_f16` 시도(643-666).
9. **커널.** CQ 가중치면 `rhs.to_cq_matrix()` → `cactus_quant_matmul`; FP16이면 thread_local 전치 → `cactus_matmul_f16`. 커널이 스레드 풀에 일을 나눠 주고 블록.
10. **반환·읽기.** 마지막 소비자였던 입력은 풀로 반환(1920-1922). MATMUL 출력은 리프/retained면 생존. `get_output(id)`가 raw 포인터를 돌려주고, 다음 `execute()`는 같은 그래프를 입력·형상·캐시만 바꿔 재사용한다.
