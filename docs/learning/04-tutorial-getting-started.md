# 04. 튜토리얼: 설치부터 첫 추론, C API, Graph API까지

> 목표: 2~3시간 안에 (1) CLI로 모델을 돌려 보고, (2) Python과 C에서 `cactus_complete`를 직접 호출하고,
> (3) Graph API로 작은 그래프를 만들어 실행하고, (4) 테스트 스위트를 돌려 본다.
> 환경: Apple Silicon Mac 또는 ARM64 Linux(라즈베리파이 5 포함). x86 PC에서는 빌드되지 않는다(02장).

---

## 단계 0. 준비물

| 항목 | 이유 |
|---|---|
| Python 3.12 (`python3.12`) | `setup` 스크립트가 3.12를 요구한다(`setup:45-52`) |
| cmake, C++20 컴파일러 (Apple clang / g++) | 세 라이브러리 빌드 |
| Linux: `libcurl4-openssl-dev` | 클라우드/텔레메트리용 curl. 없으면 `cactus build`가 거부(`compile.py:23-42`) |
| 디스크 1~5GB | 모델 번들 (Gemma 4 E2B CQ4 ≈ 0.9GB) |
| (선택) HuggingFace 토큰 | 게이트 모델 변환 시 |

```bash
# Ubuntu/Debian ARM64
sudo apt-get install python3.12 python3.12-venv python3-pip cmake build-essential libcurl4-openssl-dev
# macOS
brew install cmake python@3.12
```

---

## 단계 1. 가장 빠른 경로: Homebrew (macOS)

```bash
brew install cactus-compute/cactus/cactus
cactus run                      # 기본 모델 google/gemma-4-E2B-it 을 다운로드해 대화형 REPL 시작
```

이 포뮬러는 PyPI 휠을 virtualenv에 설치한 것이다. 소스 트리가 없으므로 `cactus build/test/benchmark/clean`은 동작하지 않는다.
학습 목적이라면 단계 2의 소스 빌드를 권한다.

---

## 단계 2. 소스에서 개발 환경 만들기

```bash
git clone https://github.com/cactus-compute/cactus && cd cactus
source ./setup            # venv 생성, python/requirements.txt 설치, `pip install -e python`, DCO 훅 설정
cactus --help             # 명령 목록 확인
cactus build              # 엔진 빌드 + bin/run, bin/transcribe 컴파일 (python/cactus/bin/)
cactus build --python     # ctypes 가 로드할 libcactus_engine.{dylib,so} 생성
```

무엇이 만들어졌는지 확인:
```bash
ls cactus-engine/build/ | grep libcactus_engine     # libcactus_engine.a, .dylib/.so
ls python/cactus/bin/                                # run, transcribe
```

> `setup`은 반드시 `source`로 실행해야 한다. 직접 실행하면 안내 메시지와 함께 종료된다(`setup:3-6`).

---

## 단계 3. CLI로 첫 추론

### 3.1 텍스트 대화
```bash
cactus run LiquidAI/LFM2-VL-450M                 # 작은 모델 (450M) 로 빠르게 시작
cactus run google/gemma-4-E2B-it --prompt "한국어로 자기소개 해줘"
```
동작 순서(`python/cactus/cli/run.py`, `cli/model.py:161-210`):
1. `weights/<name>-cq4/`(체크아웃) 또는 `~/.cache/cactus/weights/`(pip 설치)에 번들이 있는지 확인
2. 없으면 HF `Cactus-Compute/<name>`에서 런타임 버전 이하의 최신 태그 아카이브(`*-cq4.zip`)를 받아 SHA256 검증 후 풀기
3. 다운로드 실패 시 로컬 변환(`cactus convert`)으로 폴백
4. 네이티브 `bin/run` 실행 (REPL)

REPL 명령: `/image <path> [prompt]`, `/audio <path> [prompt]`, `/record`(SDL2 빌드 시), `/clear`, `reset`, `exit` (`cactus-engine/tests/run.cpp:966-982`).
각 응답 뒤에 TTFT, prefill/decode tok/s, confidence, RAM이 출력된다.

### 3.2 옵션 실험
```bash
cactus run google/gemma-4-E2B-it --bits 3.26                # 혼합 정밀도 프리빌트
cactus run google/gemma-4-E2B-it --backend cpu             # Mac에서 Metal 대신 CPU 강제
cactus run google/gemma-4-E2B-it --thinking --prompt "..."   # thinking 모드
cactus run google/gemma-4-E2B-it --image photo.jpg --prompt "무엇이 보이나요?"
cactus run Cactus-Compute/needle                            # 26M 툴콜링 모델 + 데모 툴셋
cactus run <model> --no-cloud-handoff                       # 클라우드 핸드오프 끄기
```

### 3.3 전사와 서버
```bash
cactus transcribe                                  # nvidia/parakeet-tdt-0.6b-v3 로 실시간 마이크 전사 (SDL2 필요)
cactus transcribe --file python/cactus/assets/test.wav
cactus serve google/gemma-4-E2B-it --port 8080     # OpenAI 호환 서버
curl localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"google/gemma-4-E2B-it","messages":[{"role":"user","content":"hi"}],"stream":true}'
cactus code                                        # 로컬 서버를 쓰는 터미널 코딩 에이전트 (Node 22+)
```

### 3.4 관찰 포인트
- 첫 실행 시 프리필 그래프 워밍업이 돈다(`model.cpp:872-885`).
- `cactus list`로 번들과 CQ 비트를 확인한다. 비트는 `.weights` 헤더 offset 48의 precision 값에서 읽는다(`cli/list.py:26-35`).
- 텔레메트리는 기본 ON이다. 끄려면 `CACTUS_NO_CLOUD_TELE=1` (08장 §6).

---

## 단계 4. Python에서 C API 직접 호출

```python
import json
from cactus import ensure_model, cactus_init, cactus_complete, cactus_destroy, cactus_set_backend

bundle = ensure_model("LiquidAI/LFM2-VL-450M")        # 없으면 다운로드, Path 반환
# cactus_set_backend("cpu")                           # (선택) Metal 대신 CPU

model = cactus_init(str(bundle), None, False)         # (bundle_dir, corpus_dir, cache_index)

messages = json.dumps([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "2+2는?"}
])
options = json.dumps({"max_tokens": 64, "temperature": 0.2, "stop_sequences": ["<|im_end|>"]})

def on_token(tok, tok_id, user):        # 스트리밍 콜백
    print(tok, end="", flush=True)

result = cactus_complete(model, messages, options, None, on_token)   # tools=None
print("\n", {k: result[k] for k in ("confidence", "time_to_first_token_ms", "decode_tps", "cloud_handoff")})
cactus_destroy(model)
```

- 반환값은 dict이며 키는 `success, error, cloud_handoff, response, thinking, function_calls, segments, confidence, confidence_threshold, time_to_first_token_ms, total_time_ms, prefill_tps, decode_tps, ram_usage_mb, prefill_tokens, decode_tokens, total_tokens`(`python/README.md`, `utils.h:1898-1952`).
- 옵션 키 전체 목록과 기본값은 08장 §1.2 표를 본다. `temperature` 기본 0.0은 **greedy**를 뜻하며 모델 패밀리 기본값을 쓰려면 음수를 준다(`model.cpp:4594-4597`).

### 4.1 자동 RAG 체험
```python
model = cactus_init(str(bundle), "cactus-engine/tests/assets/rag_corpus", True)
# corpus_dir 의 *.txt/*.md 를 ≤128토큰 청크로 잘라 같은 모델로 임베딩 → index.bin/data.bin 생성
result = cactus_complete(model, json.dumps([{"role":"user","content":"문서에 나온 주제는?"}]), None, None, None)
```
검색은 fp16 코사인 상위 20 → BM25 재랭크 → RRF(0.8/0.2) → 상위 5개를 시스템 메시지에 주입한다(`rag.cpp:116-208`).

### 4.2 툴 콜링
```python
tools = json.dumps([{"type":"function","function":{"name":"get_weather",
          "description":"도시의 날씨","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}])
res = cactus_complete(model, json.dumps([{"role":"user","content":"서울 날씨 알려줘"}]),
                      json.dumps({"force_tools": True}), tools, None)
print(res["function_calls"])
```
`force_tools`는 Gemma/Needle 계열에서 트라이 기반 제약 디코딩을 켠다(`constraints.cpp:437-450`).

---

## 단계 5. C/C++에서 호출

```cpp
// hello_cactus.cpp
#include "cactus_engine.h"
#include <cstdio>

static void on_token(const char* tok, uint32_t, void*) { fputs(tok, stdout); fflush(stdout); }

int main() {
    cactus_model_t m = cactus_init("weights/lfm2-vl-450m-cq4", nullptr, false);
    if (!m) { printf("init failed: %s\n", cactus_get_last_error()); return 1; }
    const char* msgs = R"([{"role":"user","content":"What is the capital of France?"}])";
    const char* opts = R"({"max_tokens":32})";
    char out[8192];
    int rc = cactus_complete(m, msgs, out, sizeof(out), opts, nullptr, on_token, nullptr, nullptr, 0);
    printf("\nrc=%d\n%s\n", rc, out);
    cactus_destroy(m);
}
```
```bash
# macOS
clang++ -std=c++20 -O2 hello_cactus.cpp -Icactus-engine -Lcactus-engine/build -lcactus_engine \
  -framework Accelerate -framework Metal -framework Foundation -framework MetalPerformanceShaders \
  -framework CoreML -framework Security -framework SystemConfiguration -framework CFNetwork \
  cactus-engine/libs/curl/macos/libcurl.a -lz -o hello_cactus
# Linux ARM64
g++ -std=c++20 -O2 hello_cactus.cpp -Icactus-engine -Lcactus-engine/build -lcactus_engine -lcurl -lpthread -o hello_cactus
```
(정확한 프레임워크 목록은 `cactus-engine/CMakeLists.txt:56-60`과 `apple/CMakeLists.txt`를 따른다. 가장 쉬운 방법은 `cactus-engine/tests/CMakeLists.txt`를 본떠 CMake 타깃을 만드는 것이다.)

---

## 단계 6. Graph API로 작은 그래프 만들기

### 6.1 C++
```cpp
#include "cactus_graph.h"
#include <vector>
#include <cstdio>

int main() {
    CactusGraph g;
    size_t a = g.input({2, 3}, Precision::FP16);
    size_t b = g.input({4, 3}, Precision::FP16);          // matmul 의 RHS 는 (N, K) 로 미리 전치된 형태
    size_t y = g.matmul(a, b, /*pretransposed_rhs=*/true); // -> {2, 4}
    size_t z = g.scalar_multiply(g.silu(y), 2.0f);

    std::vector<__fp16> da = {1,2,3, 4,5,6};
    std::vector<__fp16> db(12, (__fp16)0.5f);
    g.set_input(a, da.data(), Precision::FP16);
    g.set_input(b, db.data(), Precision::FP16);
    g.execute();
    auto* out = static_cast<__fp16*>(g.get_output(z));
    for (int i = 0; i < 8; i++) printf("%f ", (float)out[i]);
    g.hard_reset();
}
```
빌드는 `cactus-graph/tests/CMakeLists` 방식(그래프 정적 라이브러리 + 커널 링크)을 따른다.

포인트:
- 빌더는 형상만 계산한다(`builder.cpp:140-165, 1323-1352`). 메모리는 `execute()`에서 버킷 풀로부터 받는다.
- 중간 노드(`y`)는 마지막 소비자 뒤에 풀로 반환되므로 읽고 싶으면 `g.retain_outputs({y})`를 먼저 호출한다(`test_graph.cpp:41`).
- `soft_reset()`은 mmap 가중치·캐시 상태 노드만 남기고 나머지를 지운다. 같은 가중치로 다음 프롬프트를 처리할 때 쓴다(`execute.cpp:2327-2367`).

### 6.2 Python (ctypes `Graph`)
```python
import numpy as np
from cactus import Graph

g = Graph()
x = g.input((1, 64), Graph.FP16)
w = g.input((32, 64), Graph.FP16)            # (N, K)
y = g.matmul(x, w, pretransposed_rhs=True)   # Tensor 객체 반환
g.set_input(x, np.random.randn(1, 64).astype(np.float16))
g.set_input(w, np.random.randn(32, 64).astype(np.float16))
g.execute()
print(y.numpy().shape)                       # (1, 32)
g.save("/tmp/tiny.cactus")                   # CGRF 직렬화
g2 = Graph.load("/tmp/tiny.cactus")
```
`Graph.CPU`/`Graph.METAL` 상수를 op의 `backend=` 인자로 주면 노드 단위로 백엔드를 고정할 수 있다(`bindings/cactus.py:1568-1569`).

### 6.3 진짜 가중치 얹기
```python
w = g.mmap_weights("weights/lfm2-vl-450m-cq4/<tensor>.weights")   # CQ4 텐서를 zero-copy 로 INPUT 노드에 매핑
y = g.matmul(x, w)                                                 # CQ RHS → cactus_quant_matmul 경로
```
`.weights` 파일명은 번들의 `weights_manifest.json`에서 찾는다.

---

## 단계 7. 테스트와 벤치마크 돌리기

```bash
cactus test --list                        # 컴포넌트/스위트 목록
cactus test --component kernels           # cactus-kernels/test.sh: attention, conv, dsp, elementwise, matmul, quant, reduce
cactus test --component graph             # cache, dsp_ops, dynamic_shapes, graph, image, io, metal_parity, nn, ops, precision
cactus test --component engine --model LiquidAI/LFM2-VL-450M    # 번들 준비 후 test_llm, test_vlm, test_stt ...
cactus test --suite matmul                # 이름으로 하나만 (컴포넌트 자동 해석)
cactus benchmark                          # == engine/test_benchmark, README 표의 출처
cactus test --ios / --android             # 연결된 기기에서
```
- 엔진 테스트는 `CACTUS_NO_CLOUD_TELE=1`을 자동 설정한다(`--enable-telemetry`로 해제).
- Mac에서 CPU 경로만 테스트하려면 `--backend cpu` (`CACTUS_TEST_BACKEND`로 전달됨).

---

## 단계 8. 첫 모델 변환

```bash
pip install -e "python/[convert]"           # torch, transformers==5.5.4, scipy
cactus convert Qwen/Qwen2.5-0.5B ./weights/qwen25-0.5b-cq4 --bits 4
cactus run ./weights/qwen25-0.5b-cq4 --prompt "Hello"
```
출력 로그에 "Running converter (Crassus …) / IR simplifier (Pompey …) / Generator (Caesar …)"가 보이면 새 트랜스파일러 경로다(10장).
`--weights-only`로 CQ 가중치만 만들고 멈출 수도 있다.

---

## 자주 막히는 지점

| 증상 | 원인/해결 |
|---|---|
| `OSError` while loading libcactus_engine | arm64 전용. Linux는 libcurl 부재, 또는 오래된 빌드. `cactus build --python` 재실행 (`bindings/cactus.py:38-47`) |
| `cactus build`가 curl 없다고 거부 | `libcurl4-openssl-dev` 설치 |
| 로컬 변환은 되는데 `--bits 2.54` 실패 | 혼합 정밀도는 프리빌트 다운로드 전용 (`cli/model.py:41-45`) |
| Mac에서 결과가 CPU와 미묘하게 다름 | Metal 경로. `--backend cpu`로 비교. 허용 오차는 `test_metal_parity.cpp` 참고 |
| 한글 토큰이 깨져서 스트리밍됨 | 콜백이 토큰 단위로 개별 디코드하므로 멀티바이트가 쪼개질 수 있음. 버퍼링 후 출력 (`complete.cpp:1071-1074`) |
| 응답에 `cloud_handoff: true` | confidence < 임계값. `--no-cloud-handoff` 또는 옵션 `auto_handoff:false` |

다음 장(05)은 이 튜토리얼 이후 어떤 순서로 코드를 파고들지 정리한 학습 로드맵이다.
