# 02. 멀티플랫폼 지원: 빌드, 타깃, 바인딩, 배포

> 근거 파일: `cactus-*/CMakeLists.txt`, `cactus-*/build.sh`, `apple/`, `android/`, `bindings/`, `python/cactus/cli/compile.py`, `.github/workflows/*.yml`

## 0. 먼저 알아야 할 결론

1. **ARM64 전용**이다. `-march=armv8.2-a+fp16+simd+dotprod+i8mm`(`cactus-kernels/CMakeLists.txt:36`)와 무조건적인 `<arm_neon.h>` include 때문에 x86_64(리눅스, 인텔 맥, 윈도)는 빌드되지 않는다.
2. **Android는 arm64-v8a만** 지원하며 다른 ABI는 CMake가 의도적으로 실패시킨다(`android/CMakeLists.txt:7-8`).
3. **패키지 저장소에 올라가는 것은 PyPI 휠 `cactus-compute`와 Homebrew 포뮬러뿐**이다. SwiftPM, CocoaPods, Maven/AAR, pub.dev, npm, crates.io 배포물은 없다. 모바일 바인딩은 "네이티브 라이브러리를 빌드해서 앱에 복사"하는 방식이다.
4. GPU는 **Apple Metal만**, NPU는 **Apple CoreML/ANE만**(그마저 v2.2.1에서는 비활성) 있다. Android는 CPU 전용이다.

## 1. 세 라이브러리와 빌드 그래프

| 라이브러리 | CMake 타깃 | 산출물 | 링크 |
|---|---|---|---|
| `cactus-kernels/` | `cactus_kernels` (STATIC) | `libcactus_kernels.a` | Apple: Accelerate, Metal, Foundation, MetalPerformanceShaders |
| `cactus-graph/` | `cactus_graph` (STATIC) | `libcactus_graph.a` | `cactus_kernels` (add_subdirectory) |
| `cactus-engine/` | `cactus_engine` (OUTPUT_NAME `cactus_engine_core`) | `libcactus_engine_core.a` → 번들 `libcactus_engine.a` → `libcactus_engine.{dylib,so}` | `cactus_graph`, curl(+mbedtls), Apple: CoreML, Security, SystemConfiguration, CFNetwork |

- 엔진 CMake는 세 아카이브를 **하나의 `libcactus_engine.a`로 합친다**(Apple `libtool -static`, 그 외 `ar -M` MRI 스크립트, `cactus-engine/CMakeLists.txt:115-138`). 최상위 프로젝트로 빌드되면 `-force_load`/`--whole-archive`로 공유 라이브러리도 만든다(:140-171).
- 공통 컴파일 플래그(`cactus_flags` 인터페이스 타깃): C++20, `-O3 -fvisibility=hidden -fno-rtti -ffunction-sections -fdata-sections`, Clang이면 `-flto=thin`. Apple은 `-Wl,-dead_strip`, 그 외 `--gc-sections`, Android는 `--exclude-libs,ALL`.
- Apple 빌드는 `CMAKE_OSX_ARCHITECTURES arm64`를 강제하고 Metal 셰이더 소스를 `cmake/embed_msl.cmake`로 C 헤더에 박아 넣는다(런타임 컴파일, `.metallib` 없음). 비Apple은 `metal_backend_stub.cpp`, `npu.cpp` 스텁을 대신 컴파일한다.
- 버전은 `CACTUS_VERSION` 파일에서 `CACTUS_COMPILE_TIME_VERSION` define으로 들어간다.

### libcurl / mbedtls는 왜 vendoring 되어 있나
클라우드 핸드오프(`cloud.cpp`)와 텔레메트리(`telemetry_impl.cpp`)가 HTTP 클라이언트를 필요로 한다. `cactus-engine/libs/curl/{macos,ios/device,ios/simulator,android/arm64-v8a}/libcurl.a`가 프리빌트로 들어 있고, Android는 TLS를 위해 `android/mbedtls/`도 함께 링크한다. Linux는 시스템 curl(`find_package(CURL QUIET)`)이며, 없으면 CLI 빌드가 거부된다(`python/cactus/cli/compile.py:23-42`).

## 2. 지원 타깃 표

| 타깃 | 지원 | 근거 |
|---|---|---|
| macOS arm64 | 예 (static, dylib, xcframework, PyPI 휠 `macosx_14_0_arm64`, Homebrew) | `apple/build.sh:183`, `pypi.yml:55` |
| iOS 기기 arm64 (최소 16.4) | 예 | `apple/build.sh:2-3, 62-76, 165` |
| iOS 시뮬레이터 arm64 | 예 (x86_64 슬라이스 없음) | `apple/build.sh:89-102, 167` |
| visionOS / watchOS / tvOS / Catalyst | 빌드 타깃 없음 (README 벤치마크에 Vision Pro 행이 있으나 iPad 앱 호환 실행으로 보임) | build 파일에 언급 없음 |
| Android arm64-v8a (최소 API 21, 16KB 페이지 대응 링크) | 예 (`libcactus_engine.so` + `.a`) | `android/build.sh:8,44`, `android/CMakeLists.txt:17-23,43` |
| Android armeabi-v7a / x86 / x86_64 | 아니오 | `android/CMakeLists.txt:7-8` |
| Linux aarch64 (라즈베리파이 5, CI `ubuntu-24.04-arm`) | 예 (manylinux_2_28_aarch64 휠) | `build.yml:40-46`, `pypi.yml:76-88` |
| Linux x86_64, Windows | 아니오 | ARM 전용 플래그, `_WIN32` ifdef 몇 개뿐 |

## 3. 플랫폼별 빌드 경로

### 3.1 macOS / iOS (`apple/`)
```bash
cactus build --apple          # == apple/build.sh
```
- `BUILD_STATIC=true`(기본): `CMAKE_SYSTEM_NAME=iOS`, arm64, iphoneos/iphonesimulator SDK로 `apple/libcactus_engine-{device,simulator}.a` 생성.
- `BUILD_XCFRAMEWORK=true`(기본): CMake Xcode 제너레이터로 동적 `cactus.framework`(식별자 `com.cactuscompute.cactus`, 세 아카이브 `-force_load`)를 만들고 `cactus-ios.xcframework`(ios-arm64, ios-arm64-simulator), `cactus-macos.xcframework`(macos-arm64)로 묶는다. 모듈맵은 `framework module cactus { header "cactus_engine.h" }`.
- iOS 온디바이스 테스트: `cactus test --ios` → `cactus-engine/tests/ios/run.sh`가 simctl/xctrace로 기기를 고르고, `configure_xcode.rb`로 `test_*.cpp`의 `main`을 `<name>_main`으로 바꿔 하나의 앱에 넣고, 모델 번들을 .app에 복사한 뒤 `devicectl`/`simctl`로 실행한다.

### 3.2 Android (`android/`)
```bash
cactus build --android        # == android/build.sh
```
- NDK는 `ANDROID_NDK_HOME` → `$ANDROID_HOME/ndk/*` 최신 → Homebrew 경로 순으로 자동 탐지(버전 고정 없음). 툴체인 파일은 NDK 자체의 `android.toolchain.cmake`.
- `android/cactus_jni.cpp`가 `Java_com_cactus_CactusJNI_*` 심볼(init/complete/prefill/tokenize/transcribe/stream/embed/rag/index/log/telemetry)을 제공하며 `libcactus_engine.so`로 링크된다.
- 온디바이스 테스트: `cactus test --android` → `tests/android/run.sh`가 adb로 기기/에뮬레이터를 고르고 `/data/local/tmp/cactus_{tests,models,assets}`에 푸시해 실행한다.

### 3.3 Linux / macOS 호스트 (`cactus build`, `cactus build --python`)
- 플래그 없음: `cactus-engine/build.sh` 실행 후 `run`(=`tests/run.cpp` 대화형 REPL)과 `transcribe`(SDL2가 있으면 마이크 입력) 바이너리를 `python/cactus/bin/`에 컴파일한다(`compile.py:63-134`).
- `--python`: `libcactus_engine.{dylib,so}`를 만들어 ctypes가 로드하게 한다. ctypes 탐색 순서는 `CACTUS_LIB_PATH` → 휠 내장 `bindings/lib/` → `<repo>/cactus-engine/build/`(`bindings/cactus.py:15-35`).
- `build/test/benchmark/clean`은 git 체크아웃에서만 동작한다(`cli/__init__.py:449-461`).

## 4. 언어 바인딩

`bindings/README.md`가 메커니즘을 한 줄씩 요약한다. 공통점: **모든 바인딩은 C API 위의 얇은 껍데기**이며 네이티브 로직이 없다.

| 바인딩 | 메커니즘 | 표면 API | 플랫폼 | 배포 |
|---|---|---|---|---|
| Swift | Clang 모듈맵(`module cactus { header "cactus_engine.h" }`), `Cactus.swift`는 `@_exported import cactus` 한 줄 | 원시 C 함수 | iOS, macOS | xcframework 또는 `.a` + 모듈맵 수동 링크 |
| Kotlin | JNI(`object CactusJNI`, `System.loadLibrary("cactus_engine")`) + KMP `expect/actual`(iOS는 cinterop `cactus.def`) | `nativeInit/Complete/Transcribe/StreamTranscribe*/Embed/Index*` | Android, iOS(KMP) | `.so`를 `jniLibs/arm64-v8a`에 복사, `.kt` 파일 복사 |
| Flutter | Dart FFI(`DynamicLibrary.open('libcactus_engine.so')` / `process()`) | `cactusInit`, `cactusComplete`, … 톱레벨 함수 | Android, iOS, macOS | `cactus.dart` 복사 + 네이티브 라이브러리 수동 추가 |
| React Native | 구 아키텍처 브리지(`RCT_EXTERN_MODULE`, `ReactContextBaseJavaModule`), TurboModule 아님. 토큰 스트리밍은 `onToken` 이벤트 | `Cactus.init/complete/transcribe/streamTranscribe*/embed/ragQuery/index*` | iOS, Android | 브리지 파일 + Kotlin/Swift 바인딩 복사 (`choose-bindings.md`의 "npm install" 문구는 현재 실체가 없음) |
| Rust | `unsafe extern "C"` 블록, `#[link(name="cactus_engine", kind="static")]` | 모든 C 함수 | macOS, ARM Linux | `cactus.rs` 복사 + `rustc-link-search` |
| Python | ctypes(`bindings/cactus.py`), `Graph`/`Tensor` 클래스 포함 | `cactus_*` 함수 + 그래프 빌더 | macOS arm64, Linux aarch64 | PyPI 또는 `cactus build --python` |

### 바인딩 선택 요령 (`docs/choose-bindings.md` 요약 + 실체 반영)
- 네이티브 iOS 앱 → Swift (Metal 가속 포함). 네이티브 Android → Kotlin JNI. 둘 다 KMP로 공유 가능.
- 크로스플랫폼 UI → Flutter(macOS까지) 또는 RN. 단, 두 경우 모두 네이티브 라이브러리 빌드와 파일 복사가 필요하다.
- 서버/스크립트/프로토타이핑 → Python 또는 CLI(`cactus serve`가 OpenAI 호환 API).
- 게임 엔진·C++ 데스크톱 → 헤더 하나(`cactus_engine.h`)와 `libcactus_engine.a`.

## 5. CI와 릴리스 파이프라인 (`.github/workflows/`)

| 워크플로 | 트리거 | 러너 | 내용 |
|---|---|---|---|
| `build.yml` | 엔진/그래프/커널/apple/android 변경 | `macos-15`, `ubuntu-24.04-arm` | 엔진 빌드, Apple static lib, Android `.so` 컴파일 확인 |
| `cpp.yml` | 커널/그래프 변경 | 위 두 러너 매트릭스 | `cactus-kernels/test.sh`, `cactus-graph/test.sh` |
| `python.yml` | `python/**` 변경 | `ubuntu-24.04-arm` | 엔진 빌드 후 pytest (모델 다운로드가 필요한 테스트 제외) |
| `inference.yml` | 승인된 PR 리뷰 / 수동 | self-hosted linux arm64 | `cactus test --component engine`을 LFM2-VL-450M, whisper-small로 실행 |
| `dco.yml` | PR | ubuntu | 커밋마다 `Signed-off-by` 확인 |
| `docs.yml` | docs 변경 / 릴리스 | ubuntu | mkdocs-material + mike → gh-pages |
| `pypi.yml` | 릴리스에서 호출 | `macos-26`, manylinux aarch64 컨테이너 | `cactus build` → dylib/so를 휠에 동봉, `cactus-code` Node 에이전트 번들, delocate/auditwheel |
| `release.yml` | GitHub Release 발행 | — | PyPI → Homebrew tap(`cactus-compute/homebrew-cactus`) `Formula/cactus.rb` 갱신 → docs |

`brew install cactus-compute/cactus/cactus`의 실체는 "Homebrew virtualenv 안에 `pip install cactus-compute==<ver>`"이며, macOS Sonoma 이상 arm64에서만 동작한다.

## 6. 플랫폼별로 달라지는 런타임 동작 (알아두면 디버깅이 쉬운 것)

| 항목 | Apple | Android | Linux |
|---|---|---|---|
| 기본 백엔드 | Metal 사용 가능하면 Metal(`execute.cpp:17-33`) | CPU | CPU |
| 큰 matmul/attention/conv | Accelerate(cblas/vDSP)로 위임(`matmul.cpp:18-19,637`) | NEON | NEON |
| 스레드 고정 | 없음 | 성능 코어(용량 ≥ 최대의 70%)에 `sched_setaffinity`(`threading.h:204-264`) | 없음 |
| GEMV(디코드) 스레드 | iOS ≤3 / macOS ≤5 | 항상 1 | ≤5 |
| mmap 동작 | 페이지 캐시 공유로 RAM 사용량이 파일보다 훨씬 작음 | 벤더 커널에 따라 anon 복사가 일어나 RAM이 파일 크기에 근접(`blog/lfm2.5_350m.md`) | 일반 리눅스 페이지 캐시 |
| HTTP | vendored curl | vendored curl + mbedtls | 시스템 curl |
| 텔레메트리 저장 위치 | `~/Library/Caches/cactus/telemetry` | 앱 캐시 디렉터리 | `$HOME/Library/Caches/cactus/telemetry` (macOS 경로를 그대로 사용) |

## 7. 지원하지 않는 것 (근거 포함)

- Windows, x86/x86_64 호스트, Intel Mac — ARM 전용 플래그와 헤더, 휠·포뮬러의 arm64 요구
- Android 32bit/x86 ABI — CMake FATAL_ERROR
- iOS x86_64 시뮬레이터, visionOS/watchOS/tvOS/Catalyst 전용 타깃
- Vulkan, OpenCL, CUDA, NNAPI, QNN/Hexagon, MediaTek NeuroPilot, WebGPU, OpenVINO — 코드 자체가 없음(`blog/lfm2.5_350m.md:53-62`에서 이유 설명)
- SwiftPM/CocoaPods/Maven/pub.dev/npm/crates.io 패키지
- RN 신아키텍처(TurboModule)
- 혼합 정밀도 CQ2.54/CQ3.26의 **로컬 변환**(프리빌트 다운로드만, `cli/model.py:41-45`)
- 고정된 NDK 버전, 리포 자체 툴체인 파일

다음 장(03)에서 이런 선택이 다른 엔진들과 어떻게 대비되는지 본다.
