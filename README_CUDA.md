# mytetration CUDA — GPU로 같은 그림을 빠르게, 1080p 실시간 미리보기, 1e-40 까지 깊은 확대

- 출처: [DMTPARK/mytetration](https://github.com/DMTPARK/mytetration), 커밋 `21b26204e874b2d5b8e6237f1941b5f0b9fd3418`(2024-05-27).
- 원본 파일(`Power Tower Fractal (static)/` 등)은 저장소 뿌리에 그대로 두었다. 원본 파일 가운데 `README.md` 말고는 바꾼 것이 없다.
  - 이 묶음(`README_CUDA.md` + `cuda/`)은 새 파일만 더한다. 묶음을 얹는 것만으로는 원본 파일이 하나도 바뀌지 않는다.
  - 포크 저장소처럼 `README.md` 맨 위에 포크 소개 절이 붙어 있어도, 원본 README 본문은 그 절 아래에 그대로 이어진다(`cuda/tests/test_upstream.py` 가 확인한다).
- 이 파일 말고는 전부 `cuda/` 아래에 있다:
  - `mytet_cuda/`: CUDA 커널과 라이브러리, 정적 그림 명령줄(원본과 같은 매개변수·파일 이름)
  - `viewer/`: 실시간 미리보기 뷰어(숫자키 위치 프리셋 · 깊이 표시 HUD)
  - `deep/`: 깊은 확대(eps < 1e-13, 하한 1e-40) — 섭동 방식
  - `scripts/`, `bench/`: 시험이 쓰는 측정·분석 도구
  - `tests/`, `reference/`, `tester/`: 원본과 같은 결과인지 보는 시험
  - `run_viewer.cmd`, `requirements.txt`, `make_reference_data.py`

## 무엇이 빨라졌나

| 항목 | 원본(CPU, NumPy) | 이 구현(GPU) |
|---|---|---|
| D1 기본 화면(1by1_H, n=500) | 72.89 s | 0.0276 s |
| Z1 확대 화면(1by1_H, n=500, eps 0.1) | 102.12 s | 0.0409 s |
| D1 4K(n=3840) | 4,338.6 s(추정) | 1.3971 s |
| Z1 4K(n=3840) | 5,988.6 s(추정) | 2.1055 s |
| 1080p 실시간 미리보기(끌기·휠 조작 중) | – | **62.5~62.9 fps** |
| 손을 뗀 뒤 정밀(double) 그림으로 바뀌기까지 | – | **0.45~0.57 s** |
| 깊은 확대 1e-40 새 정밀 그림 한 장(위치 4곳) | – | **485.6~622.8 ms** |

측정 조건과 읽는 법:
- 측정 장비: GPU NVIDIA GeForce RTX 3090 Ti, Windows 11, Python 3.12, NumPy 2.5.4, 드라이버 617.14.
- 원본: 원본 스크립트의 `compute_tetration_divergence` 함수를 그대로 실행했다. 2회 중 두 번째 값이다.
- 이 구현: `mytet_cuda.divergence_map`을 4회 돌려 첫 회를 버린 중앙값이다. 시간은 맵 전체가 호스트 메모리에 올 때까지다. 정밀도는 원본과 같은 float64(complex128)다.
- 4K 원본 시간은 120행만 계산해 전체 행 수로 환산한 «추정»이다.
- 실시간 미리보기:
  - 뷰어 첫 판(화면마다 고르는 정밀도 · 하이브리드 표시 · 깊은 확대를 더하기 전)에서 D1·Z1 장면으로 잰 값이다. 그때는 조작 중 그림이 늘 float32 빠른 모드였다. 지금 판으로 다시 재지는 않았다.
  - 1920×1080 계산 + 960×540 창 표시다.
  - fps는 대본 입력 속도(초당 62.8개)에 묶인 값이다. 계산만 보면 1080p 한 장에 약 7 ms다.
  - 정밀 전환 시간은 마지막 입력부터 double 그림이 화면에 나올 때까지다. 정지 판정 0.25 s가 들어 있다.
- 깊은 확대: 위치 4곳(m37 · m191 · m59 · m60b), eps 1e-40, 1920×1080, N 1040. `deep_map`을 5회 새로 불러 첫 회를 버린 중앙값이다. 기준 궤도 계산 포함, 캐시 없음.
- 숫자는 모두 위 장비 한 대에서 잰 값이다. 다른 GPU·CPU에서는 달라진다.

## 원본과 같은 결과인가
- 원본 static 8종의 기본값, 확대 2곳, 작은 세트 3곳, 모두 13세트를 원본 계산(CPU)과 픽셀 단위로 비교했다.
- 원본과 다른 픽셀:
  - 13세트 합계 1,306개다. Z1 766(0.31 %), Z2 373(0.15 %), D4 105, D3 48이고, 나머지는 0~3개다. D6·P1·P2·P3은 0이다.
  - 다른 픽셀은 **전부** 판정이 반올림 경계에 걸린 경우임을 증명한다. 궤적을 한 걸음씩 원본 연산(NumPy 스칼라 `c ** z`)과 비교하고, 오차가 정해진 한계(τ) 안인지 본다. 규칙은 `cuda/reference/SETS.md`에 있다.
- 명령줄로 만든 PNG는 «원본 맵에서 증명된 픽셀만 뒤집은 맵»을 원본 그림 코드로 그린 PNG와 디코드 픽셀까지 같다. 파일 이름 규칙도 원본과 같다.
- 커널은 원본이 쓰는 NumPy 2.5.4의 복소 거듭제곱(`npy_cpow` → Windows UCRT `cpow`)과 `np.abs`(AVX2/FMA3 경로)를 흉내 낸다.
  - 다른 NumPy 판이나 다른 CPU 경로에서는 원본 계산의 마지막 비트가 달라질 수 있다. 그러면 다른 픽셀 수가 바뀔 수 있다.
  - `requirements.txt`에서 NumPy 판을 고정한 이유다.

## 요구 사항
- NVIDIA GPU와 CUDA 12를 지원하는 드라이버. 시험한 GPU는 RTX 3090 Ti(compute capability 8.6)다. CUDA 툴킷은 따로 설치하지 않아도 된다(`nvidia-cuda-nvrtc-cu12`·`nvidia-cuda-runtime-cu12` 휠을 쓴다). `nvidia-smi`가 실행돼야 한다.
- Python 3.12와 `cuda/requirements.txt`(쓴 판 그대로 고정).
- **Windows**: 뷰어는 Windows 전용이다(Win32 API, DWM). 시험과 기준 데이터 생성도 Windows에서만 확인했다(PowerShell로 메모리를 읽는다).
- **리눅스: 확인하지 않았다.** 라이브러리와 명령줄(`mytet_cuda`)에는 Windows 전용 코드가 없다. 다만 원본과의 비트 일치는 Windows UCRT 기준이라, 리눅스에서는 원본 계산 자체가 달라질 수 있다.

## 의존성과 라이선스
`cuda/requirements.txt` 의 판 그대로다. 라이선스는 설치된 패키지 메타데이터(License-Expression, 없으면 License 분류자)에서 옮겼다.

| 패키지 | 판 | 라이선스 | 쓰는 곳 |
|---|---|---|---|
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | 전부(원본과 비트 일치에 판 고정) |
| cupy-cuda12x | 14.2.0 | MIT | GPU 계산 |
| cuda-pathfinder | 1.8.3 | Apache-2.0 | CuPy 가 CUDA 라이브러리를 찾음 |
| nvidia-cuda-nvrtc-cu12 | 12.9.86 | LicenseRef-NVIDIA-Proprietary | 커널 컴파일 |
| nvidia-cuda-runtime-cu12 | 12.9.79 | LicenseRef-NVIDIA-Proprietary | CUDA 런타임 |
| mpmath | 1.3.0 | BSD | 깊은 확대의 고정밀 기준 궤도·정답 |
| gmpy2 | 2.3.2 | LGPL-3.0-or-later | mpmath 의 빠른 백엔드 |
| glfw | 2.10.2 | MIT | 뷰어 창 |
| PyOpenGL | 3.1.10 | BSD License | 뷰어 그리기 |
| pillow | 12.3.0 | MIT-CMU | PNG·HUD 글씨 |
| matplotlib | 3.11.2 | Python Software Foundation License | 원본과 같은 PNG |
| pytest | 9.1.1 | MIT | 시험 |
| contourpy 1.4.0 · fonttools 4.67.0 · cycler 0.12.1 · kiwisolver 1.5.1 · pyparsing 3.3.3 · python-dateutil 2.9.0.post0 · six 1.17.0 · packaging 26.3 | – | BSD-3-Clause · MIT · BSD License · BSD License · MIT · BSD License; Apache Software License · MIT · Apache-2.0 OR BSD-2-Clause | matplotlib 이 끌어옴 |
| colorama 0.4.6 · iniconfig 2.3.1 · pluggy 1.7.0 · pygments 2.21.0 | – | BSD License · MIT · MIT · BSD-2-Clause | pytest 가 끌어옴 |

## 설치
```bat
cd cuda
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```
Windows에서 `core.autocrlf=true`로 clone하면 원본 `.py`가 CRLF로 바뀐다. 시험과 기준 데이터 생성은 CRLF를 LF로 되돌려 읽으므로 그대로 써도 된다.

## 쓰는 법
**GPU 고르기**
- `--gpu N` 또는 환경변수 `MYTET_GPU=N`으로 고른다. N은 `nvidia-smi`의 번호이고, 기본은 0이다.
- 목록에 없는 번호(예: 7, −1, 글자)는 CUDA를 띄우기 전에 이유 한 줄과 0이 아닌 종료 코드로 거부한다.
- 고른 GPU의 여유 메모리가 2,048 MiB보다 적으면 GPU 작업을 시작하지 않는다.

**정적 그림** (원본 8종과 같은 매개변수, 같은 PNG 이름)
```bat
cd cuda
.venv\Scripts\python -m mytet_cuda 16by9_H
.venv\Scripts\python -m mytet_cuda 1by1_H --x0 -0.43138 --y0 0.895038 --eps 0.1 --gpu 0 --out-dir out
```
- 변형: `1by1_H 1by1_R 16by9_H 16by9_R 9by16_H 9by16_R 4by5_H 4by5_R`
- 매개변수: `--x0 --y0 --eps --n --max-iter --escape-radius`. 값은 원본 파일 맨 위 상수처럼 파이썬 숫자 그대로 읽는다(`--eps 5e1`이면 이름이 `..._eps_50.0.png`).
- 주지 않은 값은 그 원본 파일의 기본값이다. `--save-map`을 주면 `divergence_map.npy`도 남긴다.
- 라이브러리: `from mytet_cuda import divergence_map` → `divergence_map("16by9_H", x0, y0, eps, n, max_iter, escape_radius)`(bool 배열, 원본과 같은 배치).

**실시간 미리보기 뷰어**
- `cuda\run_viewer.cmd`를 더블클릭하거나, `.venv\Scripts\python -m viewer --interactive --gpu N`을 실행한다.
- 왼쪽 버튼으로 끌면 이동, 휠로 확대·축소(한 칸 = ×0.9), 창을 닫으면 끝난다.
- 숫자키:
  - `1`–`4`: 무늬가 깊이 이어지는 위치 4곳(m37 · m191 · m59 · m60b)의 eps 1e-15 화면으로 바로 간다. 거기서 휠로 더 들어가면 된다.
  - `0`: 처음 전체 보기(0, 0, 5)로 돌아간다.
- 창 오른쪽 위 한 줄(HUD) 읽는 법. 예: «m37 · 이동함 · eps 3.5e-16 · N 549»
  - 앞쪽: 마지막으로 누른 프리셋 이름. 끌어 옮기면 «· 이동함»이 붙는다. 프리셋을 안 눌렀으면 «1~4: 무늬 위치, 0: 처음으로».
  - `eps`: 지금 화면의 eps(유효숫자 2자리). 화면 가로 반폭이 eps 다. 작을수록 깊다.
  - `N`: 그 화면 그림의 반복 횟수(max_iter). eps ≥ 1e-13 에서 500, 그 아래는 500 + round(20·log10(1e-13/eps)) 로 휠 한 칸에 0 또는 1씩 오른다(1e-40 에서 1040).
- 깊이 한계: **eps 1e-40**. 그보다 깊이 휠을 돌리면 확대가 멈추고 그림 위에 안내가 뜬다.
- 조작 중에는 빠른 그림을 그린다.
  - 좌표 간격이 넉넉한 화면은 float32, 그렇지 않은 화면은 double 커널이다(`viewer/fast.py`).
  - 계산이 화면 주기보다 느려지면 마지막 그림을 지금 좌표로 옮겨 그리고, 새 그림은 따로 계산해 오는 대로 바꾼다(하이브리드 표시, `viewer/hybrid.py`).
  - eps ≥ 1e-13 에서는 0.25 s 멈추면 double(원본과 같은 계산)로 다시 그린다. eps < 1e-13 은 깊은 확대(섭동)로 그린다(아래 «한계»).
- OpenGL 창은 CUDA용으로 고른 GPU에서 열려야 한다.
  - 창을 만든 직후 `GL_RENDERER`의 장치 이름이 고른 GPU의 이름과 정확히 같은지 본다. 다르면 창을 바로 닫고 거부한다.
  - 보통은 화면이 붙은 GPU를 고르면 된다.
  - 이름으로만 비교하므로, 같은 모델의 GPU가 두 장 있으면 OpenGL과 CUDA가 서로 다른 장에 있어도 이 확인을 통과한다.
- 자세한 규칙: `cuda/viewer/README.md`

## 시험
```bat
cd cuda
.venv\Scripts\python make_reference_data.py
set MYTET_GPU=0
.venv\Scripts\python -m pytest
```
- `make_reference_data.py`는 원본 스크립트를 그대로 돌려 기준 맵을 만든다(CPU만, 8프로세스로 10~15분). 시험 전에 한 번 돌린다.
- 시험은 `MYTET_GPU`(기본 0)의 GPU에서 돈다. 다른 GPU에서 돌리려면 그 번호를 넣는다.
- 일부 시험은 따로 둔 측정 기록이 없으면 «not included»로 건너뛴다(이유가 표시된다).
- `scripts/`·`bench/` 의 측정 도구는 시험이 함수로 불러 쓴다. 측정 도구를 직접 돌릴 때 기본 입력(측정 기록·위치 목록 등)은 이 묶음에 없으므로 입력 파일을 인자로 넘겨야 한다.
- 창을 띄우는 깊은 확대·프리셋 창 시험은 켤 때만 돈다: `set MYTET_DEEPZOOM2_WINDOW=1`, `set MYTET_PRESETS_WINDOW=1`.
- 창을 띄우는 시험(`tests/test_viewer.py`와 위 두 창 시험)에는 보호 규칙이 있다. 사람이 쓰는 PC를 방해하지 않으려는 것이다.
  - 사용자가 5분 이상 입력이 없고, 앞에 전체 화면 앱이 없고, FreeVirtualMemory가 8 GB 이상일 때만 창을 띄운다.
  - 그때까지 30초마다 다시 보며 기다린다(최대 1시간).
  - 포커스는 빼앗지 않는다.
  - PC를 쓰면서 시험하려면 `--ignore=tests/test_viewer.py`로 이 시험만 뺀다(두 창 시험은 켜지 않으면 돌지 않는다).

## 한계
- **조작 중의 빠른 그림은 원본과 다르다.** 뷰어 첫 판(조작 중 늘 float32)에서 빠른 그림을 정밀(double) 그림과 비교하면:
  - 넓은 장면(D1): 평균 0.11 %, 최대 0.53 %
  - 확대 장면(Z1): 평균 6.46 %, 최대 8.02 %
  - 멈춘 뒤의 정밀 그림은 double로 원본과 같은 계산이다.
- **깊은 확대(eps < 1e-13)는 뷰어에서만 된다. 하한은 1e-40 이다.**
  - 원본처럼 float64 격자(`np.linspace`)를 쓰면, 격자 간격이 좌표의 반올림 단위(배정밀도 약 16자리)에 가까워져 픽셀이 뭉개진다. 원본 README 의 «1e-13 을 넘으면 픽셀화»가 이것이다.
  - 뷰어는 eps < 1e-13 에서 섭동 방식(`deep/`)으로 바꾼다. 기준점 하나의 궤도를 mpmath 로 높은 정밀도로 계산하고, 각 화소는 기준과의 작은 차이만 배정밀도로 따라간다. 정적 그림 명령줄(`mytet_cuda`)은 원본과 같이 float64 그대로다.
  - 깊은 구간 그림은 1e-40 에서 한 장에 약 0.5–0.6 s 걸린다. 움직이는 동안은 마지막 그림을 옮겨 그리고, 멈추면 다시 계산한다.
  - 원점 근처, 음의 실수축을 가로지르는 화면 등 계산할 수 없는 곳은 그림 대신 안내를 띄운다.
- **움직이는 동안의 «옮긴 그림»은 블록·화질 시험을 넘지 못한다.** 새 그림이 오기 전까지 마지막 그림을 옮겨 그리는 동안(하이브리드 표시), 확대한 장면의 창 시험 6 개(`tests/test_viewer.py` 의 `test_deepzoom_blocks` manip-DZa · manip-DZb · scroll-Z1 · scroll-DZa · scroll-DZb 와 `test_hybrid_dzb_manip_blocks`)가 블록 판정(프레임마다 블록 ≤ 2 %)과 화질 판정(정밀 그림과의 차이 평균 ≤ 10 % · 최대 ≤ 15 %)을 넘는다. 이 6 개는 «예상된 실패»(xfail)로 표시했고, 다른 판정이 실패하거나, 값이 전에 잰 최댓값의 1.25 배를 넘거나, 통과하면 실패로 센다. 고치려면 표시 방식을 새로 설계해야 한다. 멈추면 정밀 그림으로 다시 그린다.
- **문턱에 붙은 창 시험 2 개는 실행마다 결과가 바뀐다.** `tests/test_viewer.py` 의 `test_hybrid_dzb_scroll_blocks`(옮긴 그림 재생의 프레임별 블록 ≤ 2 % — 잰 최댓값 0.0182–0.0205)와 `test_hybrid_dzb_manip_replacement_gap`(새 그림 교체 간격 ≤ 0.25 s — 약 0.253 s 간격이 한 실행에 최대 3 번)이다. 이 2 개는 엄격하지 않은 «예상된 실패»(xfail)로 표시했다: 나쁜 프레임이 1 개 이하이고 그 비율이 전에 잰 최댓값의 1.25 배 이하일 때, 또는 0.25 s 를 넘는 간격이 3 개 이하이고 각각 0.30 s 이하일 때만 예상된 실패이고, 그 밖은 실패로 센다. 통과(xpassed)는 허용한다.
- **반복 횟수 경계의 남은 변화**: 화면 대부분이 거의 같은 탈출 번호를 갖는 자리에서는 휠 한 칸에 화면의 최대 약 11 % 가 바뀌는 곳이 남는다(N 이 정수이고 판정이 발산/수렴 둘뿐이라서).
- 원본과 같은 double 계산(`mytet_cuda`)으로는 1080p를 실시간(30 fps 이상)으로 그리지 못한다(RTX 3090 Ti에서 약 3.3~3.6 fps). 그래서 조작 중에는 빠른 그림을 쓰고, eps ≥ 1e-13 에서 멈춘 뒤에만 원본과 같은 계산으로 다시 그린다.
- 원본과의 비트 일치는 NumPy 2.5.4 + Windows UCRT + AVX2/FMA3 경로에 묶여 있다.
- 뷰어는 Windows 전용이다. 리눅스는 확인하지 않았다.
- **메모리 누수 후보**: 뷰어에서 화면을 움직이는 동안 새로 그린 그림마다 기록용으로 압축한 사본이 창을 닫을 때까지 남아(`viewer/deferred.py` 의 `MapArchive`), 오래 조작할수록 메모리가 늘어날 수 있다. 코드를 읽어 찾은 후보이고, 실제 뷰어에서 메모리를 재서 확인하지는 않았다.
- 뷰어의 «OpenGL = CUDA GPU» 확인은 장치 이름 비교다. 같은 모델 두 장은 구별하지 못한다.
- zoom-in 영상(`Power Tower Fractal (zoom-in)`)과 `go further/PTF_sqrt2.py`는 GPU로 옮기지 않았다.
