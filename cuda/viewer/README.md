# mytetration 실시간 미리보기 뷰어 (`cuda/viewer/`)

실행: `cuda\run_viewer.cmd`를 더블클릭하거나, `cuda`에서 `.venv\Scripts\python.exe -m viewer --interactive [--gpu N]`를 실행한다. 창이 바로 뜬다.
- GPU는 nvidia-smi 번호로 고른다(`--gpu N` 또는 환경변수 `MYTET_GPU`, 기본 0). 목록에 없는 번호는 창이나 CUDA보다 먼저 이유 한 줄과 exit 2로 거부한다.
- 사람이 직접 여는 대화식은 유휴·전체 화면 검사 없이 바로 연다. 고른 GPU의 여유가 2,048 MiB 이상인지만 보고, 모자라면 이유 한 줄을 내고 exit 3으로 끝난다.
- 시험·측정·녹화 경로(`--script`, `launch.py`, `measure.py`, `make_media.py`, 시험)는 아래 §5의 보호 규칙을 쓴다.
- 왼쪽 버튼으로 끌면 이동하고, 휠로 확대·축소한다. 창을 닫으면 끝난다.
- 숫자키 1–4 = 무늬가 깊이 이어지는 위치 m37 · m191 · m59 · m60b 의 eps 1e-15 화면으로 바로 간다. 0 = 처음 전체 보기(0, 0, 5). 값은 `viewer/presets.json`(80자리 중심 문자열)에 있다(`viewer/presets.py`).
- 창 오른쪽 위 한 줄(HUD): 앞쪽은 프리셋 이름(끌어 옮기면 «· 이동함»이 붙는다) 또는 안내 «1~4: 무늬 위치, 0: 처음으로», 뒤쪽은 늘 지금 깊이 «eps 3.5e-16 · N 549» 다.
  - eps = 입력이 가리키는 화면의 eps(유효숫자 2자리), N = 그 그림의 반복 횟수 max_iter(`deep/limits.py`).
  - eps ≥ 1e-13 은 N 500, 그 아래는 N = 500 + round(20·log10(1e-13/eps)). 휠 한 칸에 0 또는 1씩 오른다(1e-15 · 1e-20 · 1e-30 · 1e-40 에서 540 · 640 · 840 · 1040).
  - 대본 이벤트 `{"type": "key", "key": "0".."4"}`, 대본의 `"hud": true` 로 HUD 를 켠다(대본 기본은 꺼짐).
- eps < 1e-13 은 깊은 확대(§7)로 그린다. 하한은 1e-40 이다. 그보다 깊이 휠을 돌리면 확대가 멈추고 그림 위 띠에 안내가 뜬다.

대본 재생: `python -m viewer --script viewer/scripts/D1.json --mode timed|verify|media|callbacks --log <폴더>`
- `callbacks`는 대본 이벤트를 뷰어가 GLFW에 등록한 마우스 콜백(GLFW에서 다시 읽어 낸 것)으로 넘긴다. 대화식과 같은 입력 경로·같은 루프다. OS 입력 주입은 아니다.

## 1. 렌더 대상과 그림 규칙 (고정)
- **렌더 대상** = 1920×1080 RGBA8 텍스처(FBO)다.
  - 계산 격자는 원본 16by9_H 규칙과 같다: n = 1920, x 1920점, y 1080점, eps_y = eps·(9/16).
  - 픽셀 하나가 격자 점 하나다.
- **그림 규칙**: 렌더 대상 픽셀 (px, py)는 맵 `map[px, py]`가 참(발산)이면 흰색 (255,255,255,255), 거짓(수렴)이면 검은색 (0,0,0,255)이다.
  - px = 0…1919는 왼쪽에서 오른쪽, py = 0…1079는 아래에서 위다.
  - 원본 `imshow(map.T, origin='lower')`, 흑→백 색표와 같은 방향·색이다.
  - `glReadPixels`는 아래 줄부터 읽으므로, 읽은 배열 R[py, px] = 위 규칙이다.
- **창**은 960×540이다. 렌더 대상을 최근접 이웃으로 1/2 줄인다.
  - 창 픽셀 (u, v)(아래에서 위) = 렌더 대상 픽셀 (2u, 2v). 표본 위치는 이것으로 고정이다(셰이더 `texelFetch(rt, ivec2(gl_FragCoord.xy)*2)`).
- 코드: `viewer/render.py`. 기준 색칠(시험 쪽, 셰이더와 독립): `tests/viewer_check.py`의 `picture`, `window_of`.

## 2. 이동·확대 식 (고정, 코드 한 곳 = `viewer/params.py`)
- **드래그**: 렌더 대상 픽셀 Δx(오른쪽 +), Δy(아래쪽 +)만큼 그림이 커서를 따라 움직인다.
  - `x0 ← x0 − Δx·(2·eps/1919)`
  - `y0 ← y0 + Δy·(2·eps_y/1079)`, eps_y = eps·(9/16)
  - 창에서 마우스로 끌 때는 창 픽셀 이동 × 2가 Δ다.
- **휠**: s칸(확대 +, 축소 −, 정밀 휠의 분수 칸 허용). 고정점은 창 중앙이라 x0, y0는 그대로다.
  - s > 0: `eps ← eps·(0.9 ** s)`
  - s < 0: `eps ← eps·((1/0.9) ** (−s))`
  - 1칸 확대 = ×0.9, 1칸 축소 = ×1/0.9.
- 시험은 대본 파일과 이 식만으로 매개변수를 따로 계산한다.
  - 식은 `tests/viewer_check.contract_step`에 이 글에서 다시 옮겨 적었다.
  - `tests/test_viewer.py::test_formulas`가 `params.py`를 그 식에 비트까지 묶는다.
  - 뷰어의 상태는 읽지 않는다.

## 3. 대본 (`viewer/scripts/`, 생성기 `viewer/script.py`)
- D1 장면 (0, 0, 50) · Z1 장면 (−0.43138, 0.895038, 0.1). 두 장면 모두 16:9, 1920×1080. 깊은 확대 장면 DZa · DZb 는 `viewer/scripts/deepzoom_scenes.json`.
- `<장면>_manip.json`(아래 10초 조작), `<장면>_scroll.json`(10초 동안 1/6칸 휠 600개), `<장면>_one_notch.json`(1 s 쉼 → 휠 한 칸 → 1 s 쉼).
- 깊은 확대 대본: `perturb_*.json`(깊이별 · 경계 넘기), `deepzoom2_*.json`(m37 등에서 움직임·멈춤·1e-13 넘기), `presets_window.json`(숫자키 창 시험).
- 0~5 s 드래그만: 시작점을 지나는 반지름 0.5·eps 원(중심 (x0−0.5eps, y0))을 2바퀴 돈다. 320개, 64개/s.
  - 드래그는 정수 렌더 대상 픽셀이다. 원 위 점의 목표 오프셋을 정수로 반올림하고 그 차이를 쓰므로, 합이 0이 되어 끝 = 시작이다.
  - 끄는 거리는 약 6,029 px(화면 너비 × 3.14).
- 5~7.5 s 휠 확대만: 22칸을 1/7칸 × 154개로(61.6개/s). eps → eps·0.9²² < eps/10.
- 7.5~10 s 휠 축소만: −1/7칸 × 154개.
- 합계 628개, 62.8개/s, 최대 간격 16.2 ms. 끝 매개변수 = 시작(10⁻¹² 안) — `script.check`와 시험이 확인한다.
- 정밀 동치 시험용 짧은 대본 3곳(`P_left_zoom`, `P_deep`, `P_negreal`): 움직임 뒤 2.5 s 멈추고 입력 하나를 더 넣는다(빠른 모드 복귀 시험).
- `P_drag_multi`: 마우스 콜백 경로 시험용(누른 채 여러 번 이동, 놓은 뒤 이동, 다시 누름).

## 4. 모드와 시간
- **빠른 모드** = 빠른 커널(`viewer/fast_kernel.cu`), max_iter 500, escape 1e10, 같은 16by9_H 격자. 정밀도는 화면마다 고른다(`viewer/fast.py::choose_precision`): 좌표 간격이 넉넉한 화면은 float32(제품 알고리즘을 단정도로, 격자는 double로 만든 뒤 float32로 바꿈), 그렇지 않으면 double(f64) 커널이다. 프레임마다 `precision` 에 남긴다.
- **정밀 모드** = `mytet_cuda.divergence_map("16by9_H", x0, y0, eps, 1920, 500, 1e10)`(double).
- **정지 판정 지연** = 마지막 입력 뒤 0.25 s 동안 입력이 없으면 정밀 프레임을 계산한다.
- **전환 시간 상한 = 2.0 s**(마지막 입력 이벤트 → 정밀 프레임 표시).
- 휠 확대는 로그 공간에서 부드럽게 따라간다(시상수 0.06 s, 목표에 가까우면 목표 값 그대로). 식: `viewer/params.py`, 독립 재작성: `tests/viewer_check.contract_smooth`.
- 계산이 화면 주기보다 느려지면 하이브리드 표시(`viewer/hybrid.py`)로 바뀐다: 최근 새 그림 계산 시간의 중앙값이 33 ms 를 넘으면 마지막 그림을 지금 좌표로 옮겨 그리고(warped), 새 그림은 따로 계산해 도착하는 대로 바꾼다. 28 ms 이하로 내려오면 매 프레임 새로 계산하는 모드로 돌아간다.
- 빠른 프레임은 새 입력이 반영됐을 때, 휠 확대가 목표로 따라가는 동안, 그리고 마지막 입력 뒤 0.25 s(정지 판정 지연)가 지나기 전에 그린다. 옮겨 그리기 모드에서는 새 그림 계산이 진행 중일 때도 그린다. 그래서 매개변수가 같은 빠른 프레임이 정지 판정 전까지 이어질 수 있다.
  - 매 프레임 새로 계산하는 모드에서는 그리는 빠른 프레임마다 1920×1080 전체를 그 화면 매개변수로 새로 계산한다. 앞 프레임의 맵·중간 결과를 다시 쓰지 않는다(컴파일한 커널만 캐시한다).
  - 옮겨 그리기 모드에서는 계산이 끝난 그림을 지금 좌표로 옮겨 그리고, 새 그림 계산은 따로 돈다(위 하이브리드 표시). 정밀 프레임은 진행 중인 새 그림 계산이 끝난 뒤에 계산한다.
  - 시작 화면이 eps ≥ 1e-13 이면 처음에 float32·double 빠른 커널을 한 번씩 돌려 데운다. 빠른 함수 호출 수는 대본 재생 동안만 센다. 그 전의 호출(데우기와 첫 정밀 프레임)은 `meta.json` 의 `warmup_calls` 에 따로 적는다.
- **표시(present) 시각** = `glfwSwapBuffers` → `glFinish` → `DwmFlush`(다음 DWM 합성까지 기다림)가 돌아온 때(`time.perf_counter`).
  - 그래서 모든 프레임이 합성 한 번씩을 받는다. 다음 프레임은 앞 프레임이 합성된 뒤에야 바뀐다. fps 상한은 화면 주사율이다.
  - swap interval은 0이다.
- **프레임 시간** = 센 프레임의 연속 표시 시각 간격. **fps** = 센 프레임 수 ÷ (마지막 표시 − 첫 입력 전달 시각).
- **지연** = 표시 시각 − 그 프레임에 새로 반영된 첫 입력의 전달 시각(기록만).
- 대본 재생은 OS 입력 주입 없이 뷰어의 입력 처리 함수에 이벤트를 넘긴다. timed·verify·media 재생은 마우스 콜백을 등록하지 않아 실제 마우스 입력을 받지 않는다. callbacks 재생은 콜백을 등록하므로 창을 포인터에서 떨어진 곳에 연다(§5 끝).

## 5. 데스크톱 보호 (Windows)
- 창은 숨긴 채 만든 뒤 `GLFW_FOCUS_ON_SHOW = false`로 보인다(활성화 없음).
  - 대본 재생에서는 `SetWindowPos(HWND_TOPMOST, SWP_NOACTIVATE)`로, 포커스 없이 맨 위에 둔다. 그래야 픽셀이 화면에 나온다.
  - 보인 직후 앞 창이 뷰어가 아님을 기록한다.
- 대본 재생(시험·측정·녹화)은 창이나 GPU 문맥을 만들기 전에 다음을 확인한다. `viewer/launch.py`가 보고, 뷰어 진입점 `app.preflight`도 `--script`에서 다시 본다. 조건이 안 되면 `launch.py`는 30초마다 다시 보며 기다린다. 시험 중에 PC를 쓰고 있으면 창 시험이 기다리는 것은 이 때문이다.
  - 사용자 유휴 ≥ 5분(GetLastInputInfo)
  - 앞에 전체 화면 앱 없음
  - 고른 GPU 여유 ≥ 2,048 MiB
  - FreeVirtualMemory ≥ 8 GB
- 뷰어 밖 부모 프로세스의 큰 GPU 작업(pytest의 `gpu` 준비, `measure.py`의 분석, `tests/viewer_check.py`의 `check_run`·`check_display`·`quality`)도 시작 전에 FreeVirtualMemory ≥ 8,388,608 KB를 본다(`viewer_check.require_memory`). 모자라면 거부한다(pytest는 exit 3).
- 측정이 끝나면 창을 닫고 프로세스가 끝난다.
- **OpenGL과 CUDA는 같은 GPU여야 한다.**
  - OpenGL 문맥이 어느 GPU에 생기는지는 드라이버가 정한다. GLFW에는 GPU를 고르는 방법이 없다. NVIDIA 제어판의 «OpenGL 렌더링 GPU»가 «자동 선택»이면 보통 화면이 붙은 GPU를 따른다.
  - 그래서 문맥을 만든 직후, 그리기·텍스처·FBO 할당·창 이동보다 먼저 첫 GL 호출로 `GL_RENDERER`를 읽는다. 그 장치 이름(`/` 앞 부분)이 CUDA용으로 고른 GPU의 nvidia-smi 이름과 **정확히 같지 않으면** 문맥을 풀고 창을 지우고 거부한다. 앞부분만 같은 이름(예: «RTX 3090»과 «RTX 3090 Ti»)도 다르다고 본다.
  - 못 막는 것: 다른 GPU에 문맥이 «생기는» 순간 자체는 막지 못한다. 생긴 즉시 지우므로 그 위에서 그리거나 메모리를 잡지는 않는다. 실행마다 `meta.json`의 `renderer`에 실제 렌더러 이름을 남긴다.
  - 이름 비교라서 같은 모델 두 장은 구별하지 못한다. 그때는 OpenGL과 CUDA가 서로 다른 장에 있어도 통과한다.
  - 화면이 붙지 않은 GPU를 고르면 대개 이 확인에서 거부된다. 그때는 화면이 붙은 GPU를 고르거나 제어판에서 OpenGL GPU를 맞춘다.
  - CUDA는 `mytet_cuda.gpu` 규칙(고른 GPU만 보이게 하고 UUID를 다시 확인)을 따른다.
- nvidia-smi가 없거나 실패하면 이유 한 줄(`not opening the window: cannot read the GPU list with nvidia-smi …`)을 내고 exit 3으로 끝난다.
- `callbacks` 재생은 GLFW가 부르는 순서(커서 → 버튼 → 커서 …)로 등록된 콜백을 부른다. 뷰어 내부 상태(`_cursor` 등)는 건드리지 않는다.
  - 첫 누름은 커서 알림 없이 들어간다. 그래서 뷰어가 `glfw.get_cursor_pos`로 누른 위치를 읽는다.
  - 재생 동안 `glfw.get_cursor_pos`는 가상 커서 위치 (100, 40)을 돌려주는 대역으로 바뀌고 호출마다 기록된다(`meta.json`의 `get_cursor_pos_calls`). 시험은 이 기록이 정확히 한 번 `[100, 40]`인지 본다.
  - 그 뒤 누른 채 여러 번 움직이고, 놓은 뒤 움직이고(입력 없음이 맞음), 다시 누른다. 뷰어가 받은 입력 목록은 대본 이벤트와 같아야 한다(`P_drag_multi`).
  - 대본 재생 창은 마우스 포인터가 없는 자리(`desktop.window_pos_away_from_pointer`, 기록 `pointer_at_start`·`window_pos`)에 연다.

## 6. GPU 사용량
판정값 = 프로세스 안 CuPy 풀 `total_bytes` 최댓값 + CUDA 컨텍스트 몫이다. 컨텍스트 몫은 첫 할당 직전·직후 고른 GPU 사용량(nvidia-smi) 차이다. GL 컨텍스트 몫은 따로 적는다. 0.2 s nvidia-smi 추적(`gpu_trace.csv`)은 참고 기록이다.

## 7. 깊은 확대 (eps < 1e-13, 하한 1e-40)
- 원본처럼 float64 격자를 쓰면 eps 가 1e-13 근처에서 격자 간격이 좌표의 반올림 단위(배정밀도 유효숫자 약 16자리)와 비슷해져 이웃 점이 같은 좌표가 된다. 그래서 그림이 네모 블록(픽셀화)으로 뭉개진다.
- 깊은 구간은 섭동(perturbation) 방식으로 계산한다(`deep/`).
  - 화면 중심 근처 기준점 하나의 궤도를 높은 정밀도(mpmath, 설치돼 있으면 gmpy2 를 백엔드로 씀. 깊이 + 30자리)로 계산한다(`deep/reference.py`).
  - 각 화소는 기준과의 작은 차이만 배정밀도로 따라간다(`deep/perturb_kernel.cu`). 차이가 커지거나 기준이 0 에 가까워지면 재기준화한다.
  - 중심 좌표는 문자열(Decimal)로 들고 다니므로 끌기·휠이 반올림 없이 쌓인다(`deep/hp.py`, `viewer/deep_state.py`).
- 계산은 별도 프로세스(`viewer/deep_worker.py`)가 한다. 움직이는 동안은 받은 최신 그림을 지금 좌표로 옮겨 그리고, 멈추면 거친 그림(stride 8) → 1920×1080 최종 그림 순서로 바뀐다.
- 계산할 수 없는 화면은 그림 대신 안내를 띄운다: 하한 너머(beyond_L), 원점 근처(near_zero), 음의 실수축을 가로지르는 화면(branch_cut) 등.
- 정답 대조: `deep/truth.py`(고정밀 직접 계산)와 `tests/test_deepzoom2_engine_*.py`, `tests/test_perturb_*.py`.
