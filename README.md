# 🟩 영상 속 움직이는 사람·물건을 따라가며 표시해주기  

OpenCV만으로(딥러닝 없이) 영상 속 움직이는 객체를 감지하고, ID를 붙여 추적하고, 등록한 물체를 다시 찾아내는 실시간 비디오 분석 애플리케이션.  

배경 차분(MOG2 / absdiff)으로 움직임을 찾고, 모폴로지 연산과 윤곽선으로 객체를 덩어리로 만든다. 직접 구현한 Tracker 클래스가 중심점 거리 또는 IoU로 프레임 간 객체를 짝지어 ID와 이동 궤적을 유지한다. ORB/SIFT 특징점 매칭으로 미리 등록한 물체가 나타나면 "TARGET DETECTED"를 표시한다. 보너스로 NumPy만으로 구현한 Hungarian Algorithm, IoU 매칭, 칼만 필터 예측, 마우스로 지정하는 침입 감지 구역을 포함한다. 합성 테스트 영상과 정답 데이터로 ID Switch, 추적 실패, 특징점 매칭률을 자동 측정하고, 그 결과로 "고전적 컴퓨터 비전의 한계"와 "딥러닝이 필요한 이유"를 분석한 보고서를 함께 제공한다.  

<br><br>

## 🟢 README Contents  
- [주요 기능](#-주요-기능)  
- [폴더 구조](#-폴더-구조)  
- [실행 방법](#-실행-방법)  
- [키보드 / 마우스 조작](#-키보드--마우스-조작)  
- [명령줄 옵션](#-명령줄-옵션)  
- [성능 측정 재현](#-성능-측정-재현)  
- [측정 결과 요약](#-측정-결과-요약)  
- [제약 사항 준수](#-제약-사항-준수)  

<br><br>

## 🟢 주요 기능  

| 기능 | 내용 | 파일 |  
| --- | --- | --- |  
| 영상 입출력 | 웹캠/비디오 파일 입력, FPS 표시, q 종료, s 스냅샷, p 일시정지 | `main.py` |  
| 배경 차분 | MOG2 또는 absdiff + 누적 평균, 학습률 조절, 그림자 제거 | `motion_detector.py` |  
| 객체 감지 | 열림/닫힘 모폴로지, 윤곽선, 면적 필터, 바운딩 박스, 중심점 | `motion_detector.py` |  
| 객체 추적 | ID 부여/유지/삭제, 거리 또는 IoU 매칭, 궤적 저장 | `tracker.py` |  
| 특징점 매칭 | ORB/SIFT + 비율 테스트 + RANSAC 호모그래피, "TARGET DETECTED" | `target_matcher.py` |  
| (보너스) Hungarian | scipy 없이 NumPy만으로 구현, Greedy와 비교 | `hungarian.py` |  
| (보너스) 침입 감지 | 마우스로 구역 지정, 경고 출력, 로그 저장 | `intrusion_zone.py` |  
| (보너스) IoU 매칭 | `--metric iou` | `tracker.py` |  
| (보너스) 칼만 필터 | `cv2.KalmanFilter`로 다음 위치 예측 후 매칭, `--kalman` | `tracker.py` |  
| 정량 평가 | 합성 영상 + 정답으로 ID Switch, 추적 실패, 매칭률 자동 측정 | `tools/` |  

<br><br>

## 🟢 폴더 구조  

```text
learn_ai_cv_opencv_video/
├── main.py                  # 실시간 영상 분석 앱 (진입점)
├── tracker.py               # Tracker 클래스 (ID 관리, 매칭, 칼만 필터)
├── motion_detector.py       # 배경 차분 + 모폴로지 + 윤곽선
├── target_matcher.py        # ORB/SIFT 특징점 매칭
├── hungarian.py             # Hungarian Algorithm (NumPy) + Greedy 매칭
├── intrusion_zone.py        # 침입 감지 구역
├── tools/
│   ├── make_test_videos.py  # 평가용 합성 영상 + 정답(JSON) 생성
│   ├── evaluate.py          # 추적 성능 자동 측정
│   └── feature_benchmark.py # 특징점 매칭 성능 측정
├── assets/
│   ├── target.png           # 등록 물체 이미지
│   └── videos/              # 생성된 테스트 영상 (git 제외, 스크립트로 재생성)
├── docs/
│   ├── requirement.md       # 과제 요구사항
│   └── report.md            # 분석 보고서
└── outputs/                 # 스냅샷, 결과 영상, 침입 로그, 측정 결과 (git 제외)
```

<br><br>

## 🟢 실행 방법  

### 🟡 1. 설치  
- [uv](https://docs.astral.sh/uv/)가 필요하다. Python 3.11 이상.  

```bash
uv sync
```

### 🟡 2. 테스트 영상 생성 (처음 한 번)  

```bash
uv run python -m tools.make_test_videos
```

### 🟡 3. 실행  

웹캠:  

```bash
uv run python main.py
```

두 사람이 교차하는 영상 (칼만 필터 + Hungarian):  

```bash
uv run python main.py --source assets/videos/two_cross.mp4 --kalman --matcher hungarian
```

등록 물체 찾기:  

```bash
uv run python main.py --source assets/videos/target_scene.mp4 --target assets/target.png
```

마스크 창을 같이 보면서 학습률 바꿔보기:  

```bash
uv run python main.py --source assets/videos/pause_resume.mp4 --lr 0.001 --show-mask
```

<br><br>

## 🟢 키보드 / 마우스 조작  

| 입력 | 동작 |  
| --- | --- |  
| `q` | 종료 |  
| `s` | 현재 화면을 `outputs/snapshots/`에 저장 |  
| `p` | 일시정지 / 재개 |  
| `m` | 전경 마스크 창 켜기 / 끄기 |  
| 마우스 왼쪽 드래그 | 침입 감지 구역 지정 |  
| 마우스 오른쪽 클릭 | 침입 감지 구역 삭제 |  

<br><br>

## 🟢 명령줄 옵션  

| 옵션 | 기본값 | 설명 |  
| --- | --- | --- |  
| `--source` | `0` | 웹캠 번호 또는 비디오 파일 경로 |  
| `--method` | `mog2` | 배경 차분 방식 `mog2` / `absdiff` |  
| `--lr` | `-1` | 배경 학습률 (-1 = 자동) |  
| `--min-area` | `400` | 무시할 최소 덩어리 넓이 |  
| `--metric` | `distance` | 매칭 기준 `distance` / `iou` |  
| `--matcher` | `greedy` | 매칭 방법 `greedy` / `hungarian` |  
| `--kalman` | 꺼짐 | 칼만 필터 예측 사용 |  
| `--max-distance` | `80` | 같은 객체로 볼 최대 중심 거리(px) |  
| `--max-disappeared` | `15` | 이 프레임 수 넘게 안 보이면 ID 삭제 |  
| `--target` | 없음 | 재인식할 물체 이미지 |  
| `--feature` | `orb` | 특징점 방식 `orb` / `sift` |  
| `--zone` | 없음 | 침입 구역 `x1,y1,x2,y2` |  
| `--show-mask` | 꺼짐 | 전경 마스크 창 표시 |  
| `--save-video` | 없음 | 결과 영상 저장 경로 |  
| `--headless` | 꺼짐 | 창 없이 실행 |  

<br><br>

## 🟢 성능 측정 재현  

```bash
uv run python -m tools.evaluate
```

```bash
uv run python -m tools.feature_benchmark
```

- 결과는 `outputs/eval_results.md`, `outputs/feature_results.md`에 저장된다. 분석은 [docs/report.md](docs/report.md)에 있다.  

<br><br>

## 🟢 측정 결과 요약  

| 상황 | 테스트 횟수 | ID Switch | 추적 실패 |  
| --- | --- | --- | --- |  
| 단일 객체 이동 | 10 | 0 | 0 |  
| 두 객체 겹침 | 10 | 20 | 0 |  
| 객체 일시 정지 | 5 | 4 | 5 |  
| 조명 변화 | 5 | 1 | 1 |  

- 두 객체 겹침에서 거리 임계값, Hungarian, IoU는 ID Switch를 줄이지 못했다(20 → 20). 칼만 필터만 20 → 4로 줄였다.  
- 합성 영상 기준 수치다. 실제 촬영 영상 측정은 보고서에 추가해야 한다.  

<br><br>

## 🟢 제약 사항 준수  
- YOLO, SSD, R-CNN, OpenCV `dnn` 모듈, MediaPipe를 사용하지 않는다.  
- Hungarian Algorithm은 scipy 없이 NumPy만 사용한다.  
- Tracker 클래스는 별도 파일 `tracker.py`에 있다.  
