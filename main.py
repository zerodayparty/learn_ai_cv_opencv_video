"""실시간 영상 분석 애플리케이션 (움직임 감지 + 객체 추적 + 특정 물체 재인식).

실행 예시:
    uv run python main.py                                  # 웹캠(0번) 사용
    uv run python main.py --source assets/videos/two_cross.mp4 --kalman --matcher hungarian
    uv run python main.py --source assets/videos/target_scene.mp4 --target assets/target.png

키보드:
    q = 종료, s = 현재 프레임 저장(스냅샷), p = 일시정지/재개, m = 마스크 창 켜기/끄기
마우스:
    왼쪽 드래그 = 침입 감지 구역 지정, 오른쪽 클릭 = 구역 삭제
"""

import argparse  # 명령줄 옵션 처리
import os  # 폴더 생성
import time  # FPS 계산용 시간 측정
from datetime import datetime  # 스냅샷 파일 이름용 시각

import cv2  # OpenCV
import numpy as np  # 행렬 계산

from intrusion_zone import IntrusionZone  # 침입 감지 구역
from motion_detector import MotionDetector  # 움직임 감지
from target_matcher import TargetMatcher  # 특징점 매칭
from tracker import Tracker  # 객체 추적

WINDOW = "Video Tracking"  # 메인 창 이름
MASK_WINDOW = "Foreground Mask"  # 마스크 창 이름
GREEN = (0, 255, 0)  # 초록색(BGR 순서: 파랑, 초록, 빨강)
RED = (0, 0, 255)  # 빨간색
CYAN = (255, 255, 0)  # 하늘색
WHITE = (255, 255, 255)  # 흰색
YELLOW = (0, 255, 255)  # 노란색


def parse_args(argv=None):  # 명령줄 옵션을 읽는 함수
    """사용 가능한 옵션을 정의하고 읽어서 돌려준다."""
    p = argparse.ArgumentParser(description="OpenCV motion detection & multi-object tracking")  # 옵션 파서 생성
    p.add_argument("--source", default="0", help="웹캠 번호(0) 또는 비디오 파일 경로")  # 입력 소스
    p.add_argument("--target", default=None, help="재인식할 물체 이미지 경로(선택)")  # 등록 물체 이미지
    p.add_argument("--feature", choices=["orb", "sift"], default="orb", help="특징점 방식")  # ORB / SIFT
    p.add_argument("--match-every", type=int, default=1, help="N 프레임마다 특징점 매칭(속도 조절)")  # 매칭 주기
    p.add_argument("--method", choices=["mog2", "absdiff"], default="mog2", help="배경 차분 방식")  # 배경 차분 방식
    p.add_argument("--lr", type=float, default=-1.0, help="배경 학습률 (-1 = 자동)")  # 학습률
    p.add_argument("--min-area", type=int, default=400, help="무시할 최소 덩어리 넓이(픽셀)")  # 면적 필터
    p.add_argument("--metric", choices=["distance", "iou"], default="distance", help="매칭 기준")  # 거리 / IoU
    p.add_argument("--matcher", choices=["greedy", "hungarian"], default="greedy", help="매칭 방법")  # greedy / 헝가리안
    p.add_argument("--kalman", action="store_true", help="칼만 필터 예측 사용")  # 칼만 필터
    p.add_argument("--max-distance", type=float, default=80.0, help="같은 객체로 볼 최대 중심 거리(픽셀)")  # 거리 임계값
    p.add_argument("--max-disappeared", type=int, default=15, help="이 프레임 수 넘게 안 보이면 ID 삭제")  # ID 삭제 기준
    p.add_argument("--zone", default=None, help="침입 구역 x1,y1,x2,y2 (마우스로도 지정 가능)")  # 침입 구역
    p.add_argument("--resize-width", type=int, default=960, help="이 너비보다 크면 줄여서 처리(0 = 원본)")  # 처리 해상도
    p.add_argument("--show-mask", action="store_true", help="전경 마스크 창 표시")  # 마스크 창
    p.add_argument("--headless", action="store_true", help="창 없이 실행(테스트/서버용)")  # 창 없이 실행
    p.add_argument("--save-video", default=None, help="결과 영상을 저장할 경로(.mp4)")  # 결과 영상 저장
    p.add_argument("--max-frames", type=int, default=0, help="처리할 최대 프레임 수(0 = 끝까지)")  # 프레임 제한
    p.add_argument("--snapshot-dir", default="outputs/snapshots", help="스냅샷 저장 폴더")  # 스냅샷 폴더
    return p.parse_args(argv)  # 옵션을 읽어서 돌려준다


def open_source(source):  # 영상 입력을 여는 함수
    """숫자면 웹캠, 아니면 파일 경로로 연다."""
    src = int(source) if source.isdigit() else source  # "0" 같은 숫자는 웹캠 번호로 바꾼다
    cap = cv2.VideoCapture(src)  # 비디오 캡처 객체를 만든다
    if not cap.isOpened():  # 열기에 실패하면
        raise SystemExit(f"[ERROR] cannot open video source: {source}")  # 오류 메시지를 출력하고 종료한다
    return cap  # 캡처 객체를 돌려준다


def draw_text(img, text, org, color=WHITE, scale=0.6, thickness=2):  # 배경 상자가 있는 글자를 쓰는 함수
    """글자가 잘 보이도록 검은 상자를 깔고 그 위에 글자를 쓴다."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)  # 글자 크기를 잰다
    x, y = org  # 글자 위치(왼쪽 아래 기준)
    cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + base), (0, 0, 0), -1)  # 검은 배경 상자를 그린다
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)  # 글자를 쓴다


def draw_tracks(frame, tracks):  # 추적 결과(박스, ID, 궤적)를 그리는 함수
    """초록 박스 + ID 라벨 + 빨간 궤적선 + 중심점을 그린다."""
    for t in tracks:  # 보이는 객체마다
        x, y, w, h = t.bbox  # 박스 좌표
        cv2.rectangle(frame, (x, y), (x + w, y + h), GREEN, 2)  # 초록색 박스
        draw_text(frame, f"ID:{t.id}", (x, max(18, y - 6)), GREEN, 0.55, 2)  # 박스 위에 ID 라벨
        if len(t.trail) >= 2:  # 궤적 점이 2개 이상이면
            pts = np.array(t.trail, dtype=np.int32).reshape(-1, 1, 2)  # 선 그리기용 배열로 바꾼다
            cv2.polylines(frame, [pts], False, RED, 2, cv2.LINE_AA)  # 빨간 궤적선
        cx, cy = int(t.centroid[0]), int(t.centroid[1])  # 중심점 정수 좌표
        cv2.circle(frame, (cx, cy), 4, RED, -1)  # 중심점 표시


def save_snapshot(frame, folder):  # 현재 화면을 이미지로 저장하는 함수
    """folder/snapshot_날짜_시각.png 로 저장하고 경로를 돌려준다."""
    os.makedirs(folder, exist_ok=True)  # 폴더가 없으면 만든다
    path = os.path.join(folder, f"snapshot_{datetime.now():%Y%m%d_%H%M%S_%f}.png")  # 파일 이름을 정한다
    cv2.imwrite(path, frame)  # 이미지를 저장한다
    return path  # 저장 경로를 돌려준다


def main(argv=None):  # 프로그램 시작 함수
    """영상 읽기 → 감지 → 추적 → 매칭 → 화면 표시를 반복한다."""
    args = parse_args(argv)  # 옵션을 읽는다
    cap = open_source(args.source)  # 영상을 연다
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # 원본 FPS(모르면 30)

    detector = MotionDetector(method=args.method, learning_rate=args.lr, min_area=args.min_area)  # 움직임 감지기
    tracker = Tracker(  # 추적기
        max_disappeared=args.max_disappeared,  # ID 삭제 기준
        max_distance=args.max_distance,  # 거리 임계값
        metric=args.metric,  # 매칭 기준
        matcher=args.matcher,  # 매칭 방법
        use_kalman=args.kalman,  # 칼만 필터 사용 여부
    )
    matcher = TargetMatcher(args.target, method=args.feature) if args.target else None  # 등록 물체가 있으면 매처 생성
    zone_rect = [int(v) for v in args.zone.split(",")] if args.zone else None  # 구역 문자열을 숫자 4개로 바꾼다
    zone = IntrusionZone(zone_rect)  # 침입 감지 구역

    if not args.headless:  # 창을 띄우는 모드면
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)  # 메인 창을 만든다
        cv2.setMouseCallback(WINDOW, zone.on_mouse)  # 마우스 이벤트를 구역 클래스에 연결한다

    writer = None  # 결과 영상 저장기(처음 프레임에서 생성)
    show_mask = args.show_mask  # 마스크 창 표시 여부
    paused = False  # 일시정지 상태
    fps = 0.0  # 화면에 표시할 FPS(지수 이동 평균)
    prev_time = time.perf_counter()  # 직전 프레임 처리 시각
    frame_index = 0  # 처리한 프레임 번호
    target_frames = 0  # 물체를 찾은 프레임 수
    last_match = None  # 마지막 매칭 결과
    display = None  # 마지막으로 그린 화면
    start = time.perf_counter()  # 전체 시작 시각

    print("[INFO] q=quit  s=snapshot  p=pause  m=mask  | mouse drag=set zone, right click=clear zone")  # 사용법 안내

    while True:  # 영상이 끝나거나 q를 누를 때까지 반복
        if not paused:  # 일시정지가 아니면 새 프레임을 처리한다
            ok, frame = cap.read()  # 프레임 하나를 읽는다
            if not ok:  # 더 이상 읽을 프레임이 없으면
                break  # 반복을 끝낸다
            if args.resize_width and frame.shape[1] > args.resize_width:  # 너무 큰 영상이면
                scale = args.resize_width / frame.shape[1]  # 줄일 비율
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)  # 크기를 줄인다

            mask, detections = detector.apply(frame)  # 움직임 감지
            bboxes = [d[0] for d in detections]  # 박스만 모은다
            centroids = [d[1] for d in detections]  # 중심점만 모은다
            tracker.update(bboxes, centroids)  # 추적기 갱신(ID 매칭)
            visible = tracker.visible_tracks()  # 이번 프레임에 보인 객체들

            if matcher is not None and frame_index % max(1, args.match_every) == 0:  # 매칭할 차례면
                last_match = matcher.match(frame)  # 특징점 매칭
            if last_match is not None and last_match.found:  # 물체를 찾았으면
                target_frames += 1  # 찾은 프레임 수 증가

            inside = zone.check(visible, frame_index)  # 침입 구역 확인

            now = time.perf_counter()  # 현재 시각
            inst = 1.0 / max(now - prev_time, 1e-6)  # 이번 프레임의 순간 FPS
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst  # 값이 출렁이지 않게 부드럽게 평균낸다
            prev_time = now  # 시각 저장

            display = frame.copy()  # 원본은 두고 복사본에 그린다
            draw_tracks(display, visible)  # 박스, ID, 궤적
            zone.draw(display)  # 침입 구역
            draw_text(display, f"FPS: {fps:.1f}", (10, 28), YELLOW, 0.7, 2)  # FPS 표시
            info = f"{args.method} lr={args.lr:g} | {args.metric}/{args.matcher}{' +kalman' if args.kalman else ''} | objs={len(visible)}"  # 현재 설정 요약
            draw_text(display, info, (10, 56), WHITE, 0.5, 1)  # 설정 요약 표시
            if detector.last_foreground_ratio > 0.4:  # 화면 대부분이 전경으로 잡히면(조명 급변 의심)
                draw_text(display, "WARNING: sudden global change (lighting?)", (10, 84), RED, 0.55, 2)  # 경고 표시
            if last_match is not None and last_match.found:  # 물체를 찾았으면
                cv2.polylines(display, [np.int32(last_match.corners)], True, CYAN, 3, cv2.LINE_AA)  # 물체 위치를 사각형으로 표시
                draw_text(display, f"[TARGET DETECTED] Target Found ({last_match.inliers} inliers)", (10, display.shape[0] - 40), CYAN, 0.6, 2)  # 발견 메시지
            if inside:  # 구역 안에 객체가 있으면
                draw_text(display, f"INTRUSION! ID {', '.join(map(str, inside))}", (display.shape[1] - 260, 28), RED, 0.6, 2)  # 침입 경고
            draw_text(display, "Press 'q' to quit, 's' to save snapshot, 'p' to pause", (10, display.shape[0] - 12), WHITE, 0.5, 1)  # 조작법 안내

            if args.save_video:  # 결과 영상을 저장하는 경우
                if writer is None:  # 저장기가 아직 없으면
                    out_dir = os.path.dirname(args.save_video)  # 저장 폴더
                    if out_dir:  # 폴더 경로가 있으면
                        os.makedirs(out_dir, exist_ok=True)  # 폴더를 만든다
                    h, w = display.shape[:2]  # 영상 크기
                    writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))  # mp4 저장기를 만든다
                writer.write(display)  # 프레임을 저장한다

            frame_index += 1  # 프레임 번호 증가
            if args.max_frames and frame_index >= args.max_frames:  # 최대 프레임에 도달하면
                break  # 끝낸다

        if args.headless:  # 창 없는 모드면
            continue  # 키 입력 없이 다음 프레임으로

        if display is not None:  # 그릴 화면이 있으면
            shown = display.copy()  # 화면 복사
            zone.draw(shown)  # 드래그 중인 구역도 보이게 다시 그린다
            if paused:  # 일시정지 중이면
                draw_text(shown, "PAUSED", (shown.shape[1] // 2 - 50, shown.shape[0] // 2), YELLOW, 1.0, 2)  # 일시정지 표시
            cv2.imshow(WINDOW, shown)  # 메인 창에 표시
        if show_mask and not paused:  # 마스크 창을 켰으면
            cv2.imshow(MASK_WINDOW, mask)  # 마스크 표시
        key = cv2.waitKey(30 if paused else 1) & 0xFF  # 키 입력을 기다린다(일시정지 중엔 CPU 덜 쓰게 30ms)
        if key == ord("q"):  # q = 종료
            break  # 반복을 끝낸다
        if key == ord("s") and display is not None:  # s = 스냅샷 저장
            print(f"[INFO] snapshot saved: {save_snapshot(display, args.snapshot_dir)}")  # 저장하고 경로를 알려준다
        if key == ord("p"):  # p = 일시정지/재개
            paused = not paused  # 상태를 뒤집는다
            prev_time = time.perf_counter()  # 재개 시 FPS가 튀지 않도록 시각을 다시 잡는다
        if key == ord("m"):  # m = 마스크 창 켜기/끄기
            show_mask = not show_mask  # 상태를 뒤집는다
            if not show_mask:  # 끄는 경우
                cv2.destroyWindow(MASK_WINDOW)  # 마스크 창을 닫는다

    elapsed = time.perf_counter() - start  # 전체 걸린 시간
    cap.release()  # 영상 입력을 닫는다
    if writer is not None:  # 저장기가 있으면
        writer.release()  # 파일을 마무리한다
    if not args.headless:  # 창 모드였으면
        cv2.destroyAllWindows()  # 창을 모두 닫는다
    print(f"[SUMMARY] frames={frame_index} avg_fps={frame_index / max(elapsed, 1e-6):.1f} "  # 처리 프레임 수와 평균 FPS
          f"ids_created={tracker.next_id - 1} target_detected_frames={target_frames} "  # 생성된 ID 수와 물체 발견 프레임 수
          f"intrusions={zone.total_events}")  # 침입 횟수


if __name__ == "__main__":  # 이 파일을 직접 실행했을 때만
    main()  # main 함수를 실행한다
