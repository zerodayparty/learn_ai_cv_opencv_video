"""평가용 테스트 영상 + 정답(Ground Truth) 생성기.

실제로 촬영한 영상은 "정답 위치"를 사람이 일일이 적어야 하지만,
합성 영상은 물체를 우리가 직접 그리므로 매 프레임 정답을 자동으로 알 수 있다.

생성 결과 (assets/videos/):
    single_move    단일 객체 이동   (왕복 10회 = 테스트 10회)
    two_cross      두 객체 겹침     (교차 10회 = 테스트 10회)
    pause_resume   객체 일시 정지   (정지 5회 = 테스트 5회)
    light_change   조명 변화        (조명 급변 5회 = 테스트 5회)
    target_scene   등록 물체 등장    (특징점 매칭 데모)
그리고 assets/target.png (등록 물체 이미지)

실행: uv run python -m tools.make_test_videos
"""

import json  # 정답 파일 저장
import os  # 폴더 생성

import cv2  # OpenCV
import numpy as np  # 행렬 계산

W, H, FPS = 640, 360, 30  # 영상 너비, 높이, 초당 프레임 수
WARMUP = 30  # 처음 1초는 배경만 보여줘서 배경 모델이 학습할 시간을 준다
OUT_DIR = os.path.join("assets", "videos")  # 영상 저장 폴더
TARGET_PATH = os.path.join("assets", "target.png")  # 등록 물체 이미지 경로


def make_background(w=W, h=H, seed=7):  # 정지된 배경 이미지를 만드는 함수
    """그라데이션 + 무작위 사각형으로 '방 안' 같은 배경을 만든다."""
    rng = np.random.default_rng(seed)  # 항상 같은 배경이 나오도록 씨앗값 고정
    yy, xx = np.mgrid[0:h, 0:w]  # 각 픽셀의 (y, x) 좌표
    base = 90 + 50 * (xx / w) + 30 * (yy / h)  # 왼쪽 위는 어둡고 오른쪽 아래는 밝은 그라데이션
    bg = np.dstack([base * 0.9, base, base * 1.05]).astype(np.uint8)  # 3채널(BGR) 이미지로 만든다
    for _ in range(25):  # 사각형 25개를 그린다(가구, 벽지 무늬 느낌)
        x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))  # 시작점
        x2, y2 = x1 + int(rng.integers(20, 120)), y1 + int(rng.integers(10, 60))  # 끝점
        color = tuple(int(c) for c in rng.integers(60, 180, 3))  # 차분한 색
        cv2.rectangle(bg, (x1, y1), (x2, y2), color, -1)  # 채운 사각형
    return cv2.GaussianBlur(bg, (7, 7), 0)  # 살짝 흐리게 해서 자연스럽게 만든다


def make_person(color, w=44, h=100, seed=0):  # 사람 모양 스프라이트(그림 조각)를 만드는 함수
    """머리(원) + 몸통(타원) 모양 이미지와, 그려질 영역을 나타내는 마스크를 돌려준다."""
    rng = np.random.default_rng(seed)  # 무늬용 난수
    img = np.zeros((h, w, 3), np.uint8)  # 빈 이미지
    mask = np.zeros((h, w), np.uint8)  # 빈 마스크
    cv2.circle(mask, (w // 2, 14), 12, 255, -1)  # 머리 영역
    cv2.ellipse(mask, (w // 2, 60), (w // 2 - 1, 38), 0, 0, 360, 255, -1)  # 몸통 영역
    img[:] = color  # 전체를 기본 색으로 칠한다
    for _ in range(12):  # 옷 무늬(작은 점 12개)
        cx, cy = int(rng.integers(4, w - 4)), int(rng.integers(30, h - 4))  # 점 위치
        shade = tuple(int(max(0, c - 60)) for c in color)  # 조금 어두운 색
        cv2.circle(img, (cx, cy), 3, shade, -1)  # 점을 그린다
    return img, mask  # 이미지와 마스크를 돌려준다


def make_target(seed=3):  # 등록 물체(포스터 같은 평면 물체) 이미지를 만드는 함수
    """특징점이 많이 나오도록 글자, 체커보드, 도형을 섞은 200x150 이미지."""
    rng = np.random.default_rng(seed)  # 항상 같은 그림이 나오도록 고정
    img = np.full((150, 200, 3), 235, np.uint8)  # 밝은 회색 바탕
    for i in range(4):  # 체커보드 가로 4칸
        for j in range(4):  # 체커보드 세로 4칸
            if (i + j) % 2 == 0:  # 번갈아 가며
                cv2.rectangle(img, (10 + i * 15, 10 + j * 15), (25 + i * 15, 25 + j * 15), (20, 20, 20), -1)  # 검은 칸
    for _ in range(14):  # 무작위 도형 14개
        x, y = int(rng.integers(80, 190)), int(rng.integers(10, 140))  # 위치
        color = tuple(int(c) for c in rng.integers(0, 200, 3))  # 색
        if rng.random() < 0.5:  # 절반은 원
            cv2.circle(img, (x, y), int(rng.integers(4, 14)), color, -1)  # 원
        else:  # 나머지는 삼각형
            pts = np.array([[x, y], [x + int(rng.integers(8, 20)), y + 4], [x + 3, y + int(rng.integers(8, 20))]], np.int32)  # 꼭짓점 3개
            cv2.fillPoly(img, [pts], color)  # 삼각형
    cv2.putText(img, "CV", (12, 120), cv2.FONT_HERSHEY_DUPLEX, 1.6, (180, 30, 30), 3)  # 큰 글자
    cv2.putText(img, "TARGET-26", (85, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1)  # 작은 글자
    cv2.rectangle(img, (2, 2), (197, 147), (0, 0, 0), 2)  # 테두리
    return img  # 이미지를 돌려준다


def paste(frame, sprite, mask, x, y):  # 스프라이트를 프레임 위에 붙이는 함수
    """(x, y)를 왼쪽 위로 해서 mask가 255인 곳만 덮어쓴다. 화면 밖은 잘라낸다."""
    h, w = mask.shape  # 스프라이트 크기
    x, y = int(round(x)), int(round(y))  # 정수 좌표
    fx1, fy1 = max(0, x), max(0, y)  # 프레임 쪽 시작점(화면 밖이면 0으로)
    fx2, fy2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)  # 프레임 쪽 끝점
    if fx1 >= fx2 or fy1 >= fy2:  # 겹치는 부분이 없으면
        return  # 아무것도 안 한다
    sx1, sy1 = fx1 - x, fy1 - y  # 스프라이트 쪽 시작점
    region = frame[fy1:fy2, fx1:fx2]  # 덮어쓸 프레임 영역
    m = mask[sy1:sy1 + (fy2 - fy1), sx1:sx1 + (fx2 - fx1)] > 0  # 해당 마스크
    region[m] = sprite[sy1:sy1 + (fy2 - fy1), sx1:sx1 + (fx2 - fx1)][m]  # 마스크 영역만 복사한다


def mask_bbox(mask, x, y):  # 화면에 실제로 그려진 영역의 바운딩 박스(정답 박스)
    """마스크를 (x, y)에 놓았을 때 화면 안에 보이는 부분의 박스를 돌려준다."""
    full = np.zeros((H, W), np.uint8)  # 화면 크기의 빈 마스크
    paste(full, mask, mask, x, y)  # 마스크 자체를 화면 위치에 붙인다(흑백끼리 복사)
    ys, xs = np.nonzero(full)  # 칠해진 픽셀 좌표
    if len(xs) == 0:  # 화면 밖이면
        return None, full  # 박스 없음
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)], full  # 박스와 마스크


def ping_pong(t, period, x_min, x_max):  # 좌우 왕복 위치를 계산하는 함수
    """t 프레임일 때 x_min ↔ x_max 를 period 프레임마다 한 번씩 이동하는 위치."""
    k, r = divmod(t, period)  # 몇 번째 이동인지(k), 그 안에서 몇 프레임째인지(r)
    s = r / period  # 0~1 사이 진행률
    return x_min + (x_max - x_min) * (s if k % 2 == 0 else 1 - s)  # 짝수 번째는 오른쪽, 홀수 번째는 왼쪽으로


def render(name, condition, n_frames, objects_at, light_at=None, events=None, extra=None):  # 영상과 정답을 만들어 저장하는 공통 함수
    """objects_at(f) -> [(gt_id, sprite, mask, x, y), ...] (나중 원소가 앞에 그려짐)."""
    os.makedirs(OUT_DIR, exist_ok=True)  # 저장 폴더를 만든다
    bg = make_background()  # 배경
    rng = np.random.default_rng(42)  # 카메라 노이즈용 난수
    writer = cv2.VideoWriter(os.path.join(OUT_DIR, f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))  # mp4 저장기
    frames_gt = []  # 프레임별 정답
    for f in range(n_frames):  # 프레임마다
        frame = bg.copy()  # 배경을 복사한다
        objs = objects_at(f) if f >= WARMUP else []  # 준비 시간 이후에만 물체를 그린다
        masks = []  # 각 물체의 화면 마스크(가림 비율 계산용)
        gt = []  # 이번 프레임 정답
        for gid, sprite, mask, x, y in objs:  # 물체마다(뒤에서 앞 순서)
            paste(frame, sprite, mask, x, y)  # 그린다
            box, full = mask_bbox(mask, x, y)  # 정답 박스
            masks.append((gid, box, full))  # 기록한다
        for i, (gid, box, full) in enumerate(masks):  # 각 물체의 가림 비율 계산
            if box is None:  # 화면 밖이면
                continue  # 건너뛴다
            covered = np.zeros_like(full)  # 앞 물체들이 덮은 영역
            for _, _, other in masks[i + 1:]:  # 나보다 나중(앞)에 그려진 물체들
                covered |= other  # 덮은 영역을 합친다
            area = int(np.count_nonzero(full))  # 내 넓이
            occ = float(np.count_nonzero(full & covered)) / area if area else 0.0  # 가려진 비율
            gt.append({"id": gid, "bbox": box, "occlusion": round(occ, 3)})  # 정답 기록
        factor = light_at(f) if light_at else 1.0  # 이번 프레임 조명 밝기 배율
        if factor != 1.0:  # 조명이 바뀌었으면
            frame = cv2.convertScaleAbs(frame, alpha=factor, beta=0)  # 화면 전체 밝기를 바꾼다
        noise = rng.normal(0, 3.0, frame.shape)  # 카메라 센서 노이즈(표준편차 3)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)  # 노이즈를 더한다
        writer.write(frame)  # 프레임 저장
        frames_gt.append(gt)  # 정답 저장
    writer.release()  # 영상 파일 마무리
    meta = {"name": name, "condition": condition, "fps": FPS, "width": W, "height": H,  # 영상 정보
            "frames": frames_gt, "events": events or []}  # 프레임 정답과 테스트 이벤트 구간
    if extra:  # 추가 정보가 있으면
        meta.update(extra)  # 합친다
    with open(os.path.join(OUT_DIR, f"{name}.json"), "w", encoding="utf-8") as fp:  # 정답 파일을 연다
        json.dump(meta, fp)  # 저장한다
    print(f"[OK] {name}: {n_frames} frames ({n_frames / FPS:.1f}s), events={len(meta['events'])}")  # 완료 메시지


def main():  # 모든 영상을 만든다
    os.makedirs("assets", exist_ok=True)  # assets 폴더를 만든다
    cv2.imwrite(TARGET_PATH, make_target())  # 등록 물체 이미지 저장
    blue, blue_m = make_person((200, 120, 40), seed=1)  # 파란 옷 사람
    orange, orange_m = make_person((40, 140, 230), seed=2)  # 주황 옷 사람
    period = 75  # 한 번 가로지르는 데 75프레임(2.5초)
    x_lo, x_hi = 40, W - 40 - 44  # 왕복 범위(화면 안에서만)

    n = WARMUP + period * 10  # 준비 1초 + 이동 10회 = 780프레임(26초)
    ev = [{"index": i + 1, "start": WARMUP + i * period, "end": WARMUP + (i + 1) * period - 1, "label": f"traverse #{i + 1}"} for i in range(10)]  # 이벤트 구간
    render("single_move", "단일 객체 이동", n,  # 단일 객체 이동 영상
           lambda f: [(1, blue, blue_m, ping_pong(f - WARMUP, period, x_lo, x_hi), 150)], events=ev)  # 한 사람이 왕복

    ev = [{"index": i + 1, "start": WARMUP + i * period, "end": WARMUP + (i + 1) * period - 1, "label": f"crossing #{i + 1}"} for i in range(10)]  # 교차 이벤트 구간
    render("two_cross", "두 객체 겹침", n,  # 두 객체 겹침 영상
           lambda f: [(1, blue, blue_m, ping_pong(f - WARMUP, period, x_lo, x_hi), 140),  # 파란 사람: 왼→오 (뒤쪽)
                      (2, orange, orange_m, ping_pong(f - WARMUP, period, x_hi, x_lo), 150)], events=ev)  # 주황 사람: 오→왼 (앞쪽)

    move, stop = 60, 90  # 60프레임 이동, 90프레임(3초) 정지
    waypoints = [40, 150, 260, 370, 480, 556]  # 멈추는 위치들

    def pause_obj(f):  # 일시 정지 시나리오의 물체 위치
        t = f - WARMUP  # 준비 시간을 뺀 시간
        k, r = divmod(t, move + stop)  # k번째 구간, 구간 안 r프레임째
        k = min(k, 4)  # 마지막 구간 이후에는 마지막 위치 유지
        x0, x1 = waypoints[k], waypoints[k + 1]  # 이번 구간 출발/도착
        x = x0 + (x1 - x0) * min(r, move) / move if t < 5 * (move + stop) else x1  # 이동 중이면 보간, 아니면 도착 위치
        return [(1, blue, blue_m, x, 150)]  # 물체 1개

    n_p = WARMUP + 5 * (move + stop)  # 780프레임
    ev = [{"index": i + 1, "start": WARMUP + i * (move + stop), "end": WARMUP + (i + 1) * (move + stop) - 1, "label": f"stop #{i + 1} (frames {WARMUP + i * (move + stop) + move}~{WARMUP + (i + 1) * (move + stop) - 1})"} for i in range(5)]  # 정지 이벤트 구간
    render("pause_resume", "객체 일시 정지", n_p, pause_obj, events=ev)  # 일시 정지 영상

    levels = [1.0, 1.45, 0.75, 1.35, 0.65, 1.0]  # 조명 밝기 단계(1.0 = 원래 밝기)
    jumps = [WARMUP + 75 + i * 150 for i in range(5)]  # 조명이 바뀌는 프레임

    def light(f):  # 프레임별 밝기 배율
        return levels[sum(f >= j for j in jumps)]  # 지나간 급변 횟수만큼 단계를 올린다

    ev = [{"index": i + 1, "start": WARMUP + i * 150, "end": WARMUP + (i + 1) * 150 - 1, "label": f"light x{levels[i]}→x{levels[i + 1]} @frame {jumps[i]}"} for i in range(5)]  # 조명 이벤트 구간
    render("light_change", "조명 변화", WARMUP + 750,  # 조명 변화 영상
           lambda f: [(1, blue, blue_m, ping_pong(f - WARMUP, period, x_lo, x_hi), 150)], light_at=light, events=ev)  # 사람 1명 + 조명 급변

    target = make_target()  # 등록 물체
    t_mask = np.full(target.shape[:2], 255, np.uint8)  # 물체 전체가 보이는 마스크

    def target_obj(f):  # 등록 물체 + 방해용 사람
        t = f - WARMUP  # 준비 시간을 뺀 시간
        objs = [(1, orange, orange_m, ping_pong(t, 90, x_lo, x_hi), 200)]  # 사람 1명이 계속 왕복
        if 60 <= t < 600:  # 2초~20초 사이에만 등록 물체가 등장
            objs.insert(0, (99, target, t_mask, -200 + (t - 60) * 1.5, 40))  # 왼쪽 밖에서 오른쪽으로 천천히 이동
        return objs  # 물체 목록

    render("target_scene", "등록 물체 등장", WARMUP + 720, target_obj,  # 특징점 매칭 데모 영상
           extra={"target_id": 99})  # 정답에서 등록 물체 ID


if __name__ == "__main__":  # 직접 실행했을 때만
    main()  # 영상 생성
