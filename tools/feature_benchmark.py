"""특징점 매칭 성능 측정 도구 (ORB / SIFT).

등록 물체(assets/target.png)를 배경 위에 회전·가림·축소·어둡게 해서 붙인 뒤
추출된 특징점 수, 매칭된 특징점 수, 매칭률을 표로 만든다.
그리고 target_scene.mp4 에서 프레임별 탐지율도 측정한다.

실행: uv run python -m tools.feature_benchmark     (결과: outputs/feature_results.md)
"""

import json  # 정답 파일 읽기
import os  # 경로 처리

import cv2  # OpenCV
import numpy as np  # 행렬 계산

from target_matcher import TargetMatcher  # 특징점 매칭기
from tools.make_test_videos import TARGET_PATH, make_background  # 같은 배경/등록 이미지 사용


def make_scene(target, angle=0.0, occlude=0.0, scale=1.0, light=1.0):  # 시험용 장면을 만드는 함수
    """배경 가운데에 등록 물체를 변형해서 붙인 640x360 이미지를 돌려준다."""
    scene = make_background()  # 배경
    obj = target.copy()  # 원본은 두고 복사
    if occlude > 0:  # 가림 조건이면
        w = obj.shape[1]  # 물체 너비
        x0 = int(w * (1 - occlude))  # 오른쪽에서부터 occlude 비율만큼
        obj[:, x0:] = (90, 90, 90)  # 회색 판으로 덮는다(앞에 다른 물건이 있는 상황)
    if scale != 1.0:  # 크기 조건이면
        obj = cv2.resize(obj, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)  # 크기를 바꾼다
    h, w = obj.shape[:2]  # 물체 크기
    diag = int(np.ceil(np.hypot(w, h)))  # 회전해도 잘리지 않을 정사각형 크기
    canvas = np.zeros((diag, diag, 3), np.uint8)  # 회전용 캔버스
    mask = np.zeros((diag, diag), np.uint8)  # 물체 영역 마스크
    ox, oy = (diag - w) // 2, (diag - h) // 2  # 캔버스 가운데 위치
    canvas[oy:oy + h, ox:ox + w] = obj  # 물체를 가운데 놓는다
    mask[oy:oy + h, ox:ox + w] = 255  # 마스크도 같은 위치
    M = cv2.getRotationMatrix2D((diag / 2, diag / 2), angle, 1.0)  # 회전 행렬
    canvas = cv2.warpAffine(canvas, M, (diag, diag))  # 물체 회전
    mask = cv2.warpAffine(mask, M, (diag, diag), flags=cv2.INTER_NEAREST)  # 마스크 회전
    sx, sy = (scene.shape[1] - diag) // 2, (scene.shape[0] - diag) // 2  # 장면 가운데 위치
    roi = scene[sy:sy + diag, sx:sx + diag]  # 붙일 영역
    roi[mask > 0] = canvas[mask > 0]  # 물체 부분만 덮어쓴다
    if light != 1.0:  # 조명 조건이면
        scene = cv2.convertScaleAbs(scene, alpha=light, beta=0)  # 전체 밝기 변경
    return scene  # 장면을 돌려준다


def main():  # 측정을 실행하고 표를 저장한다
    target = cv2.imread(TARGET_PATH)  # 등록 이미지
    if target is None:  # 없으면
        raise SystemExit("run `uv run python -m tools.make_test_videos` first")  # 먼저 생성하라고 안내
    conditions = [  # (표시 이름, 변형 설정)
        ("정면", {}),  # 변형 없음
        ("30도 회전", {"angle": 30}),  # 30도 회전
        ("60도 회전", {"angle": 60}),  # 60도 회전
        ("30% 가림", {"occlude": 0.3}),  # 30% 가림
        ("50% 가림", {"occlude": 0.5}),  # 50% 가림
        ("(추가) 50% 축소", {"scale": 0.5}),  # 크기 절반
        ("(추가) 조명 50%", {"light": 0.5}),  # 밝기 절반
        ("(추가) 60도 회전 + 50% 가림", {"angle": 60, "occlude": 0.5}),  # 복합 조건
    ]
    out = []  # 결과 문서 줄
    os.makedirs("outputs", exist_ok=True)  # 출력 폴더
    for method in ["orb", "sift"]:  # 두 방식
        matcher = TargetMatcher(target, method=method)  # 매칭기
        rows = []  # 표 데이터
        for label, kw in conditions:  # 조건마다
            scene = make_scene(target, **kw)  # 장면 생성
            r = matcher.match(scene)  # 매칭
            rows.append(f"| {label} | {r.target_keypoints} | {r.good_matches} | {r.match_rate:.1%} | {r.inliers} | {'O' if r.found else 'X'} |")  # 한 줄
            if method == "orb":  # 확인용 이미지는 ORB 기준으로 한 번만 저장
                cv2.imwrite(os.path.join("outputs", f"feature_scene_{len(rows)}.png"), scene)  # 장면 저장
        out += [f"## {method.upper()}", "",  # 제목
                "| 조건 | 추출된 특징점 수 | 매칭된 특징점 수 | 매칭률 | RANSAC 인라이어 | 탐지 |",  # 머리줄
                "| --- | --- | --- | --- | --- | --- |"] + rows + [""]  # 구분줄과 데이터

    with open(os.path.join("assets", "videos", "target_scene.json"), encoding="utf-8") as fp:  # 영상 정답
        gt = json.load(fp)  # 읽기
    rows = []  # 영상 탐지율 표
    for method in ["orb", "sift"]:  # 두 방식
        matcher = TargetMatcher(TARGET_PATH, method=method)  # 매칭기
        cap = cv2.VideoCapture(os.path.join("assets", "videos", "target_scene.mp4"))  # 영상 열기
        tp = fn = partial_total = partial_found = absent_found = 0  # 찾음 / 놓침 / 일부만 보인 프레임 / 일부만 보일 때 찾음 / 없는데 찾음(오탐)
        for frame_gt in gt["frames"]:  # 프레임마다
            ok, frame = cap.read()  # 프레임 읽기
            if not ok:  # 끝이면
                break  # 종료
            vis = [g for g in frame_gt if g["id"] == gt["target_id"]]  # 등록 물체 정답
            ratio = vis[0]["bbox"][2] * vis[0]["bbox"][3] / (200 * 150) if vis else 0.0  # 화면에 보이는 물체 비율
            found = matcher.match(frame).found  # 매칭 결과
            if ratio >= 0.5:  # 절반 이상 보이면 "보임"
                tp += found  # 보이는데 찾음
                fn += not found  # 보이는데 놓침
            elif ratio > 0:  # 화면 가장자리에 일부만 걸쳐 있으면
                partial_total += 1  # 부분 노출 프레임 수
                partial_found += found  # 부분 노출인데 찾음
            else:  # 물체가 화면에 전혀 없으면
                absent_found += found  # 없는데 찾았다고 함 = 진짜 오탐
        cap.release()  # 영상 닫기
        rows.append(f"| {method.upper()} | {tp + fn} | {tp} | {tp / max(tp + fn, 1):.1%} | {partial_found}/{partial_total} | {absent_found} |")  # 한 줄
    out += ["## target_scene.mp4 프레임별 탐지율", "",  # 제목
            "| 방식 | 물체 50% 이상 노출 프레임 | 탐지 성공 | 탐지율 | 50% 미만 부분 노출 시 탐지 | 오탐(화면에 없는데 탐지) |",  # 머리줄
            "| --- | --- | --- | --- | --- | --- |"] + rows + [""]  # 데이터

    path = os.path.join("outputs", "feature_results.md")  # 출력 파일
    with open(path, "w", encoding="utf-8") as f:  # 파일 열기
        f.write("\n".join(out))  # 저장
    print("\n".join(out))  # 화면 출력
    print(f"[OK] saved: {path}")  # 저장 위치 안내


if __name__ == "__main__":  # 직접 실행했을 때만
    main()  # 측정 실행
