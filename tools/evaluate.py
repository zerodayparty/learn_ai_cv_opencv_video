"""추적 성능 정량 평가 도구.

합성 영상의 정답(Ground Truth)과 추적 결과를 비교해 아래를 자동으로 센다.
    - ID Switch : 정답 객체에 붙은 ID가 바뀌고, 바뀐 ID가 5프레임 이상 유지된 경우 1회
    - 추적 실패 : 정답 객체가 화면에 있는데 10프레임 이상 연속으로 매칭된 ID가 없는 경우 1회
    - 가림 구간 : 50% 이상 가려진 프레임은 추적 실패 계산에서 제외하고 비고에 기록

실행: uv run python -m tools.evaluate            (결과: outputs/eval_results.md)
"""

import json  # 정답 파일 읽기
import os  # 경로 처리
import sys  # 명령줄 인자

import cv2  # OpenCV
import numpy as np  # 행렬 계산

from hungarian import hungarian_assign  # 정답-추적 결과 1:1 매칭용
from motion_detector import MotionDetector  # 움직임 감지
from tracker import Tracker  # 객체 추적

VIDEO_DIR = os.path.join("assets", "videos")  # 테스트 영상 폴더
SWITCH_MIN_FRAMES = 5  # ID Switch 인정 최소 유지 프레임
FAIL_MIN_FRAMES = 10  # 추적 실패 인정 최소 연속 미검출 프레임
OCCLUSION_LIMIT = 0.5  # 이 비율 이상 가려지면 실패 계산에서 제외


def load_gt(name):  # 정답 파일을 읽는 함수
    with open(os.path.join(VIDEO_DIR, f"{name}.json"), encoding="utf-8") as fp:  # 파일을 연다
        return json.load(fp)  # JSON을 파이썬 객체로 바꿔 돌려준다


def run_tracking(name, det_kw=None, trk_kw=None):  # 영상 하나를 끝까지 추적하는 함수
    """프레임마다 [(track_id, bbox), ...] 목록을 돌려준다."""
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, f"{name}.mp4"))  # 영상을 연다
    detector = MotionDetector(**(det_kw or {}))  # 감지기(설정 적용)
    tracker = Tracker(**(trk_kw or {}))  # 추적기(설정 적용)
    results = []  # 프레임별 추적 결과
    while True:  # 끝까지 반복
        ok, frame = cap.read()  # 프레임 읽기
        if not ok:  # 끝이면
            break  # 종료
        _, dets = detector.apply(frame)  # 감지
        tracker.update([d[0] for d in dets], [d[1] for d in dets])  # 추적
        results.append([(t.id, t.bbox) for t in tracker.visible_tracks()])  # 이번 프레임에 보인 객체 기록
    cap.release()  # 영상 닫기
    return results  # 결과를 돌려준다


def assign_ids(gt_frames, pred_frames):  # 정답 객체마다 어떤 ID가 붙었는지 프레임별로 구하는 함수
    """{정답ID: [(프레임, 추적ID 또는 None, 가림비율), ...]} 를 돌려준다."""
    seq = {}  # 결과 저장
    for f, gts in enumerate(gt_frames):  # 프레임마다
        preds = pred_frames[f] if f < len(pred_frames) else []  # 같은 프레임의 추적 결과
        assigned = {}  # 정답ID → 추적ID
        if gts and preds:  # 둘 다 있으면
            cost = np.full((len(gts), len(preds)), np.inf)  # 비용표(기본 무한대 = 매칭 금지)
            for i, g in enumerate(gts):  # 정답 객체마다
                gx, gy, gw, gh = g["bbox"]  # 정답 박스
                gc = (gx + gw / 2, gy + gh / 2)  # 정답 중심
                for j, (_, pb) in enumerate(preds):  # 추적 결과마다
                    pc = (pb[0] + pb[2] / 2, pb[1] + pb[3] / 2)  # 추적 중심
                    d = float(np.hypot(gc[0] - pc[0], gc[1] - pc[1]))  # 중심 거리
                    if d <= max(gw, gh) * 0.75:  # 정답 박스 크기 기준으로 충분히 가까우면
                        cost[i, j] = d  # 후보로 인정
            for i, j in hungarian_assign(cost):  # 1:1 최적 매칭(한 ID가 두 정답에 동시에 붙지 않게)
                if np.isfinite(cost[i, j]):  # 허용된 짝만
                    assigned[gts[i]["id"]] = preds[j][0]  # 정답ID → 추적ID
        for g in gts:  # 정답 객체마다
            seq.setdefault(g["id"], []).append((f, assigned.get(g["id"]), g["occlusion"]))  # 기록 추가
    return seq  # 결과를 돌려준다


def count_events(seq):  # ID Switch와 추적 실패를 세는 함수
    """(switch_frames, fail_frames, occluded_frames) — 각각 발생 프레임 목록."""
    switches, fails, occluded = [], [], []  # 결과 목록
    current, cand, cand_len, cand_start = None, None, 0, 0  # 현재 ID, 후보 ID, 후보 유지 길이, 후보 시작 프레임
    miss_len, miss_start = 0, None  # 연속 미검출 길이와 시작 프레임
    for f, tid, occ in seq:  # 프레임 순서대로
        if occ >= OCCLUSION_LIMIT:  # 50% 이상 가려졌으면
            occluded.append(f)  # 가림 프레임으로 기록
            if tid is None:  # ID도 없으면
                continue  # 실패 계산에서 제외(연속 길이는 끊지도 늘리지도 않음)
        if tid is None:  # ID가 없으면(미검출)
            if miss_len == 0:  # 미검출이 새로 시작되면
                miss_start = f  # 시작 프레임 기록
            miss_len += 1  # 연속 미검출 길이 증가
            if miss_len == FAIL_MIN_FRAMES:  # 기준(10프레임)에 딱 도달하면
                fails.append(miss_start)  # 실패 1회(시작 프레임 기준)
            cand, cand_len = None, 0  # 후보 ID 연속성도 끊긴다
            continue  # 다음 프레임
        miss_len = 0  # 검출되었으므로 미검출 길이 초기화
        if tid == current:  # 기존 ID 그대로면
            cand, cand_len = None, 0  # 후보 초기화
            continue  # 다음 프레임
        if tid == cand:  # 같은 후보가 계속되면
            cand_len += 1  # 유지 길이 증가
        else:  # 새 후보면
            cand, cand_len, cand_start = tid, 1, f  # 후보 등록
        if cand_len >= SWITCH_MIN_FRAMES:  # 후보가 5프레임 이상 유지되면
            if current is not None:  # 이전 ID가 있었다면
                switches.append(cand_start)  # ID Switch 1회
            current, cand, cand_len = cand, None, 0  # 후보를 현재 ID로 확정
    return switches, fails, occluded  # 결과를 돌려준다


def evaluate(name, det_kw=None, trk_kw=None):  # 영상 하나를 평가하는 함수
    """전체 합계와 이벤트(테스트 구간)별 결과를 돌려준다."""
    gt = load_gt(name)  # 정답
    preds = run_tracking(name, det_kw, trk_kw)  # 추적 실행
    seqs = assign_ids(gt["frames"], preds)  # 정답별 ID 순서
    all_sw, all_fail, all_occ = [], [], []  # 전체 기록
    for gid, seq in seqs.items():  # 정답 객체마다
        sw, fl, oc = count_events(seq)  # 이벤트 계산
        all_sw += sw  # 합치기
        all_fail += fl  # 합치기
        all_occ += oc  # 합치기
    per_event = []  # 테스트 구간별 결과
    for ev in gt["events"]:  # 구간마다
        inside = lambda fs: sum(ev["start"] <= f <= ev["end"] for f in fs)  # 구간 안 개수를 세는 작은 함수
        per_event.append({**ev, "switch": inside(all_sw), "fail": inside(all_fail), "occluded": inside(all_occ)})  # 구간 결과 저장
    ids_created = len({tid for frame in preds for tid, _ in frame})  # 생성된 고유 ID 수
    return {"name": name, "condition": gt["condition"], "tests": len(gt["events"]),  # 기본 정보
            "switch": len(all_sw), "fail": len(all_fail), "occluded": len(all_occ),  # 합계
            "ids": ids_created, "events": per_event}  # 생성 ID 수, 구간별 결과


def failure_rate(r):  # 조건별 실패율 = (ID Switch 또는 추적 실패가 1건 이상 발생한 테스트 수) / 전체 테스트 수
    bad = sum(1 for e in r["events"] if e["switch"] or e["fail"])  # 문제가 생긴 테스트 수
    return bad / r["tests"] if r["tests"] else 0.0  # 비율


def table(rows, header):  # 마크다운 표 문자열을 만드는 함수
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]  # 머리줄과 구분줄
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]  # 데이터 줄
    return "\n".join(lines)  # 줄바꿈으로 이어 붙인다


def main():  # 모든 실험을 실행하고 결과를 마크다운으로 저장한다
    out = []  # 결과 문서 줄 목록
    scenarios = ["single_move", "two_cross", "pause_resume", "light_change"]  # 4가지 조건

    base = {name: evaluate(name) for name in scenarios}  # 기본 설정(MOG2 자동 학습률, 거리, greedy)
    rows = []  # 표 1 데이터
    for r in base.values():  # 조건마다
        note = f"가림≥50% {r['occluded']}프레임 제외, 생성 ID {r['ids']}개, 실패율 {failure_rate(r):.0%}"  # 비고
        rows.append([r["condition"], r["tests"], r["switch"], r["fail"], note])  # 한 줄
    out += ["## 1. 추적 성능 (기본 설정: MOG2 lr=auto, distance, greedy, max_distance=80)", "",  # 제목
            table(rows, ["상황", "테스트 횟수", "ID Switch", "추적 실패", "비고"]), ""]  # 표

    rec = []  # 기록 템플릿 데이터
    for r in base.values():  # 조건마다
        for e in r["events"]:  # 구간마다
            memo = e["label"] + (f", 가림 {e['occluded']}프레임" if e["occluded"] else "")  # 관찰 메모
            rec.append([r["name"] + ".mp4", r["condition"], f"{e['start']}~{e['end']}", e["switch"], e["fail"], memo])  # 한 줄
    out += ["## 2. 이벤트별 기록 (영상명 - 조건 - 프레임 범위 - ID Switch 수 - 실패 수 - 관찰 메모)", "",  # 제목
            table(rec, ["영상명", "조건", "프레임 범위", "ID Switch 수", "실패 수", "관찰 메모"]), ""]  # 표

    rows = []  # 학습률 비교 데이터
    for method in ["mog2", "absdiff"]:  # 두 배경 차분 방식
        for lr in [-1.0, 0.001, 0.01, 0.1]:  # 학습률 4가지(-1 = 자동/기본)
            if method == "absdiff" and lr == -1.0:  # absdiff에서 -1은 0.02와 같으므로
                continue  # 생략
            row = [method, "auto" if lr < 0 else lr]  # 줄 시작
            for name in ["single_move", "pause_resume", "light_change"]:  # 학습률 영향이 큰 조건들
                r = evaluate(name, {"method": method, "learning_rate": lr})  # 평가
                row.append(f"SW {r['switch']} / FAIL {r['fail']} / ID {r['ids']}")  # 결과 칸
            rows.append(row)  # 한 줄 저장
    out += ["## 3. 배경 학습률 비교 (SW=ID Switch, FAIL=추적 실패, ID=생성된 ID 수)", "",  # 제목
            table(rows, ["방식", "학습률", "단일 이동", "일시 정지", "조명 변화"]), ""]  # 표

    configs = [  # 겹침 문제를 "규칙"으로 해결하려는 시도들
        ("기본 (distance, greedy, d=80)", {}),  # 기본
        ("거리 임계값 축소 d=40", {"max_distance": 40}),  # 임계값 줄이기
        ("거리 임계값 확대 d=150", {"max_distance": 150}),  # 임계값 늘리기
        ("ID 유지 시간 축소 max_disappeared=3", {"max_disappeared": 3}),  # 빨리 삭제
        ("ID 유지 시간 확대 max_disappeared=40", {"max_disappeared": 40}),  # 오래 유지
        ("Hungarian 매칭", {"matcher": "hungarian"}),  # 최적 할당
        ("IoU 매칭 (greedy)", {"metric": "iou"}),  # IoU 기준
        ("IoU 매칭 (hungarian)", {"metric": "iou", "matcher": "hungarian"}),  # IoU + 최적 할당
        ("칼만 필터 (greedy)", {"use_kalman": True}),  # 예측 위치로 매칭
        ("칼만 필터 + Hungarian", {"use_kalman": True, "matcher": "hungarian"}),  # 예측 + 최적 할당
        ("칼만 + Hungarian + max_disappeared=40", {"use_kalman": True, "matcher": "hungarian", "max_disappeared": 40}),  # 가장 많이 조합
    ]
    rows = []  # 비교 데이터
    for label, kw in configs:  # 설정마다
        r2 = evaluate("two_cross", None, kw)  # 겹침 영상 평가
        r1 = evaluate("single_move", None, kw)  # 단일 이동 영상도 함께 평가(부작용 확인)
        keep = 1 - failure_rate(r2)  # 추적 유지율 = 문제없이 끝난 테스트 비율
        rows.append([label, r2["switch"], r2["fail"], f"{keep:.0%}", r2["ids"], f"SW {r1['switch']} / FAIL {r1['fail']}"])  # 한 줄
    out += ["## 4. 겹침(two_cross) 해결 시도 비교 (테스트 10회)", "",  # 제목
            table(rows, ["설정", "ID Switch", "추적 실패", "추적 유지율", "생성 ID 수", "단일 이동 부작용"]), ""]  # 표

    os.makedirs("outputs", exist_ok=True)  # 출력 폴더
    path = os.path.join("outputs", "eval_results.md")  # 출력 파일
    with open(path, "w", encoding="utf-8") as fp:  # 파일 열기
        fp.write("\n".join(out))  # 저장
    print("\n".join(out))  # 화면에도 출력
    print(f"\n[OK] saved: {path}", file=sys.stderr)  # 저장 위치 안내


if __name__ == "__main__":  # 직접 실행했을 때만
    main()  # 평가 실행
