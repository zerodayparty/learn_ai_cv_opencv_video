"""객체 추적(Tracker) 모듈.

매 프레임 감지된 바운딩 박스들을 이전 프레임의 객체들과 짝지어서
같은 물체에는 같은 ID를 계속 붙여준다.

- 매칭 기준(metric): 중심점 거리("distance") 또는 IoU("iou")
- 매칭 방법(matcher): 가장 가까운 것부터 고르는 "greedy" 또는 최적 할당 "hungarian"
- 칼만 필터(use_kalman): 다음 위치를 예측해서 그 예측 위치로 매칭
"""

from collections import deque  # 길이 제한이 있는 목록(궤적 저장용)

import cv2  # OpenCV (칼만 필터 사용)
import numpy as np  # 행렬 계산

from hungarian import greedy_assign, hungarian_assign  # 직접 구현한 매칭 함수들


def bbox_center(bbox):  # 바운딩 박스의 중심점을 구하는 함수
    """(x, y, w, h) 박스의 중심점 (cx, cy)를 돌려준다."""
    x, y, w, h = bbox  # 박스를 왼쪽 위 좌표와 너비, 높이로 나눈다
    return (x + w / 2.0, y + h / 2.0)  # 가운데 좌표를 계산해 돌려준다


def bbox_iou(a, b):  # 두 박스가 얼마나 겹치는지(IoU) 계산하는 함수
    """IoU = 겹친 넓이 / 합친 넓이. 0이면 안 겹치고 1이면 완전히 같다."""
    ax, ay, aw, ah = a  # 첫 번째 박스
    bx, by, bw, bh = b  # 두 번째 박스
    ix1, iy1 = max(ax, bx), max(ay, by)  # 겹친 영역의 왼쪽 위 좌표
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)  # 겹친 영역의 오른쪽 아래 좌표
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)  # 겹친 넓이 (안 겹치면 0)
    union = aw * ah + bw * bh - inter  # 두 박스를 합친 넓이
    return inter / union if union > 0 else 0.0  # 비율을 돌려준다 (0으로 나누기 방지)


def create_kalman(cx, cy):  # 위치 (x, y)와 속도 (vx, vy)를 추정하는 칼만 필터를 만드는 함수
    """등속도(일정한 속도) 운동을 가정한 칼만 필터를 만든다."""
    kf = cv2.KalmanFilter(4, 2)  # 상태 4개(x, y, vx, vy), 측정값 2개(x, y)
    kf.transitionMatrix = np.array(  # 다음 상태 = 현재 위치 + 속도 라는 규칙
        [[1, 0, 1, 0],  # x 다음값 = x + vx
         [0, 1, 0, 1],  # y 다음값 = y + vy
         [0, 0, 1, 0],  # vx는 그대로
         [0, 0, 0, 1]],  # vy는 그대로
        dtype=np.float32,  # OpenCV는 float32를 사용한다
    )
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)  # 측정값은 위치(x, y)만 본다
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03  # 움직임 자체의 불확실성(작을수록 등속 운동을 믿음)
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0  # 감지 결과의 불확실성
    kf.errorCovPost = np.diag([1, 1, 25, 25]).astype(np.float32)  # 처음 속도는 모르므로 불확실성을 크게 둔다
    kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)  # 시작 상태: 현재 위치, 속도 0
    return kf  # 만든 칼만 필터를 돌려준다


class Track:  # 추적 중인 객체 1개를 표현하는 클래스
    """ID, 현재 박스, 궤적, 사라진 프레임 수 등을 저장한다."""

    def __init__(self, track_id, bbox, centroid, trail_length, use_kalman):  # 새 객체를 등록할 때 호출된다
        self.id = track_id  # 고유 ID
        self.bbox = tuple(int(v) for v in bbox)  # 현재 바운딩 박스
        self.centroid = centroid  # 현재 중심점
        self.disappeared = 0  # 연속으로 안 보인 프레임 수
        self.hits = 1  # 지금까지 매칭된 총 횟수
        self.trail = deque(maxlen=trail_length)  # 이동 경로(최근 trail_length개 점만 저장)
        self.trail.append((int(centroid[0]), int(centroid[1])))  # 첫 위치를 궤적에 추가한다
        self.kalman = create_kalman(*centroid) if use_kalman else None  # 칼만 필터 사용 시 생성
        self.predicted = centroid  # 다음 프레임에서 있을 것으로 예상되는 위치

    def predict(self):  # 이번 프레임에서 객체가 어디 있을지 예측한다
        """칼만 필터가 있으면 예측 위치를, 없으면 마지막 위치를 사용한다."""
        if self.kalman is not None:  # 칼만 필터를 쓰는 경우
            pred = self.kalman.predict()  # 다음 상태를 예측한다
            self.predicted = (float(pred[0, 0]), float(pred[1, 0]))  # 예측된 (x, y)를 저장한다
        else:  # 칼만 필터를 안 쓰는 경우
            self.predicted = self.centroid  # 마지막으로 본 위치가 곧 예상 위치다
        return self.predicted  # 예측 위치를 돌려준다

    def predicted_bbox(self):  # 예측 위치로 옮긴 박스를 만든다 (IoU 매칭용)
        """현재 박스 크기를 유지한 채 중심만 예측 위치로 옮긴다."""
        _, _, w, h = self.bbox  # 박스 크기만 가져온다
        px, py = self.predicted  # 예측된 중심
        return (px - w / 2.0, py - h / 2.0, w, h)  # 예측 중심을 기준으로 박스를 다시 만든다

    def update(self, bbox, centroid):  # 이번 프레임 감지 결과와 매칭되었을 때 호출된다
        """박스와 중심점을 갱신하고 궤적에 점을 추가한다."""
        self.bbox = tuple(int(v) for v in bbox)  # 박스를 새 값으로 바꾼다
        self.centroid = centroid  # 중심점을 새 값으로 바꾼다
        if self.kalman is not None:  # 칼만 필터를 쓰면
            self.kalman.correct(np.array([[centroid[0]], [centroid[1]]], dtype=np.float32))  # 실제 측정값으로 예측을 보정한다
        self.trail.append((int(centroid[0]), int(centroid[1])))  # 궤적에 새 점을 추가한다
        self.disappeared = 0  # 다시 보였으므로 사라진 횟수를 0으로 되돌린다
        self.hits += 1  # 매칭 횟수를 1 늘린다

    def mark_missed(self):  # 이번 프레임에서 짝을 찾지 못했을 때 호출된다
        """사라진 프레임 수를 1 늘린다."""
        self.disappeared += 1  # 안 보인 프레임 수 증가


class Tracker:  # 여러 객체의 ID를 관리하는 추적기
    """감지 결과 목록을 받아 객체별 ID를 유지·생성·삭제한다."""

    def __init__(  # 추적기 설정값을 받는다
        self,
        max_disappeared=15,  # 이 프레임 수보다 오래 안 보이면 ID를 삭제한다
        max_distance=80.0,  # 중심점 거리가 이보다 멀면 같은 객체로 보지 않는다(픽셀)
        metric="distance",  # 매칭 기준: "distance"(중심점 거리) 또는 "iou"
        matcher="greedy",  # 매칭 방법: "greedy" 또는 "hungarian"
        use_kalman=False,  # 칼만 필터 예측 사용 여부
        iou_threshold=0.1,  # IoU 매칭 시 이 값 이상 겹쳐야 같은 객체로 인정한다
        trail_length=64,  # 궤적으로 저장할 최대 점 개수
    ):
        if metric not in ("distance", "iou"):  # 지원하지 않는 기준이면
            raise ValueError(f"metric must be 'distance' or 'iou', got {metric!r}")  # 오류를 알려준다
        if matcher not in ("greedy", "hungarian"):  # 지원하지 않는 매칭 방법이면
            raise ValueError(f"matcher must be 'greedy' or 'hungarian', got {matcher!r}")  # 오류를 알려준다
        self.max_disappeared = max_disappeared  # 설정 저장
        self.max_distance = float(max_distance)  # 설정 저장
        self.metric = metric  # 설정 저장
        self.matcher = matcher  # 설정 저장
        self.use_kalman = use_kalman  # 설정 저장
        self.iou_threshold = float(iou_threshold)  # 설정 저장
        self.trail_length = trail_length  # 설정 저장
        self.tracks = {}  # 현재 추적 중인 객체들 {ID: Track}
        self.next_id = 1  # 다음에 새로 줄 ID 번호

    def _register(self, bbox, centroid):  # 새 객체를 등록한다
        """새 ID를 부여하고 추적 목록에 추가한다."""
        track = Track(self.next_id, bbox, centroid, self.trail_length, self.use_kalman)  # 새 Track을 만든다
        self.tracks[self.next_id] = track  # 목록에 넣는다
        self.next_id += 1  # 다음 ID 번호를 1 늘린다

    def _cost_matrix(self, track_list, bboxes, centroids):  # 객체와 감지 결과 사이의 비용표를 만든다
        """행 = 기존 객체, 열 = 새 감지 결과. 값이 작을수록 같은 객체일 가능성이 크다."""
        n, m = len(track_list), len(bboxes)  # 행과 열 개수
        cost = np.zeros((n, m), dtype=np.float64)  # 비용표를 0으로 만든다
        valid = np.zeros((n, m), dtype=bool)  # 매칭 허용 여부표
        for i, track in enumerate(track_list):  # 기존 객체마다
            px, py = track.predicted  # 예측 위치(칼만 미사용 시 마지막 위치)
            pbox = track.predicted_bbox()  # 예측 위치로 옮긴 박스
            for j in range(m):  # 새 감지 결과마다
                if self.metric == "distance":  # 거리 기준이면
                    cx, cy = centroids[j]  # 감지 결과의 중심
                    d = float(np.hypot(cx - px, cy - py))  # 두 중심 사이의 직선 거리
                    cost[i, j] = d  # 거리 자체가 비용이다
                    valid[i, j] = d <= self.max_distance  # 너무 멀면 매칭 금지
                else:  # IoU 기준이면
                    iou = bbox_iou(pbox, bboxes[j])  # 겹침 비율을 계산한다
                    cost[i, j] = 1.0 - iou  # 많이 겹칠수록 비용이 작아지도록 뒤집는다
                    valid[i, j] = iou >= self.iou_threshold  # 너무 적게 겹치면 매칭 금지
        return cost, valid  # 비용표와 허용표를 돌려준다

    def update(self, bboxes, centroids=None):  # 매 프레임 호출되는 핵심 함수
        """감지된 박스 목록을 받아 ID를 갱신하고, 현재 추적 목록 {ID: Track}을 돌려준다."""
        bboxes = [tuple(b) for b in bboxes]  # 박스들을 튜플 목록으로 통일한다
        if centroids is None:  # 중심점이 따로 주어지지 않았으면
            centroids = [bbox_center(b) for b in bboxes]  # 박스 중심을 중심점으로 쓴다
        centroids = [(float(c[0]), float(c[1])) for c in centroids]  # 실수 좌표로 통일한다

        track_list = list(self.tracks.values())  # 기존 객체 목록
        for track in track_list:  # 모든 기존 객체에 대해
            track.predict()  # 이번 프레임 위치를 예측한다

        if not track_list:  # 기존 객체가 하나도 없으면
            for bbox, c in zip(bboxes, centroids):  # 감지된 것들을 모두
                self._register(bbox, c)  # 새 객체로 등록한다
            return self.tracks  # 결과를 돌려준다

        pairs = []  # 매칭 결과
        if bboxes:  # 감지 결과가 하나라도 있으면
            cost, valid = self._cost_matrix(track_list, bboxes, centroids)  # 비용표를 만든다
            if self.matcher == "hungarian":  # 헝가리안 방식이면
                gated = np.where(valid, cost, np.inf)  # 금지된 칸은 무한대 비용으로 막는다
                pairs = [(i, j) for i, j in hungarian_assign(gated) if valid[i, j]]  # 최적 할당 후 금지 칸 짝은 버린다
            else:  # greedy 방식이면
                pairs = greedy_assign(cost, valid)  # 가장 가까운 짝부터 고른다

        matched_tracks, matched_dets = set(), set()  # 짝이 생긴 객체/감지 번호 기록
        for i, j in pairs:  # 짝마다
            track_list[i].update(bboxes[j], centroids[j])  # 기존 객체를 새 위치로 갱신한다(같은 ID 유지)
            matched_tracks.add(i)  # 객체 i는 짝이 있음
            matched_dets.add(j)  # 감지 j는 짝이 있음

        for i, track in enumerate(track_list):  # 모든 기존 객체를 다시 보면서
            if i in matched_tracks:  # 이미 짝이 있으면
                continue  # 건너뛴다
            track.mark_missed()  # 이번에 안 보였다고 기록한다
            if track.disappeared > self.max_disappeared:  # 너무 오래 안 보였으면
                del self.tracks[track.id]  # ID를 삭제한다

        for j, (bbox, c) in enumerate(zip(bboxes, centroids)):  # 감지 결과를 다시 보면서
            if j not in matched_dets:  # 짝이 없는 감지 결과는
                self._register(bbox, c)  # 새로 나타난 객체이므로 새 ID를 준다

        return self.tracks  # 현재 추적 목록을 돌려준다

    def visible_tracks(self):  # 이번 프레임에 실제로 보인 객체만 고른다
        """disappeared == 0 인(이번 프레임에 매칭된) 객체 목록."""
        return [t for t in self.tracks.values() if t.disappeared == 0]  # 조건에 맞는 것만 돌려준다
