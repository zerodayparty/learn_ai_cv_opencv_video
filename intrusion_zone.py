"""침입 감지 구역 모듈.

마우스로 화면에 사각형을 그리면, 객체의 중심점이 그 안에 들어올 때
경고 메시지를 출력하고 로그 파일에 기록한다.
"""

import os  # 폴더 생성용
from datetime import datetime  # 현재 시각 기록용

import cv2  # OpenCV


class IntrusionZone:  # 침입 감지 구역 클래스
    """왼쪽 버튼 드래그 = 구역 지정, 오른쪽 클릭 = 구역 삭제."""

    def __init__(self, rect=None, log_path="outputs/intrusion_log.txt"):  # 시작 구역과 로그 경로를 받는다
        self.rect = self._normalize(rect) if rect else None  # 구역 (x1, y1, x2, y2), 없으면 None
        self.log_path = log_path  # 로그 파일 경로
        self._drag_start = None  # 드래그 시작점
        self._drag_now = None  # 드래그 중 현재 위치
        self.inside_ids = set()  # 지금 구역 안에 있는 ID들
        self.total_events = 0  # 지금까지 발생한 침입 횟수

    @staticmethod  # 객체 없이 호출 가능한 함수
    def _normalize(rect):  # 좌표 순서를 (작은 값, 큰 값)으로 정리한다
        x1, y1, x2, y2 = (int(v) for v in rect)  # 정수로 바꾼다
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))  # 왼쪽 위 / 오른쪽 아래 순서로 맞춘다

    def on_mouse(self, event, x, y, flags, param):  # cv2.setMouseCallback에 연결할 함수
        """마우스 이벤트를 받아 구역을 그린다."""
        if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 버튼을 누르면
            self._drag_start = (x, y)  # 드래그 시작점 저장
            self._drag_now = (x, y)  # 현재 위치도 같게
        elif event == cv2.EVENT_MOUSEMOVE and self._drag_start is not None:  # 누른 채로 움직이면
            self._drag_now = (x, y)  # 현재 위치 갱신
        elif event == cv2.EVENT_LBUTTONUP and self._drag_start is not None:  # 버튼을 떼면
            x0, y0 = self._drag_start  # 시작점
            if abs(x - x0) > 5 and abs(y - y0) > 5:  # 너무 작은 사각형(실수 클릭)은 무시
                self.rect = self._normalize((x0, y0, x, y))  # 구역 확정
                self.inside_ids.clear()  # 구역이 바뀌었으니 기록 초기화
                print(f"[ZONE] intrusion zone set: {self.rect}")  # 콘솔에 알린다
            self._drag_start = None  # 드래그 종료
            self._drag_now = None  # 드래그 종료
        elif event == cv2.EVENT_RBUTTONDOWN:  # 오른쪽 클릭이면
            self.rect = None  # 구역 삭제
            self.inside_ids.clear()  # 기록 초기화
            print("[ZONE] intrusion zone cleared")  # 콘솔에 알린다

    def check(self, tracks, frame_index):  # 이번 프레임의 객체들이 구역 안에 있는지 확인한다
        """구역 안에 있는 ID 목록을 돌려준다. 새로 들어온 ID는 경고 + 로그 기록."""
        if self.rect is None:  # 구역이 없으면
            return []  # 확인할 것이 없다
        x1, y1, x2, y2 = self.rect  # 구역 좌표
        now_inside = set()  # 이번 프레임에 구역 안에 있는 ID
        for t in tracks:  # 객체마다
            cx, cy = t.centroid  # 중심점
            if x1 <= cx <= x2 and y1 <= cy <= y2:  # 중심점이 구역 안이면
                now_inside.add(t.id)  # 안에 있다고 기록한다
        for tid in sorted(now_inside - self.inside_ids):  # 이번에 "새로" 들어온 ID만
            self.total_events += 1  # 침입 횟수 증가
            msg = f"{datetime.now().isoformat(timespec='seconds')} frame={frame_index} INTRUSION id={tid} zone={self.rect}"  # 로그 문장
            print(f"[WARNING] {msg}")  # 콘솔에 경고 출력
            log_dir = os.path.dirname(self.log_path)  # 로그 폴더 경로
            if log_dir:  # 폴더 경로가 있으면
                os.makedirs(log_dir, exist_ok=True)  # 폴더가 없으면 만든다
            with open(self.log_path, "a", encoding="utf-8") as f:  # 로그 파일을 이어쓰기 모드로 연다
                f.write(msg + "\n")  # 한 줄 기록한다
        self.inside_ids = now_inside  # 현재 상태를 저장한다
        return sorted(now_inside)  # 구역 안 ID 목록을 돌려준다

    def draw(self, frame):  # 화면에 구역을 그린다
        """확정 구역은 빨간색(침입 중) / 노란색(평소), 드래그 중에는 하늘색으로 그린다."""
        if self.rect is not None:  # 구역이 있으면
            x1, y1, x2, y2 = self.rect  # 좌표
            color = (0, 0, 255) if self.inside_ids else (0, 255, 255)  # 침입 중이면 빨강, 아니면 노랑(BGR 순서)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # 사각형을 그린다
            cv2.putText(frame, "ZONE", (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)  # 라벨을 쓴다
        if self._drag_start is not None and self._drag_now is not None:  # 드래그 중이면
            cv2.rectangle(frame, self._drag_start, self._drag_now, (255, 255, 0), 1)  # 임시 사각형을 그린다
