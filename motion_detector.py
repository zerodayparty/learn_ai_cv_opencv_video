"""움직임 감지(배경 차분 + 모폴로지 + 윤곽선) 모듈.

1) 배경 차분: 가만히 있는 배경과 지금 프레임을 비교해 "달라진 곳"(전경)을 찾는다.
2) 모폴로지: 점 같은 노이즈는 지우고(열림), 구멍 난 덩어리는 메운다(닫힘).
3) 윤곽선: 덩어리의 테두리를 찾아 바운딩 박스와 중심점을 계산한다.
"""

import cv2  # OpenCV
import numpy as np  # 행렬 계산


class MotionDetector:  # 움직이는 물체를 찾아주는 클래스
    """method="mog2"(OpenCV 내장 배경 모델) 또는 "absdiff"(직접 구현한 차영상)."""

    def __init__(  # 감지기 설정값을 받는다
        self,
        method="mog2",  # 배경 차분 방식
        learning_rate=-1.0,  # 배경 학습률 (MOG2에서 -1은 자동, absdiff에서 -1은 0.02 사용)
        min_area=400,  # 이 넓이(픽셀 수)보다 작은 덩어리는 무시한다
        max_area_ratio=0.5,  # 화면의 이 비율보다 큰 덩어리는 무시한다(조명 급변 등)
        history=500,  # MOG2가 배경으로 기억하는 프레임 수
        var_threshold=25,  # MOG2 전경 판단 민감도(클수록 둔감)
        detect_shadows=True,  # MOG2 그림자 감지 여부(그림자는 회색 127로 표시됨)
        diff_threshold=30,  # absdiff에서 "달라졌다"고 볼 밝기 차이
        open_kernel=3,  # 열림 연산 커널 크기(노이즈 제거)
        close_kernel=15,  # 닫힘 연산 커널 크기(구멍 메우기)
    ):
        if method not in ("mog2", "absdiff"):  # 지원하지 않는 방식이면
            raise ValueError(f"method must be 'mog2' or 'absdiff', got {method!r}")  # 오류를 알려준다
        self.method = method  # 설정 저장
        self.learning_rate = float(learning_rate)  # 설정 저장
        self.min_area = min_area  # 설정 저장
        self.max_area_ratio = max_area_ratio  # 설정 저장
        self.diff_threshold = diff_threshold  # 설정 저장
        self.open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))  # 열림용 원형 커널
        self.close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))  # 닫힘용 원형 커널
        self.subtractor = None  # MOG2 배경 모델(필요할 때 생성)
        self.background = None  # absdiff용 배경 이미지(누적 평균)
        if method == "mog2":  # MOG2 방식이면
            self.subtractor = cv2.createBackgroundSubtractorMOG2(  # OpenCV의 가우시안 혼합 배경 모델을 만든다
                history=history, varThreshold=var_threshold, detectShadows=detect_shadows  # 설정값 전달
            )
        self.last_foreground_ratio = 0.0  # 직전 프레임에서 전경이 차지한 비율(조명 급변 감지용)

    def _foreground_mask(self, frame):  # 전경 마스크(흰색 = 움직임)를 만든다
        """배경과 다른 곳은 255, 같은 곳은 0인 흑백 이미지를 돌려준다."""
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)  # 살짝 흐리게 해서 자잘한 노이즈를 줄인다
        if self.method == "mog2":  # MOG2 방식
            fg = self.subtractor.apply(blurred, learningRate=self.learning_rate)  # 배경 모델과 비교 + 배경 갱신
            _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)  # 그림자(127)는 버리고 확실한 전경(255)만 남긴다
            return fg  # 마스크를 돌려준다
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)  # absdiff 방식: 흑백으로 바꾼다
        if self.background is None:  # 첫 프레임이면
            self.background = gray.astype(np.float32)  # 지금 화면을 배경으로 기억한다
            return np.zeros_like(gray)  # 비교할 대상이 없으므로 빈 마스크를 돌려준다
        diff = cv2.absdiff(gray, cv2.convertScaleAbs(self.background))  # 배경과의 밝기 차이(절댓값)
        _, fg = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)  # 차이가 큰 곳만 흰색으로
        alpha = self.learning_rate if self.learning_rate >= 0 else 0.02  # 학습률(-1이면 기본값 0.02)
        cv2.accumulateWeighted(gray, self.background, alpha)  # 배경 = (1-α)×배경 + α×현재 화면 으로 천천히 갱신
        return fg  # 마스크를 돌려준다

    def apply(self, frame):  # 한 프레임을 처리하는 핵심 함수
        """(마스크, 감지 목록)을 돌려준다. 감지 목록 원소 = (bbox, centroid, area)."""
        fg = self._foreground_mask(frame)  # 전경 마스크를 구한다
        self.last_foreground_ratio = float(np.count_nonzero(fg)) / fg.size  # 화면 중 전경 비율을 기록한다
        mask = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.open_k)  # 열림 연산: 작은 점 노이즈 제거
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_k)  # 닫힘 연산: 덩어리 안 구멍 메우기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 바깥 윤곽선만 찾는다
        max_area = self.max_area_ratio * mask.shape[0] * mask.shape[1]  # 허용 최대 넓이
        detections = []  # 감지 결과 목록
        for cnt in contours:  # 윤곽선마다
            area = cv2.contourArea(cnt)  # 덩어리 넓이
            if area < self.min_area or area > max_area:  # 너무 작거나 너무 크면
                continue  # 무시한다(면적 필터링)
            x, y, w, h = cv2.boundingRect(cnt)  # 덩어리를 감싸는 사각형
            m = cv2.moments(cnt)  # 모멘트(무게중심 계산용 값)
            if m["m00"] > 0:  # 넓이가 0이 아니면
                cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]  # 무게중심 좌표
            else:  # 넓이가 0이면(이론상 거의 없음)
                cx, cy = x + w / 2.0, y + h / 2.0  # 박스 가운데를 중심으로 쓴다
            detections.append(((x, y, w, h), (cx, cy), area))  # 결과에 추가한다
        return mask, detections  # 마스크와 감지 결과를 돌려준다
