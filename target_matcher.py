"""특징점 매칭으로 "미리 등록한 물체"를 영상에서 다시 찾는 모듈.

1) 등록 이미지에서 특징점(모서리처럼 눈에 띄는 점)과 설명자(descriptor)를 뽑는다.
2) 매 프레임에서도 특징점을 뽑아 두 이미지의 설명자를 서로 비교한다.
3) 비율 테스트(ratio test)로 애매한 매칭을 버리고, RANSAC 호모그래피로 모양이 맞는 매칭만 남긴다.
"""

from dataclasses import dataclass  # 결과를 담는 간단한 클래스를 만들기 위해 사용

import cv2  # OpenCV
import numpy as np  # 행렬 계산


@dataclass  # 아래 클래스를 "값 묶음"으로 만든다
class MatchResult:  # 한 번 매칭한 결과
    found: bool  # 물체를 찾았는지 여부
    target_keypoints: int  # 등록 이미지에서 뽑은 특징점 수
    frame_keypoints: int  # 현재 프레임에서 뽑은 특징점 수
    good_matches: int  # 비율 테스트를 통과한 매칭 수
    inliers: int  # RANSAC 호모그래피로 모양까지 맞는 매칭 수
    corners: np.ndarray | None = None  # 찾은 물체의 네 꼭짓점(화면 좌표), 못 찾으면 None

    @property  # 함수처럼 계산하지만 값처럼 읽을 수 있게 한다
    def match_rate(self):  # 매칭률
        """매칭률 = 비율 테스트 통과 매칭 수 / 등록 이미지 특징점 수."""
        return self.good_matches / self.target_keypoints if self.target_keypoints else 0.0  # 0으로 나누기 방지


class TargetMatcher:  # 등록 물체를 찾는 클래스
    """method="orb"(빠름, 기본 opencv) 또는 "sift"(정확, opencv-contrib 필요)."""

    def __init__(self, target, method="orb", ratio=0.75, min_inliers=12, nfeatures=1500):  # 설정을 받는다
        if isinstance(target, str):  # 경로 문자열이 들어오면
            image = cv2.imread(target)  # 파일에서 이미지를 읽는다
            if image is None:  # 읽기에 실패하면
                raise FileNotFoundError(f"target image not found: {target}")  # 오류를 알려준다
        else:  # 이미 이미지 배열이면
            image = target  # 그대로 사용한다
        self.method = method.lower()  # 방식 이름을 소문자로 저장한다
        self.ratio = ratio  # 비율 테스트 기준(작을수록 엄격)
        self.min_inliers = min_inliers  # "찾았다"고 인정할 최소 인라이어 수
        if self.method == "orb":  # ORB 방식이면
            self.detector = cv2.ORB_create(nfeatures=nfeatures)  # ORB 특징점 추출기를 만든다
            norm = cv2.NORM_HAMMING  # ORB 설명자는 0/1 비트라서 해밍 거리로 비교한다
        elif self.method == "sift":  # SIFT 방식이면
            self.detector = cv2.SIFT_create(nfeatures=nfeatures)  # SIFT 특징점 추출기를 만든다
            norm = cv2.NORM_L2  # SIFT 설명자는 실수라서 유클리드 거리로 비교한다
        else:  # 그 외
            raise ValueError(f"method must be 'orb' or 'sift', got {method!r}")  # 오류를 알려준다
        self.matcher = cv2.BFMatcher(norm)  # 모든 쌍을 비교하는 브루트포스 매처
        self.target_image = image  # 등록 이미지를 저장한다
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 흑백으로 바꾼다
        self.target_kp, self.target_des = self.detector.detectAndCompute(gray, None)  # 등록 이미지의 특징점과 설명자
        h, w = gray.shape  # 등록 이미지 크기
        self.target_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)  # 등록 이미지 네 꼭짓점

    def match(self, frame):  # 한 프레임에서 물체를 찾는다
        """MatchResult를 돌려준다."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame  # 흑백으로 바꾼다
        kp, des = self.detector.detectAndCompute(gray, None)  # 프레임의 특징점과 설명자
        n_target = len(self.target_kp)  # 등록 특징점 수
        n_frame = len(kp)  # 프레임 특징점 수
        if self.target_des is None or des is None or n_frame < 2:  # 비교할 것이 없으면
            return MatchResult(False, n_target, n_frame, 0, 0)  # 못 찾음
        knn = self.matcher.knnMatch(self.target_des, des, k=2)  # 각 등록 특징점마다 가장 비슷한 2개를 찾는다
        good = [p[0] for p in knn if len(p) == 2 and p[0].distance < self.ratio * p[1].distance]  # 1등이 2등보다 확실히 좋을 때만 채택
        inliers, corners = 0, None  # 기본값
        if len(good) >= 4:  # 호모그래피 계산에는 최소 4쌍이 필요하다
            src = np.float32([self.target_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)  # 등록 이미지 쪽 좌표
            dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)  # 프레임 쪽 좌표
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)  # RANSAC으로 모양 변환(호모그래피)을 추정한다
            if H is not None and mask is not None:  # 추정에 성공하면
                inliers = int(mask.sum())  # 변환과 잘 맞는 매칭 수
                corners = cv2.perspectiveTransform(self.target_corners, H)  # 등록 이미지 꼭짓점을 화면 좌표로 옮긴다
                if not cv2.isContourConvex(corners.astype(np.int32)):  # 꼬인 사각형이면(잘못된 변환)
                    inliers, corners = 0, None  # 무효 처리한다
        found = inliers >= self.min_inliers  # 인라이어가 충분하면 "찾았다"
        return MatchResult(found, n_target, n_frame, len(good), inliers, corners if found else None)  # 결과를 돌려준다
