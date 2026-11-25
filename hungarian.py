"""Hungarian Algorithm(헝가리안 알고리즘)을 scipy 없이 NumPy만으로 구현한 모듈.

비용 행렬(cost matrix)이 주어지면, 각 행(기존 객체)과 각 열(새 감지 결과)을
1:1로 짝지었을 때 "비용의 합"이 가장 작아지는 조합을 찾아준다.
"""

import numpy as np  # 행렬 계산을 위해 NumPy를 불러온다


def hungarian_assign(cost_matrix):  # 비용 행렬을 받아 최적 짝 목록을 돌려주는 함수
    """최소 비용 1:1 할당 결과를 [(행 번호, 열 번호), ...] 형태로 돌려준다."""
    cost = np.asarray(cost_matrix, dtype=np.float64)  # 입력을 실수(float64) 행렬로 바꾼다
    if cost.ndim != 2 or cost.size == 0:  # 2차원이 아니거나 비어 있으면
        return []  # 짝지을 것이 없으므로 빈 목록을 돌려준다

    finite = np.isfinite(cost)  # 무한대(inf)나 NaN이 아닌 칸을 찾는다
    big = (np.abs(cost[finite]).max() + 1.0) * 1e3 if finite.any() else 1e9  # 계산이 깨지지 않도록 "아주 큰 값"을 정한다
    cost = np.where(finite, cost, big)  # 무한대 칸은 아주 큰 값으로 바꿔서 계산한다

    transposed = cost.shape[0] > cost.shape[1]  # 행이 열보다 많으면 뒤집어야 한다(알고리즘은 행 ≤ 열에서 동작)
    if transposed:  # 행이 더 많다면
        cost = cost.T  # 행과 열을 바꾼다(전치)

    n, m = cost.shape  # n = 행 개수, m = 열 개수 (n ≤ m)
    u = np.zeros(n + 1)  # 행 쪽 "잠재값(potential)" (1번부터 사용, 0번은 빈 자리)
    v = np.zeros(m + 1)  # 열 쪽 잠재값
    p = np.zeros(m + 1, dtype=np.int64)  # p[j] = 열 j에 배정된 행 번호 (0이면 아직 없음)
    way = np.zeros(m + 1, dtype=np.int64)  # way[j] = 열 j로 오기 직전에 지나온 열 (경로 되짚기용)

    for i in range(1, n + 1):  # 행을 하나씩 추가하면서 최적 배정을 늘려간다
        p[0] = i  # 가상의 0번 열에 현재 행 i를 잠시 올려둔다
        j0 = 0  # 현재 서 있는 열 번호
        minv = np.full(m + 1, np.inf)  # 각 열까지 가는 "최소 추가 비용"
        used = np.zeros(m + 1, dtype=bool)  # 이번 탐색에서 이미 방문한 열 표시
        while True:  # 비어 있는 열을 찾을 때까지 반복한다
            used[j0] = True  # 현재 열을 방문했다고 표시한다
            i0 = p[j0]  # 현재 열에 연결된 행을 가져온다
            cur = cost[i0 - 1, :] - u[i0] - v[1:]  # 그 행에서 각 열로 갈 때의 "줄어든 비용"을 한 번에 계산한다
            free = ~used[1:]  # 아직 방문하지 않은 열만 고른다
            better = free & (cur < minv[1:])  # 기존 최소값보다 더 싼 열을 찾는다
            minv[1:][better] = cur[better]  # 더 싼 비용으로 갱신한다
            way[1:][better] = j0  # 그 열로 올 때 지나온 열을 기록한다
            masked = np.where(free, minv[1:], np.inf)  # 방문한 열은 후보에서 제외한다
            j1 = int(np.argmin(masked)) + 1  # 가장 싼 다음 열을 고른다 (+1은 1번부터 세기 때문)
            delta = masked[j1 - 1]  # 그 열까지의 최소 비용
            used_idx = np.nonzero(used)[0]  # 방문한 열 번호들
            u[p[used_idx]] += delta  # 방문한 열에 연결된 행들의 잠재값을 올린다
            v[used_idx] -= delta  # 방문한 열들의 잠재값을 내린다
            minv[1:][free] -= delta  # 방문 안 한 열들의 최소 비용도 그만큼 줄인다
            j0 = j1  # 다음 열로 이동한다
            if p[j0] == 0:  # 그 열이 비어 있으면(아무 행도 배정 안 됨)
                break  # 증가 경로를 찾았으므로 멈춘다
        while True:  # 찾은 경로를 거꾸로 따라가며 배정을 한 칸씩 밀어준다
            j1 = way[j0]  # 직전 열을 가져온다
            p[j0] = p[j1]  # 직전 열의 행을 현재 열로 옮긴다
            j0 = j1  # 한 칸 뒤로 이동한다
            if j0 == 0:  # 시작점(0번 열)까지 돌아왔으면
                break  # 되짚기를 끝낸다

    pairs = []  # 결과 짝 목록
    for j in range(1, m + 1):  # 모든 열을 확인하면서
        if p[j] != 0:  # 행이 배정된 열이라면
            row, col = int(p[j] - 1), int(j - 1)  # 0번부터 세는 번호로 바꾼다
            pairs.append((col, row) if transposed else (row, col))  # 뒤집었었다면 다시 원래 방향으로 돌려 저장한다
    pairs.sort()  # 행 번호 순서로 정렬한다
    return pairs  # 최적 짝 목록을 돌려준다


def greedy_assign(cost_matrix, valid_mask=None):  # 비교용: 가장 싼 짝부터 욕심껏(greedy) 고르는 함수
    """비용이 작은 짝부터 차례대로 고른다. 전체 합이 최소라는 보장은 없다."""
    cost = np.asarray(cost_matrix, dtype=np.float64)  # 입력을 실수 행렬로 바꾼다
    if cost.ndim != 2 or cost.size == 0:  # 비어 있으면
        return []  # 빈 목록을 돌려준다
    if valid_mask is None:  # 허용 칸 정보가 없으면
        valid_mask = np.isfinite(cost)  # 유한한 값인 칸만 허용한다
    rows, cols = np.nonzero(valid_mask)  # 허용된 칸의 (행, 열) 위치를 모은다
    order = np.argsort(cost[rows, cols], kind="stable")  # 비용이 작은 순서로 정렬한다
    used_rows, used_cols = set(), set()  # 이미 짝지어진 행/열 기록
    pairs = []  # 결과 짝 목록
    for k in order:  # 싼 짝부터 하나씩 본다
        r, c = int(rows[k]), int(cols[k])  # 현재 후보의 행과 열
        if r in used_rows or c in used_cols:  # 둘 중 하나라도 이미 짝이 있으면
            continue  # 건너뛴다
        used_rows.add(r)  # 행을 사용했다고 표시한다
        used_cols.add(c)  # 열을 사용했다고 표시한다
        pairs.append((r, c))  # 짝을 저장한다
    pairs.sort()  # 행 번호 순서로 정렬한다
    return pairs  # 결과를 돌려준다
