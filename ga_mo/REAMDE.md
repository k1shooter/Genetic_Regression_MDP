## 📂 Multi-Objective GA Solution

`ga-mo/` 디렉토리는 **NSGA-II (Non-dominated Sorting Genetic Algorithm II)** 알고리즘을 기반으로 한 다목적 최적화(Multi-Objective Optimization) 구현체입니다.

### 목적 함수 (Objectives)
이 모델은 다음 두 가지 목표를 동시에 최적화하여 Pareto Front를 탐색합니다:
1.  **Maximize F1-Score**: 결함 예측 성능의 극대화.
2.  **Minimize Tree Complexity**: 모델의 복잡도 최소화 (설명 가능성 향상).

### 트리 복잡도 (Tree Complexity) 계산 방식
트리의 복잡도(Size)는 **전체 노드의 개수**로 정의됩니다.
*   계산 식: `Complexity = 1 (Self) + sum(Child.Size for Child in Children)`
*   예: `x1 + x2`는 `add(x1, x2)` 형태이므로, 루트(`add`) 1개 + 자식(`x1`) 1개 + 자식(`x2`) 1개 = 총 **3**의 복잡도를 가집니다.

## 🚀 실행 방법

프로젝트 루트 디렉토리에서 다음 명령어를 실행하세요:

### Single-Objective GA
```bash
python ga/main.py
```

### Multi-Objective GA (Pareto Optimization)
```bash
python ga-mo/main.py
```

실행 시, 각 데이터셋(CM1, JM1 등)에 대해 GA가 실행되며, 학습된 최적의 공식과 정확도, F1-Score가 출력됩니다.
최종 결과는 `ga_results.csv` 또는 `ga_mo_results.csv` 파일로 저장됩니다.
