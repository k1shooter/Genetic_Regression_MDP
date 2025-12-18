# 🚀 GA Defect Prediction Analysis using NASA MDP Datasets

## 📄 프로젝트 개요
본 프로젝트는 NASA MDP (Metrics Data Program) 소프트웨어 결함 예측 데이터셋을 활용해, GA 기반 Symbolic Regression 구현 및 분석을 목표로 합니다.

## 🔗 데이터셋 출처
본 프로젝트에 사용된 MDP 데이터셋의 출처는 다음과 같습니다:

[NASA Defect Dataset (klainfo/NASADefectDataset)](https://github.com/klainfo/NASADefectDataset)

## 🚀 실행 방법

1) 데이터 로드 및 preprocessing

```
python preprocessing.py
```

2) 기존 classifier 실행

```commandline
# 2-1. DNN + optuna & Random Forest + optuna
python classifiers/optuna_tuning.py
```
```
# 2-2. Random Forest + CHIRPS
python classifiers/chirps_full.py
```
```
# 2-3. Naive Bayes
classifiers/naive_bayes.py
```

3) GA algorithm 으로 수식 생성

```commandline
python ga_mo/main.py
```

4) 성능 최종비교

```commandline
python evaluation.py
```

## 🧩 Tree Complexity & Optimization

본 프로젝트에서는 결함 예측 모델의 성능뿐만 아니라 **설명 가능성(Explainability)**과 **일반화(Generalizability)**를 확보하기 위해 **Tree Complexity(트리 복잡도)**를 최소화하는 것을 중요한 목표로 삼았습니다.

### Tree Complexity 계산
트리 복잡도는 수식 트리를 구성하는 **전체 노드의 개수(Size)**로 정의됩니다.
- **계산 식**: `Complexity = 1 (Self) + sum(Child.Size for Child in Children)`
- 모든 함수 노드와 터미널 노드(변수 및 상수)를 포함한 총 노드 수를 합산하여 계산합니다.

### Optimization Parameters (Objectives)
다목적 유전 알고리즘(NSGA-II)을 사용하여 다음 두 가지 파라미터를 동시에 최적화합니다:
1.  **Maximize F1 Score**: `1 - F1 Score`를 최소화하는 방향으로 설정하여, 예측 정확도를 높입니다.
2.  **Minimize Tree Complexity**: 트리의 크기(`size()`)를 최소화하여, 과적합(Overfitting)을 방지하고 사람이 이해하기 쉬운 간단한 공식을 유도합니다.