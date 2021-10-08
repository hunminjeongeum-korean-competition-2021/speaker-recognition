# 화자인식 인공지능 경진대회

- [문제2 화자 인식 데이터셋 설명](#문제-2-화자인식-dataset-설명)

## 대회 규칙

- **주제**
  - 음성 데이터에서 발화자 일치 여부를 도출해낼 수 있는 인공지능 알고리즘 개발
- **평가**
  - EER(Equal Error Rate)
- **NSML GPU 지원**
  - Tesla V100-SXM2-32GB 1개
- **<u>외부 데이터 및 사전 학습 모델 사용 불가</u>**

## 문제 2 **[화자인식 Dataset 설명]**

- 매칭된 두개의 음성파일을 읽어 같은 발화자인지 다른 발화자인지 추론

  | 전체 크기 |                 파일수                  | NSML 데이터셋 이름 |
  | :-------: | :-------------------------------------: | :---: |
  |  42.2GB   | train_data(239,378)<br>test_data(1,221) | speaker |

### Train Dataset

- `root_path/train/train_data/` (239,279개의 wav 파일 \*확장자 없는 레이블 형태)

  ```
  idx_000001
  idx_000002
  idx_000003
  idx_000004
  ...
  idx_239375
  idx_239376
  idx_239377
  idx_239378
  ```

### Train Lable

- `root_path/train/train_label`

- `train_label (DataFrame 형식, 238,822rows)`

  - columns - `["file_name", "file_name_", "label"]`

  - `file_name` - train_data 폴더에 존재하는 임의의 wav파일명 (ex. idx_000001)

  - `file_name_` - train_data 폴더에 존재하는 임의의 wav파일명 (ex. idx_000002)

  - `label` - 두 파일에 녹음된 발화자가 동일인물이면 1, 다른인물이면 0 (Binary Classification)

### Test Dataset

- `root_path/test/test_data/wav/` (1,221개의 wav 파일 \*확장자 없는 레이블 형태)

  ```
  idx_000001
  idx_000002
  idx_000003
  idx_000004
  ...
  idx_001218
  idx_001219
  idx_001220
  idx_001221
  ```

- `root_path/test/test_data/test_data`

- `test_data (DataFrame 형식, 30,493rows)`

  - columns - `["file_name", "file_name_]`

  - `file_name` - wav 폴더에 존재하는 임의의 wav파일명 (ex. idx_000001)

  - `file_name_` - wav 폴더에 존재하는 임의의 wav파일명 (ex. idx_000002)

  - test_data에 `label` column을 추가하여 추론값을 대입 (최종 제출 format)
