# 2-4 스마트에디터의 그래머리 (문장 교정/교열) 기능 고도화

- 네이버 사용자가 작성한 문장을 문법적으로 맞는 문장으로 교정/교열 하는 모델을 만듭니다.


## 데이터
- 학습데이터
  * `train/train_data/train_data`: 문법 오류가 섞인 문장
  * `train/train_data/train_annotation`: 문법 오류에 대한 annotation
  * `train/train_data/train_corpus`: 교정되지 않은 문장
  * `train/train_label`: 교정/교열된 문장
- 평가 데이터
  * `test/test_data`: 문법 오류가 섞인 문장
  * `test/test_label`: 교정/교열된 문장
- 평가 더미 데이 
  * `test_submit/test_data`: 문법 오류가 섞인 문장
  * `test_submit/test_label`: 교정/교열된 문장
- 문법 오류가 섞인 문장들(`*_data`)과 교정/교열된 문장들(`*_label`)은 line-by-line으로 매핑됩니다.


## 평가
- Corpus-level [GLEU](https://www.aclweb.org/anthology/P07-1044/) score로 평가 
- [`nltk.translate.gleu_score.corpus_gleu`](https://www.nltk.org/_modules/nltk/translate/gleu_score.html) 스크립트를 사용


## 베이스라인
- [Transformer](https://arxiv.org/abs/1706.03762) 기반의 sequence-to-sequence 모델
- 대량의 unlabeled corpus (`train_corpus`)를 활용하여 pre-training (또는 semi-supervised learning) 방식으로 학습하거나 에러 타입 (`train_annotation`)을 예측하도록 multi-task learning을 하면 추가 성능 향상을 얻을 수도 있습니다.  


## 모델 학습
```
nsml run -d airush2021-2-4 -e train.py
``` 
- 필요에 따라 `-a`로 argument 입력 가능


## 모델 제출
```
nsml submit {SESSION} {CHECKPOINT}
```

## 추가 정보

### Annotation 설명

- "perfect" : 교정/교열이 필요없는 완벽한 문장
- "spacing" : 띄어쓰기 교정
- "pasting" : 붙여쓰기 교정
- "tense" : 시제 교정
- "honorific" : 경어체 교정
- "punctuation" : 구두점 교정
- "typo" : 오탈자 교정 (위 분류에 없는 경우 모두 수렴)
- "advanced" : 윤문 처리 (더 매끄러운 문장)
