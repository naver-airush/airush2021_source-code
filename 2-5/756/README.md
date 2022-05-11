# 2-5 쇼핑 카탈로그 클러스터링

## 문제

![Figure](https://open.oss.navercorp.com/storage/user/3/files/d4a88c80-c865-11eb-9e50-27ed62a631c7)

- 네이버 쇼핑은 여러 판매처에서 판매 중인 동일 상품들의 가격을 비교해 주는 서비스를 제공하고 있다.
- 가격 비교를 위해서는 먼저 여러 판매자가 등록한 같은 상품들을 하나로 묶어야 한다. 
- 이렇게 묶인 상품의 집합을 `카탈로그`라고 부른다. 
- 같은 상품이라도, 여러 판매자들이 각각 서로 다른 `상품명`으로 판매를 하고 있다. 
- 같은 `카탈로그`에 속하는 `상품명` 예시
  - 아래 7개의 상품은 모두 `농심 백산수 2L`라는 동일 카탈로그에 속하는 상품이다.
```
  * 농심 백산수 2L 1병 생수 
  * 농심 백산수 2L
  * 백산수 생수 2L, 낱개
  * (24개이상 구입시 개당 20원씩 할인) 농심 백산수 2L
  * 농심 백산수 2L x 1펫 / 생수 샘물 물 박스포장
  * [농심]백산수 2L x 1개
  * NS473 백두산 백산수 2L
```
- **본 과제는 판매자가 등록한 상품명 텍스트(`query`)를 입력으로 받아, 주어진 `database`에서 그와 동일한 상품들을 모두 찾아내는 문제이다.**


## 데이터셋

- airush2021-2-5
  - `train`
    - 네이버 쇼핑의 `식품` 카테고리에서 뽑은 상품 160,008개
    - 데이터 위치 : `train/train_data` 디렉토리
    - 데이터 형식 : `상품ID(nv_mid)` / `상품명(prod_nm)` / `카탈로그ID(match_nv_mid)`
    - 각 상품은 unique한 `상품ID(nv_mid)`를 가진다.
    - 같은 카탈로그에 속하는 상품은 동일한 `카탈로그ID(match_nv_mid)`를 갖는다. 예를 들어, 아래 데이터 예시에서 맨 위 3개의 상품은 동일 카탈로그에 속하므로, 동일한 match_nv_mid(10062684657)을 갖고 있다.
    - 데이터 예시
![Figure](https://open.oss.navercorp.com/storage/user/3/files/e9852000-c865-11eb-9c94-0963c31fde06)

  - `test`
    - `test` 데이터셋은 `database`와 `query`로 구성되어 있다.
    - `database`
      - 네이버 쇼핑의 `식품` 카테고리에서 뽑은 상품 90,516개
      - 데이터 위치 : `test/test_data/database` 디렉토리
      - 데이터 형식 : `상품ID(nv_mid)` / `상품명(prod_nm)`
      - `train` 데이터와는 달리, `카탈로그ID(match_nv_mid)` 필드가 없다.
    - `query`
      - `query`는 위 `database`의 subset이며, 개수는 8,640개이다.
      - 데이터 위치 : `test/test_data/query` 디렉토리
      - 데이터 형식 : `상품ID(nv_mid)` / `상품명(prod_nm)`
    - 데이터 예시
![Figure](https://open.oss.navercorp.com/storage/user/3/files/f30e8800-c865-11eb-8fa0-82ced97afac5)

  - 필드 정보 
    - nv_mid (string) : 상품ID
    - prod_nm (string) : 상품명
    - match_nv_mid (string) : 카탈로그ID (train 데이터셋에만 있고, test 데이터셋에는 없음) 

    
## 결과 제출 포맷

- 결과 제출은 main.py의 infer() 함수에서 이루어진다.
- 아래 예시와 같이 `database`와 `query`가 주어졌다면, 
  - `query`의 `nv_mid_002`는 `database`의 `nv_mid_001`, `nv_mid_002`, `nv_mid_005`와 동일한 상품이며, 
  - `query`의 `nv_mid_003`은 `database`의 `nv_mid_003`, `nv_mid_004`와 동일한 상품이다.
- database
  
| nv_mid | prod_nm |
| --- | --- |
| nv_mid_001 | (무료배송) 삼다수 2L |
| nv_mid_002 | 삼다수 2L 12병 |
| nv_mid_003 | 저분자 피쉬 콜라겐 펩타이드 150g |
| nv_mid_004 | 지웨이 슈가 먹는 저분자 피쉬 콜라겐 펩타이드 150g |
| nv_mid_005 | 삼다수 2L |

- query
  
| nv_mid | prod_nm |
| --- | --- |
| nv_mid_002 | 삼다수 2L 12병 |
| nv_mid_003 | 저분자 피쉬 콜라겐 펩타이드 150g |

- `query`에 속하는 각각의 상품에 대해, 그와 동일한 상품을 `database`에서 모두 찾아서 제출하면 된다. 
즉, main.py의 infer() 함수에서 아래와 같은 list를 결과로 return하면 결과가 제출된다.
```
return [
  ('nv_mid_002', ['nv_mid_001', 'nv_mid_002', 'nv_mid_005]),
  ('nv_mid_003', ['nv_mid_003', 'nv_mid_004'])
]
  
```

## Getting started
- 접근 
  - 먼저 주어진 상품명 텍스트를 embedding하는 model을 학습시키고, 
  - 학습된 모델을 사용해서, test_data/database의 상품명들을 embedding한 후,
  - test_data/query의 각 상품명에 대해서 database 상품명들 중, embedding이 유사한 것을 search하는 것이 일반적인 접근법이다. 
  - 유사 embedding search는 main.py의 infer() 함수 부분를 수정하여, 구현하면 된다. 
  
- 학습
```
nsml run -d airush2021-2-5 -e main.py
```

- 리더보드 제출
```
nsml submit {session} {checkpoint}
```

## evaluation metric
* mean f1-score
  * test_data/query 의 각 상품에 대해 match된 결과의 precision과 recall의 f1 score를 모든 상품에 대해 평균한 값
  * evaluation.py 코드 참조.


## 기타 

- Team blog: https://medium.com/naver-shopping-dev
- Contact: 오광진 kj.oh@navercorp.com
 
 
## FAQ

Q : Pretrained model 사용이 가능한가요?

A : 사용 가능합니다.
