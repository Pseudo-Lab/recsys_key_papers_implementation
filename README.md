# 추천시스템 주요 논문 리뷰 및 구현


## Welcome
가짜연구소 6기 '추천시스템 주요 논문 리뷰 및 구현' 스터디의 추천 알고리즘 구현 & 실행 & 리팩토링 & 공부를 위한 repository입니다.

본 스터디의 목표는 다음과 같습니다.
- 가장 기본적인 Collaborative Filtering부터 최근 핫한 그래프 기반의 추천모델까지 원리를 이해할 수 있음
- 추천시스템의 발전 과정을 파악할 수 있음
- 읽은 논문은 새로운 논문을 부른다😃 논문 리뷰 경험을 기반으로 스터디가 끝난 이후에도 빠르게 서치할 수 있는 힘 기르기


## Contributors

| 순번  | 발표날짜      | 이름                  | 모델명(노션 발표자료 링크)                                                                                                                                    | 논문명                                                                                                                                 |
|-----|-----------|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| 1   | 2023.3.26 | 이경찬                 | [SASRec](https://www.notion.so/chanrankim/SASRec-23cfd848c75143f890adc7cc17dba8a3?pvs=4)                                                           | [Self-Attentive Sequential Recommendation](https://arxiv.org/pdf/1808.09781.pdf)                                                    |
| 2   | 2023.4.2  | 박순혁                 | [NGCF](https://www.notion.so/chanrankim/NGCF-4b7770de468947b99268df3cd6bd1823?pvs=4)                                                               | [Neural graph collaborative filtering](https://arxiv.org/abs/1905.08108)                                                            |
| 3   | 2023.4.9  | 박혜리                 | [FPMC](https://www.notion.so/chanrankim/FPMC-27d788aa42ba408688656e93ad87c0ee?pvs=4)                                                               | [Factorizing personalized markov chains for next-basket recommendation](https://dl.acm.org/doi/10.1145/1772690.1772773)             |
| 4   | 2023.4.16 | 김용직                 | [GraphRec](https://www.notion.so/chanrankim/GraphRec-eac38c4df31640969a397a3417360b8e?pvs=4)                                                       | [Graph neural networks for social recommendation](https://arxiv.org/abs/1902.07243)                                                 |
| 5   | 2023.4.23 | 남궁민상, 최은희, 이경찬      | 공동발표   (Collaborative Filtering)                                                                                                                   | [추천시스템 survey paper & 기본다지기 1](https://www.notion.so/chanrankim/1-02e110cf185c49e9a3a54e1dcbc73af7?pvs=4)                           |
| 6   | 2023.4.30 | 이한빈                 | [PinSage](https://www.notion.so/chanrankim/PinSage-4f7944cd667844de8fe8ad7fb84a59c5?pvs=4)                                                         | [Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://arxiv.org/abs/1806.01973)                           
| 7   | 2023.5.7  | 박지원                 | [MeLU](https://www.notion.so/chanrankim/MeLU-04cd72fd9f5b40719a7a524ce07fdde3?pvs=4)                                                               | [MeLU: Meta-Learned User Preference Estimator for Cold-Start Recommendation](https://arxiv.org/abs/1908.00413)                      |
| 8   | 2023.5.14 | 신용근                 | [NCF](https://www.notion.so/chanrankim/NCF-4f9ad71a139240fda402c9a8a5f90744?pvs=4)                                                                 | [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)                                                                  |
| 9   | 2023.5.21 | 남궁민상, 박혜리, 이남준, 박순혁 | 공동발표   (GNN강의 CS224W-[웹사이트](http://web.stanford.edu/class/cs224w/)/[영상](https://www.youtube.com/playlist?list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn)) | [추천시스템 survey paper & 기본다지기 2](https://www.notion.so/chanrankim/2-a94b8e65c76242e5acee1344f20930e7?pvs=4)                           |
| 10  | 2023.5.28 | 최은희                 | [GRU4Rec](https://www.notion.so/chanrankim/GRU4Rec-665fafe4c13c42109419eb6695623500?pvs=4)                                                         | [Session-based Recommendations with Recurrent Neural Networks](https://arxiv.org/abs/1511.06939)                                    |
| 11  | 2023.6.4  | 이남준                 | [SR-GNN](https://www.notion.so/chanrankim/SR-GNN-6ffa6e246e1849e585ed88521733d809?pvs=4)                                                           | [Session-based Recommendation with Graph Neural Networks](https://arxiv.org/abs/1811.00855)                                         |
| 12  | 2023.6.11 | 이한빈, 이권동, 박지원  | 공동발표([Recommender Systems: A Primer](https://arxiv.org/abs/2302.02579))                                                                            | [추천시스템 survey paper & 기본다지기 3](https://www.notion.so/chanrankim/3-a23b0a4dc541454d851b7082de0086e0?pvs=4)                                                                                                    |
| 13  | 2023.6.18 | 남궁민상                | [KPRN](https://www.notion.so/chanrankim/KPRN-c6ea655e688f49f29482869111dd95b4?pvs=4)                                                               | [Explainable reasoning over knowledge graphs for recommendation](https://arxiv.org/abs/1905.08108)                                  |
| 14  | 2023.6.25 | 이권동                 | [BERT4Rec](https://www.notion.so/chanrankim/BERT4Rec-62a7124085be40d8864f6c6033525c94?pvs=4)                                                       | [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/abs/1904.06690) |





