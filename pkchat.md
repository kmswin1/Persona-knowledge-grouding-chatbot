# Paper Idea

다음주 주말 (3/26)

문장 자체로 constrain
distant supervision (relation을 무시하고 NER이 같으면 유사 문장으)

Candidate Generation - IE (distant supervision)
KG를 그대로 모델에 넣는다?

발표 끝나고,
아니면 일요일마다

1. 민상님 : distant supervision (IE 수준 높이기)
2. 민식님 : Baseline constrain (tf-idf candidate based constrain) / S-BERT 써서 해보는거

발표 끝나고 얘기를

1. 민식님 : Dialogue / Persona Sentence Embedding 처리
2. 민상님 : Knowledge IE 추출 및 퀄리티 보기 (GenIE)

## KG IE result

Sentosa -> is a -> Singapore main island

Sentosa -> famous for -> Universal studios

## Prompting

Sentosa is famous for universal studios -> embedding

## GNN

둘다 해보기

## Persona

I like island vacation -> embedding

## Dialogue

Where is the famous place here? -> embedding

## Query prompted embedding space

Sentosa -> famous for -> Universal studios
object (Universal studios) -> constrained beam search
(baseline 모델 돌리면서 아웃풋만 빔서치)

blah - universal studios - blah
