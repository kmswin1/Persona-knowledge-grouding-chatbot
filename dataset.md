# Dataset Information

Dialog topics : landmark (GLDv2)

## size
Knowlege - 5316 Wiki pages (8000+ characters per page)
Persona - 27170 sentences
Dialogue - 14,452 dialogues, 12 average turns, 173,424 utterances total

## Each Dialogue
persona - 5 sentences (ex - experience, preference, possession, hobby or interest)
```
we let them to extract the keywords in the given
Wikipedia page and make the persona sentences by means of
the keywords.
```
Something we are replicating in terms of architecture!!

```
Meanwhile, the workers were also allowed to create topic-agnostic
persona sentences.
```
Let's review some examples.... and weed this out if possible.

Human / Machine created by single person

```
In this situation, the machine
answers the question by considering both knowledge and
persona or only knowledge
```
Something we can review data....

### Idea

grounded on Persona
Question -> Persona -> (Knowledge)

grounded on knowledge
Question -> Knowledge

### etc

Machine response is much longer (x3.5)
44518 Knowledge answers, 42194 Persona-Knowledge Answers

Inform / Confirm / Suggest - Can all use our model? Any structural finetuning?

## Baseline

Retreival module - 5 passages per question. TFIDF estimated with BERTScore
(Maybe something we can do in addition to S-BERT selections)
