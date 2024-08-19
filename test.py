from evaluate import load
bertscore = load("bertscore")
predictions = ["why do you kick him", "hello my friend how are you"]
references = ["my name is ryu", "how do you do"]
results = bertscore.compute(predictions=predictions, references=references, lang="en" , model_type = "microsoft/deberta-xlarge-mnli")
print(results)

roguescore = load("rouge")
results = roguescore.compute(predictions=predictions, references=references , use_aggregator=False)
print(results)