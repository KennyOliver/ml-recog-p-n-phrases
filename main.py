from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer


positive_texts = [
  "we love you",
  "they love us",
  "you are good",
  "he is good",
  "they love mary"
]

negative_texts =  [
  "we hate you", 
  "they hate us",
  "you are bad",
  "he is bad",
  "we hate mary"
]

test_texts = [
  "they love mary",
  "they are good",
  "why do you hate mary",
  "they are almost always good",
  "we are very bad"
]


training_texts = negative_texts + positive_texts
training_labels = ["negative"] * len(negative_texts) + ["positive"] * len(positive_texts)

vectorizer = CountVectorizer()
vectorizer.fit(training_texts)
print(vectorizer.vocabulary_)

training_vectors = vectorizer.transform(training_texts)
testing_vectors = vectorizer.transform(test_texts)

classifier = tree.DecisionTreeClassifier()
classifier.fit(training_vectors, training_labels)
predictions = classifier.predict(testing_vectors)
print(predictions)


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(5,5))
tree.plot_tree(classifier,feature_names = vectorizer.get_feature_names(), rounded = True, filled = True) 
fig.savefig('tree.png')