SMS Spam Detection using Naive Bayes
This project classifies SMS messages as Spam or Not Spam (Ham) using a Naive Bayes classifier.

Project Overview
The goal is to detect spam messages using machine learning techniques. We use the Naive Bayes algorithm, a popular model for text classification.
1. Import Required Libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
pandas: Handles the data in a DataFrame.
CountVectorizer: Converts text to a matrix of token counts.
train_test_split: Splits the dataset into training and test sets.
MultinomialNB: Naive Bayes algorithm for classification.
accuracy_score, confusion_matrix: Evaluate the model’s performance.
2. Load the Data
python
Copy code
data = pd.read_csv(r"C:\Users\theri\Downloads\sms+spam+collection\SMSSpamCollection", 
                   sep='\t', 
                   header=None, 
                   names=['Label', 'Message'])
sep='\t' specifies that the data is tab-separated.
header=None and names=['Label', 'Message'] rename the columns for easier access.
3. Label Encoding
data['Label'] = data['Label'].map({'ham': 0, 'spam': 1})
Converts labels: ham (not spam) to 0 and spam to 1, so the model can process them as numeric values.
4. Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(data['Message'], data['Label'], test_size=0.2, random_state=1)
train_test_split divides the data into training (80%) and test (20%) sets.
random_state=1 ensures consistent results.
5. Text Vectorization
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
CountVectorizer converts each message to a count vector (bag of words), turning text into a numerical format.
fit_transform on X_train creates the count matrix, while transform on X_test uses the same matrix to process test data.
6. Train the Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)
Initializes a Naive Bayes model with MultinomialNB.
fit trains the model on the training set.
7. Evaluate the Model
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
predict generates predictions on the test data.
accuracy_score calculates the model’s accuracy.
confusion_matrix shows counts of true/false positives/negatives, giving insight into performance.
8. Test with a New Message
new_message = ["Start your Data Analytics career by joining Crio’s program!"]
new_message_vec = vectorizer.transform(new_message)
print("Prediction for new message:", "Spam" if model.predict(new_message_vec)[0] == 1 else "Not Spam")
transform vectorizes a new sample message.
predict classifies the new message as "Spam" or "Not Spam".
Sample Output
Accuracy: Displays the model's performance score.
Confusion Matrix: Shows detailed classification results (TP, FP, TN, FN).
Prediction for New Message: Prints "Spam" or "Not Spam" for a sample input message.
Requirements
Install necessary libraries:
pip install pandas scikit-learn
Conclusion
This project demonstrates basic text classification with Naive Bayes, useful for spam detection. You can modify and experiment further to understand more complex text processing techniques.

