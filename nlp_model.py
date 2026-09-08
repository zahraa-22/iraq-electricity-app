import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

print("⏳ جاري قراءة البيانات الضخمة وتدريب النموذج...")

# تحميل داتا سيت كاجل المعالجة
df = pd.read_csv("kaggle_processed_iraq.csv")

X = df["Complaint_Text"]
y = df["Fault_Type"]

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تحويل النصوص إلى متجه عددي احترافي (TF-IDF)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# تدريب النموذج
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# تقييم النموذج
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 دقة النموذج على داتا سيت كاجل: {accuracy * 100:.2f}%\n")

# حفظ الموديل والـ Vectorizer
joblib.dump(model, "nlp_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ تم حفظ nlp_model.pkl و vectorizer.pkl بنجاح!")