import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
sys.stdout.reconfigure(encoding='utf-8')
# 1. تحميل البيانات
data = pd.read_csv("data.csv")

# 2. فصل الميزات عن الهدف
X = data.drop(["Name", "Dengue"], axis=1, errors="ignore")
y = data["Dengue"]

# معالجة الأعمدة النصية (ترميز فئوي)
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype('category').cat.codes

# 3. تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. تطبيع البيانات (مهم جداً مع KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. تدريب نموذج KNN
model = KNeighborsClassifier(
    n_neighbors=5,   # عدد الجيران (ممكن تجرب 3, 7, 11 ..)
    metric='minkowski',  # مسافة إقليدية
    p=2
)
model.fit(X_train, y_train)
#joblib.dump(model, "knn_model.pkl")
#joblib.dump(scaler, "scaler.pkl")

# 6. التقييم
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]  # احتمالية الإصابة

print("📊 تقرير التصنيف:")
print(classification_report(y_test, y_pred))

print("\n📌 مصفوفة الالتباس:")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC:", roc_auc_score(y_test, y_prob))
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns

# احتمالات التنبؤ
y_prob = model.predict_proba(X_test)[:,1]

# 1️⃣ ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# 2️⃣ Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

plt.figure(figsize=(6,6))
plt.plot(recall, precision, color="green")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# 3️⃣ Confusion Matrix رسومية
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Dengue","Dengue"], yticklabels=["Not Dengue","Dengue"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
