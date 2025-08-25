# 🦟 Dengue Detection using Machine Learning

هذا المشروع بيستخدم خوارزميات تعلم الآلة للتنبؤ بحمى الضنك اعتمادًا على:
- الأعراض (Fever, Headache, JointPain, Bleeding, Rash)
- فحوصات الدم (Platelets, WBC, Hematocrit)
- عوامل إضافية (Age, Gender)

---

## 📂 محتويات المشروع
- `dengue_extended_dataset.csv` → Dataset تجريبية (2000 سجل).
- `train_model.py` → كود تدريب النموذج باستخدام RandomForest.
- `model.pkl` → النموذج المدرب (يمكن تحميله واستخدامه مباشرة للتنبؤ).
- `notebook.ipynb` → Jupyter Notebook يحتوي على خطوات التدريب والتحليل.
- `README.md` → وصف المشروع.

---

## ⚙️ المتطلبات
- Python 3.8+
- المكتبات:
  ```bash
  pip install pandas scikit-learn matplotlib seaborn
  
🚀 طريقة التشغيل
git clone https://github.com/esmail830/dengue-detection.git
cd dengue-detection
📊 النتائج

Accuracy: ~91%

ROC-AUC: ~0.93


أهم Features: Bleeding, Platelets, WBC
📌 ملاحظات

البيانات هنا مصطنعة للتجريب (Synthetic)، وليست بيانات طبية حقيقية.

يمكن تحسين الأداء بإضافة Features إضافية أو استخدام نماذج مثل XGBoost/LightGBM.

الهدف: عرض كيف يمكن استخدام ML في دعم القرار الطبي.
