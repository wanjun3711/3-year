import pandas as pd
import xgboost as xgb
import streamlit as st

# 读取训练集数据
train_data = pd.read_csv('train_data.csv')

# 分离输入特征和目标变量
X = train_data[['Age', 'Sex', 'Histologic Type', 'Grade',
                'T stage', 'Surgery', 'Radiation', 'Chemotherapy',
                'Bone metastasis', 'Brain metastasis', 'Liver metastasis', 'Lung metastasis']]
y = train_data['Vital status']

# 创建并训练XGBoost模型
xgb_params = {
    'eta': 0.8970487704196783,
    'gamma': 8.90083169546758,
    'max_depth': 4,
    'min_child_weight': 0,
    'n_estimators': 41,
    'subsample': 0.8419386306424858,
}

xgb_model = xgb.XGBClassifier(**xgb_params)

# 特征映射
class_mapping = {0: "Alive", 1: "Dead"}
sex_mapper = {'male': 1, 'female': 2}
histologic_type_mapper = {"Adenocarcinoma": 1, "Squamous–cell carcinoma": 2}
grade_mapper = {"Grade I": 4, "Grade II": 1, "Grade III": 2, "Grade IV": 3}
t_stage_mapper = {"T1": 4, "T2": 1, "T3": 2, "T4": 3}
surgery_mapper = {"NO": 2, "Yes": 1}
radiation_mapper = {"NO": 2, "Yes": 1}
chemotherapy_mapper = {"NO": 2, "Yes": 1}
bone_metastasis_mapper = {"NO": 2, "Yes": 1}
brain_metastasis_mapper = {"NO": 2, "Yes": 1}
liver_metastasis_mapper = {"NO": 2, "Yes": 1}  
lung_metastasis_mapper = {"NO": 2, "Yes": 1}

# 训练XGBoost模型
xgb_model.fit(X, y)

# 预测函数
def predict_Vital_status(age, sex, histologic_type, grade,
                         t_stage, surgery, radiation, chemotherapy,
                         bone_metastasis, brain_metastasis, liver_metastasis, lung_metastasis):
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex_mapper[sex]],
        'Histologic Type': [histologic_type_mapper[histologic_type]],
        'Grade': [grade_mapper[grade]],
        'T stage': [t_stage_mapper[t_stage]],
        'Surgery': [surgery_mapper[surgery]],
        'Radiation': [radiation_mapper[radiation]],
        'Chemotherapy': [chemotherapy_mapper[chemotherapy]],
        'Bone metastasis': [bone_metastasis_mapper[bone_metastasis]],
        'Brain metastasis': [brain_metastasis_mapper[brain_metastasis]],
        'Liver metastasis': [liver_metastasis_mapper[liver_metastasis]],  
        'Lung metastasis': [lung_metastasis_mapper[lung_metastasis]]
    })
    prediction = xgb_model.predict(input_data)[0]
    probability = xgb_model.predict_proba(input_data)[0][0]  # 获取属于类别0的概率
    class_label = class_mapping[prediction]
    return class_label, probability

# 创建Web应用程序
st.title("3-year survival of ECM patients based on XGBoost")
st.sidebar.write("Variables")

age = st.sidebar.number_input("Age", min_value=0, max_value=99, step=1)  # 允许用户输入具体数值
sex = st.sidebar.selectbox("Sex", ('male', 'female'))
histologic_type = st.sidebar.selectbox("Histologic Type", options=list(histologic_type_mapper.keys()))
grade = st.sidebar.selectbox("Tumor grade", options=list(grade_mapper.keys()))
t_stage = st.sidebar.selectbox("T stage", options=list(t_stage_mapper.keys()))
surgery = st.sidebar.selectbox("Surgery", options=list(surgery_mapper.keys()))
radiation = st.sidebar.selectbox("Radiation", options=list(radiation_mapper.keys()))
chemotherapy = st.sidebar.selectbox("Chemotherapy", options=list(chemotherapy_mapper.keys()))
bone_metastasis = st.sidebar.selectbox("Bone metastasis", options=list(bone_metastasis_mapper.keys()))
brain_metastasis = st.sidebar.selectbox("Brain metastasis", options=list(brain_metastasis_mapper.keys()))
liver_metastasis = st.sidebar.selectbox("Liver metastasis", options=list(liver_metastasis_mapper.keys())) 
lung_metastasis = st.sidebar.selectbox("Lung metastasis", options=list(lung_metastasis_mapper.keys()))

if st.button("Predict"):
    prediction, probability = predict_Vital_status(
        age, sex, histologic_type, grade,
        t_stage, surgery, radiation, chemotherapy,
        bone_metastasis, brain_metastasis, liver_metastasis, lung_metastasis
    )

    st.write("Predict survival at the last observation:", prediction)
    st.write("Probability of 3-year survival is:", probability)
