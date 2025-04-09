import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 加载数据集
df = pd.read_csv('diabetes.csv')
diabetes_mean_df = df.groupby('Outcome').mean()

# 准备训练数据
X = df.drop('Outcome',axis=1)
y = df['Outcome']

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model=LogisticRegression()
model.fit(X_train, y_train)

train_y_pred = model.predict(X_train)
test_y_pred = model.predict(X_test)

def app():
    # 设置页面标题和图标
    st.set_page_config(
        page_title="糖尿病预测应用",
        page_icon="🏥",
        layout="wide",
    )

    # 创建两列布局
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://img.freepik.com/free-vector/diabetes-blood-test-abstract-concept-vector-illustration-world-diabetes-day-blood-sugar-level-test-glucose-meter-endocrinology-diagnostic-laboratory-insulin-production-abstract-metaphor_335657-6320.jpg", width=300)
    
    with col2:
        st.title('糖尿病预测系统')
        st.write('基于机器学习的糖尿病风险预测工具')
        st.write('📱 此应用可在任何设备上访问')

    # 输入表单 
    st.sidebar.title('输入特征')
    preg = st.sidebar.slider('怀孕次数', 0, 17, 3)
    glucose = st.sidebar.slider('血糖', 0, 199, 117)
    bp = st.sidebar.slider('血压', 0, 122, 72)
    skinthickness = st.sidebar.slider('皮肤厚度', 0, 99, 23)
    insulin = st.sidebar.slider('胰岛素', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('糖尿病家族史', 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider('年龄', 21, 81, 29)

    input_data = [[preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]]
    input_data_nparray = np.asarray(input_data)
    # 对输入数据进行标准化处理
    scaled_input_data = scaler.transform(input_data_nparray)

    prediction = None 

    if st.sidebar.button('预测', key='predict_button'):
        prediction = model.predict(scaled_input_data)

    # 添加一个分隔线
    st.markdown("---")
    
    # 预测结果部分
    st.subheader('预测结果')
    if prediction is not None:
        if prediction[0] == 1:
            st.warning('⚠️ 根据输入的特征，模型预测该人患有糖尿病。')
        else:
            st.success('✅ 根据输入的特征，模型预测该人没有糖尿病。')
    else:
        st.info('请在左侧输入健康指标后点击"预测"按钮')

    # 添加一个分隔线
    st.markdown("---")
    
    # 数据集摘要
    st.header('数据集摘要')
    st.write(df.describe())
    
    # 添加注意事项
    st.caption("注意：本预测系统仅供参考，不能替代专业医疗诊断。如有健康问题，请咨询专业医疗人员。")
    
    # 添加作者信息
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2025 李卓阳 曲阜师范大学")

if __name__ == '__main__':
    app()
