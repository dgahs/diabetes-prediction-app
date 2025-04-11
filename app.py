import numpy as np
import pandas as pd
import streamlit as st
import sqlite3
import matplotlib.pyplot as plt
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib

# 设置 matplotlib 支持英文字体，避免中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 服务器上可用的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建数据库连接和表
def init_db():
    conn = sqlite3.connect('health_records.db')
    c = conn.cursor()

    # 检查 records 表是否存在
    c.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='records'")
    if c.fetchone()[0] == 0:
        # 如果 records 表不存在，则创建表
        c.execute('''CREATE TABLE records
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    patient_id INTEGER NOT NULL,    
                    timestamp TEXT,   
                    pregnancies INTEGER,   
                    glucose REAL,   
                    bloodpressure REAL,   
                    skinthickness REAL,   
                    insulin REAL,   
                    bmi REAL,   
                    dpf REAL,   
                    age INTEGER,   
                    prediction INTEGER)''')
        print("Created 'records' table.")
    
    conn.commit()
    conn.close()

# 初始化数据库
init_db()

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

# 创建并训练逻辑回归模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 评估模型性能
train_y_pred = model.predict(X_train)
test_y_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_y_pred)
test_accuracy = accuracy_score(y_test, test_y_pred)

print(f"训练集准确率：{train_accuracy:.4f}")
print(f"测试集准确率：{test_accuracy:.4f}")
print("\n分类报告：")
print(classification_report(y_test, test_y_pred))

# 保存预测结果到数据库
def save_to_db(patient_id, features, prediction):
    pregnancies, glucose, bp, skinthickness, insulin, bmi, dpf, age = features
    conn = sqlite3.connect('health_records.db')
    c = conn.cursor()
    
    # 确保 prediction 是整数类型
    prediction_int = int(prediction)
    
    # 插入记录并显示调试信息
    st.write(f"存储预测结果：患者 ID={patient_id}, 预测结果={prediction_int}")
    
    c.execute("INSERT INTO records (patient_id, timestamp, pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age, prediction) VALUES (?, DATETIME('now', 'localtime'), ?, ?, ?, ?, ?, ?, ?, ?, ?)",   
            (patient_id, pregnancies, glucose, bp, skinthickness, insulin, bmi, dpf, age, prediction_int))   
    
    conn.commit()
    conn.close()

# 获取历史记录
def get_history_records(page=1, limit=10, date=None):
    conn = sqlite3.connect('health_records.db')
    c = conn.cursor()

    query = "SELECT * FROM records"
    params = []

    if date:
        query += " WHERE DATE(timestamp) = ?"
        params.append(date)

    # 获取总记录数
    total_records = c.execute(query, tuple(params)).fetchall()

    # 添加排序和分页
    query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    params.append(limit)
    params.append((page - 1) * limit)  # 计算偏移量

    records = c.execute(query, tuple(params)).fetchall()
    conn.close()
    return records, len(total_records)  # 返回记录和总条数

# 获取特定患者记录
def get_patient_records(patient_id):  
    conn = sqlite3.connect('health_records.db')
    c = conn.cursor()

    query = "SELECT * FROM records WHERE patient_id = ? ORDER BY timestamp DESC"
    records = c.execute(query, (patient_id,)).fetchall()
    conn.close()
    return records

# 获取糖尿病统计数据
def get_diabetes_statistics():
    conn = sqlite3.connect('health_records.db')
    c = conn.cursor()
    
    # 修改 SQL 查询，确保处理不同数据类型，将所有非 1 的值视为 0
    total_diabetic = c.execute("SELECT COUNT(*) FROM records WHERE prediction = 1").fetchone()[0]
    total_normal = c.execute("SELECT COUNT(*) FROM records WHERE prediction IS NULL OR prediction != 1").fetchone()[0]
    
    # 总记录数
    total_all = c.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    if total_all > 0:
        st.write(f"总记录数：{total_all}, 糖尿病：{total_diabetic}, 正常：{total_normal}")
    
    conn.close()

    return total_diabetic, total_normal

# 获取正常患者和糖尿病患者各项指标的均值
def get_diabetes_means():
    conn = sqlite3.connect('health_records.db')
    c = conn.cursor()

    # 修改 SQL 查询，对 prediction 进行处理以确保正确分组
    # 计算正常患者的均值 (所有非 1 的值视为正常)
    normal_means = c.execute("""
        SELECT AVG(pregnancies), AVG(glucose), AVG(bloodpressure), AVG(skinthickness),
               AVG(insulin), AVG(bmi), AVG(age)
        FROM records WHERE prediction IS NULL OR prediction != 1
    """).fetchone()

    # 计算糖尿病患者的均值 (确保 prediction = 1)
    diabetic_means = c.execute("""
        SELECT AVG(pregnancies), AVG(glucose), AVG(bloodpressure), AVG(skinthickness),
               AVG(insulin), AVG(bmi), AVG(age)
        FROM records WHERE prediction = 1
    """).fetchone()

    conn.close()

    return {
        'normal': normal_means,
        'diabetic': diabetic_means
    }

# 从数据库获取所有记录用于数据可视化
def get_xxdata_from_db():
    conn = sqlite3.connect('health_records.db')
    try:
        # 查询时直接处理 prediction 列，确保所有错误值变为 0（正常）
        query = """
        SELECT 
            pregnancies, 
            glucose, 
            bloodpressure, 
            skinthickness, 
            insulin, 
            bmi, 
            dpf, 
            age, 
            CASE
                WHEN prediction IS NULL OR prediction != 1 THEN 0
                ELSE 1
            END as prediction 
        FROM records
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"获取数据时出错: {str(e)}")
        conn.close()
        return pd.DataFrame()  # 返回空DataFrame

def app():
    # 设置页面标题和图标
    st.set_page_config(
        page_title="糖尿病预测应用",
        page_icon="🏥",
        layout="wide",
    )

    # 创建导航菜单
    menu = ["预测", "历史记录", "数据可视化", "患者管理", "统计分析"]
    choice = st.sidebar.selectbox("导航", menu)

    # 预测页面
    if choice == "预测":
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
        patient_id = st.sidebar.number_input('患者 ID', min_value=1, step=1, value=1000)
        preg = st.sidebar.slider('怀孕次数', 0, 17, 3)
        glucose = st.sidebar.slider('血糖', 0, 199, 117)
        bp = st.sidebar.slider('血压', 0, 122, 72)
        skinthickness = st.sidebar.slider('皮肤厚度', 0, 99, 23)
        insulin = st.sidebar.slider('胰岛素', 0, 846, 30)
        bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
        dpf = st.sidebar.slider('糖尿病家族史', 0.078, 2.42, 0.3725, 0.001)
        age = st.sidebar.slider('年龄', 21, 81, 29)

        features = [preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]
        input_data = [features]
        input_data_nparray = np.asarray(input_data)
        # 对输入数据进行标准化处理
        scaled_input_data = scaler.transform(input_data_nparray)

        prediction = None
        prediction_text = None

        if st.sidebar.button('预测', key='predict_button'):
            # 使用逻辑回归模型进行预测
            prediction = model.predict(scaled_input_data)
            
            # 保存预测结果到数据库
            save_to_db(patient_id, features, prediction[0])
            
            if prediction[0] == 1:
                prediction_text = "⚠️ 根据输入的特征，模型预测该人患有糖尿病。"
            else:
                prediction_text = "✅ 根据输入的特征，模型预测该人没有糖尿病。"

        # 添加一个分隔线
        st.markdown("---")
        
        # 预测结果部分
        st.subheader('预测结果')
        if prediction_text:
            if prediction[0] == 1:
                st.warning(prediction_text)
            else:
                st.success(prediction_text)
        else:
            st.info('请在左侧输入健康指标后点击"预测"按钮')

        # 添加一个分隔线
        st.markdown("---")
        
        # 数据集摘要
        st.header('数据集摘要')
        st.write(df.describe())
        
    # 历史记录页面
    elif choice == "历史记录":
        st.title("历史预测记录")
        
        # 日期筛选
        col1, col2 = st.columns([1, 3])
        with col1:
            date_filter = st.date_input("选择日期", datetime.date.today())
        with col2:
            if st.button("查询"):
                date_str = date_filter.strftime("%Y-%m-%d")
                records, total = get_history_records(date=date_str)
                st.success(f"找到 {total} 条记录")
            else:
                records, total = get_history_records()
        
        # 分页控件
        page = st.number_input("页码", min_value=1, value=1, step=1)
        records, total = get_history_records(page=page)
        
        # 显示记录
        if records:
            # 转换记录为 DataFrame 以便于显示
            columns = ["ID", "患者 ID", "时间戳", "怀孕次数", "血糖", "血压", "皮肤厚度", "胰岛素", "BMI", "糖尿病家族史", "年龄", "预测结果"]
            records_df = pd.DataFrame(records, columns=columns)
            st.dataframe(records_df)
            
            # 添加分页信息
            st.caption(f"显示第 {page} 页，总共 {total} 条记录")
        else:
            st.info("没有找到历史记录")
        
        # 数据导出按钮
        if st.button("导出数据为 CSV"):
            if 'records_df' in locals():
                records_df.to_csv("diabetes_predictions.csv", index=False)
                st.success("数据已导出到 diabetes_predictions.csv")
            else:
                st.warning("没有数据可导出")
    
    # 数据可视化页面
    elif choice == "数据可视化":
        st.title("数据可视化分析")
        
        # 获取数据库中的记录
        db_data = get_xxdata_from_db()
        
        if not db_data.empty:
            st.write("数据库中共有 {} 条记录".format(len(db_data)))
            
            # 数据概览
            st.subheader("数据概览")
            st.dataframe(db_data.head())
            
            # 创建多个标签页进行不同的可视化
            tabs = ["患者分布", "特征分析", "相关性", "比较分析"]
            selected_tab = st.radio("选择可视化类型", tabs, horizontal=True)
            
            if selected_tab == "患者分布":
                st.subheader("患者预测结果分布")
                
                try:
                    # 确保 prediction 列处理正确，所有非 1 值设为 0
                    db_data['prediction'] = db_data['prediction'].apply(lambda x: 1 if x == 1 else 0)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    diabetic_count = db_data[db_data['prediction'] == 1].shape[0]
                    normal_count = db_data[db_data['prediction'] == 0].shape[0]
                    
                    # 检查是否有数据
                    if diabetic_count > 0 or normal_count > 0:
                        ax.pie([diabetic_count, normal_count], 
                              labels=['糖尿病患者', '正常人群'], 
                              autopct='%1.1f%%',
                              colors=['#ff9999','#66b3ff'])
                        ax.set_title("患者预测结果分布")
                        st.pyplot(fig)
                    else:
                        st.warning("没有有效的预测结果数据")
                except Exception as e:
                    st.error(f"处理预测结果数据时出错：{str(e)}")
                    st.info("尝试检查数据库中的预测结果格式")
                
            elif selected_tab == "特征分析":
                st.subheader("特征数据分布")
                
                # 选择要可视化的特征
                features = ["pregnancies", "glucose", "bloodpressure", "skinthickness", 
                           "insulin", "bmi", "dpf", "age"]
                feature_names = ["怀孕次数", "血糖", "血压", "皮肤厚度", 
                                "胰岛素", "BMI", "糖尿病家族史", "年龄"]
                
                selected_feature = st.selectbox("选择特征", feature_names)
                feature_idx = feature_names.index(selected_feature)
                
                # 根据预测结果分组显示特征
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # 计算糖尿病患者和正常人群的直方图
                    diabetic_data = db_data[db_data['prediction'] == 1][features[feature_idx]]
                    normal_data = db_data[db_data['prediction'] == 0][features[feature_idx]]
                    
                    # 直方图
                    ax.hist(diabetic_data, alpha=0.5, label='Diabetic', bins=15, color='#ff9999')
                    ax.hist(normal_data, alpha=0.5, label='Normal', bins=15, color='#66b3ff')
                    
                    ax.set_xlabel(feature_names[feature_idx])
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'{feature_names[feature_idx]} Distribution')
                    ax.legend()
                    
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"绘制特征分布时出错: {str(e)}")
                    st.info("请确保数据库中有足够的记录")
            
            elif selected_tab == "相关性":
                st.subheader("特征相关性分析")
                
                try:
                    # 复制数据，以免修改原始数据
                    numeric_data = db_data.copy()
                    
                    # 转换所有列为数值类型
                    for col in numeric_data.columns:
                        try:
                            numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
                        except:
                            st.warning(f"列 '{col}' 转换为数值类型失败，将从相关性分析中排除")
                            numeric_data = numeric_data.drop(col, axis=1)
                    
                    # 处理缺失值
                    numeric_data = numeric_data.dropna()
                    
                    if len(numeric_data) > 0 and len(numeric_data.columns) > 1:
                        # 计算相关性矩阵
                        corr_matrix = numeric_data.corr()
                        
                        # 显示热力图
                        fig, ax = plt.subplots(figsize=(10, 8))
                        im = ax.imshow(corr_matrix, cmap='coolwarm')
                        
                        # 添加每个单元格的数值
                        for i in range(len(corr_matrix.columns)):
                            for j in range(len(corr_matrix.index)):
                                text = ax.text(j, i, round(corr_matrix.iloc[i, j], 2),
                                            ha="center", va="center", color="black")
                        
                        # 设置坐标轴
                        ax.set_xticks(np.arange(len(corr_matrix.columns)))
                        ax.set_yticks(np.arange(len(corr_matrix.index)))
                        ax.set_xticklabels(corr_matrix.columns)
                        ax.set_yticklabels(corr_matrix.index)
                        
                        # 旋转 x 轴标签
                        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                        
                        ax.set_title("Feature Correlation Heatmap")
                        fig.colorbar(im)
                        fig.tight_layout()
                        
                        st.pyplot(fig)
                    else:
                        st.error("数据中没有足够的有效数值数据来计算相关性")
                except Exception as e:
                    st.error(f"相关性分析出错：{str(e)}")
                    st.info("请检查数据格式和质量")
                
            elif selected_tab == "比较分析":
                st.subheader("糖尿病患者与正常人群特征比较")
                
                try:
                    # 处理预测列
                    db_data['prediction'] = pd.to_numeric(db_data['prediction'], errors='coerce')
                    db_data = db_data.dropna(subset=['prediction'])
                    db_data['prediction'] = db_data['prediction'].astype(int)
                    
                    # 对比数据
                    if 0 in db_data['prediction'].values and 1 in db_data['prediction'].values:
                        # 计算每组的均值
                        db_means = db_data.groupby('prediction').mean()
                        
                        # 分组条形图
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # 创建柱状图
                        features = [col for col in db_means.columns if col != 'prediction']
                        x = np.arange(len(features))  # 特征位置
                        width = 0.35  # 柱的宽度
                        
                        # 使用值而不是索引
                        normal_values = db_means.loc[0].values
                        diabetic_values = db_means.loc[1].values
                        
                        normal = ax.bar(x - width/2, normal_values, width, label='Normal', color='#66b3ff')
                        diabetic = ax.bar(x + width/2, diabetic_values, width, label='Diabetic', color='#ff9999')
                        
                        # 添加一些文本元素
                        ax.set_title('Comparing Features: Diabetic vs Normal')
                        ax.set_xticks(x)
                        ax.set_xticklabels(features, rotation=45, ha='right')
                        ax.legend()
                        
                        # 自动调整布局
                        fig.tight_layout()
                        
                        st.pyplot(fig)
                    else:
                        st.warning("数据中缺少正常人群或糖尿病患者的记录，无法进行比较")
                except Exception as e:
                    st.error(f"比较分析出错：{str(e)}")
                    st.info("请确保有足够的两种类型的数据进行比较")
        else:
            st.info("数据库中没有记录，请先进行一些预测")
    
    # 患者管理页面
    elif choice == "患者管理":
        st.title("患者记录管理")
        
        # 创建两列布局
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("患者查询")
            patient_id = st.number_input("输入患者 ID", min_value=1, step=1)
            
            if st.button("查询"):
                records = get_patient_records(patient_id)
                if records:
                    st.success(f"找到 {len(records)} 条患者记录")
                    
                    # 转换记录为 DataFrame
                    columns = ["ID", "患者 ID", "时间戳", "怀孕次数", "血糖", "血压", "皮肤厚度", "胰岛素", "BMI", "糖尿病家族史", "年龄", "预测结果"]
                    records_df = pd.DataFrame(records, columns=columns)
                    st.dataframe(records_df)
                    
                    # 绘制患者的预测历史趋势
                    if len(records) > 1:
                        st.subheader("患者预测历史趋势")
                        
                        # 提取时间和预测结果
                        dates = [record[2] for record in records]
                        predictions = [record[11] for record in records]
                        
                        # 创建趋势图
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(dates, predictions, marker='o', linestyle='-', color='#ff9999')
                        ax.set_title(f"Patient {patient_id}'s Diabetes Risk Trend")
                        ax.set_xlabel("Date")
                        ax.set_ylabel("Prediction (1=Diabetic, 0=Normal)")
                        ax.set_yticks([0, 1])
                        ax.set_yticklabels(['Normal', 'Diabetic'])
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        
                        st.pyplot(fig)
                else:
                    st.warning(f"没有找到 ID 为 {patient_id} 的患者记录")
        
        with col2:
            st.subheader("患者统计")
            
            total_diabetic, total_normal = get_diabetes_statistics()
            
            # 显示统计信息
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("糖尿病患者数量", total_diabetic)
            with col_b:
                st.metric("正常人群数量", total_normal)
            
            # 患者风险饼图
            if total_diabetic + total_normal > 0:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie([total_diabetic, total_normal], 
                       labels=['Diabetic Risk', 'Normal'], 
                       autopct='%1.1f%%',
                       colors=['#ff9999','#66b3ff'],
                       explode=(0.1, 0))
                ax.set_title("Patient Risk Distribution")
                st.pyplot(fig)
    
    # 统计分析页面
    elif choice == "统计分析":
        st.title("统计分析")
        
        # 获取原始数据集和预测数据的统计信息
        db_data = get_xxdata_from_db()
        
        st.subheader("数据集对比")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("原始数据集统计")
            st.dataframe(diabetes_mean_df)
        
        with col2:
            if not db_data.empty:
                st.write("预测数据统计")
                try:
                    # 确保 prediction 列处理正确，所有非 1 值设为 0
                    db_data['prediction'] = db_data['prediction'].apply(lambda x: 1.0 if x == 1.0 else 0.0)
                    
                    # 尝试进行分组计算
                    grouped_means = db_data.groupby('prediction').mean()
                    st.dataframe(grouped_means)
                except Exception as e:
                    st.error(f"计算预测数据统计时出错：{str(e)}")
                    st.info("正在显示所有数据的统计信息，不进行分组")
                    st.dataframe(db_data.describe())
            else:
                st.info("数据库中没有预测记录")
        
        # 获取病患和正常人群的特征均值比较
        means_data = get_diabetes_means()
        
        # 检查是否有有效数据进行比较
        if (means_data['normal'][0] is not None and 
            means_data['diabetic'][0] is not None and
            not all(x is None for x in means_data['normal']) and
            not all(x is None for x in means_data['diabetic'])):
            
            st.subheader("特征均值比较")
            
            # 创建比较条形图
            feature_names = ["怀孕次数", "血糖", "血压", "皮肤厚度", "胰岛素", "BMI", "年龄"]
            
            # 转换数据为数值类型
            norm_vals = [float(x) if x is not None else 0 for x in means_data['normal']]
            diab_vals = [float(x) if x is not None else 0 for x in means_data['diabetic']]
            
            if all(v == 0 for v in norm_vals) and all(v == 0 for v in diab_vals):
                st.warning("没有足够的数据进行特征均值比较")
            else:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                x = np.arange(len(feature_names))
                width = 0.35
                
                # 绘制柱状图
                ax.bar(x - width/2, norm_vals, width, label='Normal', color='#66b3ff')
                ax.bar(x + width/2, diab_vals, width, label='Diabetic', color='#ff9999')
                
                ax.set_title('Average Values by Group')
                ax.set_xticks(x)
                ax.set_xticklabels(feature_names, rotation=45, ha='right')
                ax.legend()
                
                fig.tight_layout()
                st.pyplot(fig)
                
                # 增加分析解读
                st.subheader("分析解读")
                
                # 计算百分比差异
                pct_diff = []
                for i in range(len(norm_vals)):
                    if norm_vals[i] > 0:
                        diff = (diab_vals[i] - norm_vals[i]) / norm_vals[i] * 100
                        pct_diff.append(diff)
                    else:
                        pct_diff.append(0)
                
                # 找出差异最大的特征
                if pct_diff:
                    max_diff_idx = np.argmax(np.abs(pct_diff))
                    
                    st.write(f"在各项指标中，**{feature_names[max_diff_idx]}** 的差异最为显著，糖尿病患者比正常人群高出约 {abs(pct_diff[max_diff_idx]):.1f}%。")
                    
                    if diab_vals[1] > norm_vals[1]:
                        st.write("**血糖**水平在糖尿病患者中明显较高，这与医学研究一致。")
                        
                    if diab_vals[5] > norm_vals[5]:
                        st.write("**BMI**(体重指数) 在糖尿病患者中也有显著升高，表明肥胖可能是糖尿病的危险因素。")
                        
                    st.write("这些数据分析结果可以帮助医疗专业人员更好地识别糖尿病风险因素，为患者提供更精准的预防建议。")
        else:
            st.info("数据库中的记录不足以进行统计分析")
    
    # 添加作者信息
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2025 李卓阳 曲阜师范大学")

if __name__ == '__main__':
    app()
