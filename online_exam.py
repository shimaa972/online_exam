import streamlit as st
import pandas as pd
import pyodbc
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# الاتصال بقاعدة البيانات
def load_data():
    try:
        conn = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=db22845.public.databaseasp.net;'
            'DATABASE=db22845;'
            'UID=db22845;'
            'PWD=12345678'
        )
        
        query = """
        SELECT 
            S.StudentKey,
            S.FullName AS StudentName,
            S.Age AS StudentAge,
            B.Branch_name AS Branch,
            C.Certificate_Name AS Certificate,
            SA.salary AS Salary,
            SA.job_type AS JobType,
            CASE WHEN SA.salary IS NOT NULL THEN 1 ELSE 0 END AS IsEmployed,
            (SELECT COUNT(*) FROM [db22845].[dbo].[DimStudent]) AS TotalStudents,
            (SELECT COUNT(DISTINCT [StudentKey]) FROM [db22845].[dbo].[Student_Activity] 
             WHERE [salary] IS NOT NULL) AS EmployedStudents
        FROM 
            dbo.DimStudent S
        LEFT JOIN 
            dbo.Student_Activity SA ON S.StudentKey = SA.StudentKey
        LEFT JOIN 
            dbo.dimbranch B ON SA.BranchKey = B.BranchKey
        LEFT JOIN 
            dbo.dimcertificate C ON SA.CertificateKey = C.CertificateKey
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        # تنظيف البيانات
        df['Branch'] = df['Branch'].fillna('Unknown')
        df['Certificate'] = df['Certificate'].fillna('Unknown')
        df['JobType'] = df['JobType'].fillna('Unknown')
        df['StudentAge'] = df['StudentAge'].fillna(df['StudentAge'].median())
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# دالة الـ Dropdown المعدلة لإظهار الاسم و ID فقط
def student_selectbox(df):
    unique_students = df[['StudentKey', 'StudentName']].drop_duplicates()
    unique_students = unique_students.sort_values('StudentName')
    options = [f"{row['StudentName']} (ID: {row['StudentKey']})" 
              for _, row in unique_students.iterrows()]
    selected = st.selectbox("Select Student", options)
    student_id = int(selected.split('(ID: ')[1].replace(')', ''))
    student_name = selected.split(' (ID:')[0]
    return student_id, student_name

# لوحة التحليل
def analytics_dashboard(df):
    st.subheader("📊 Employment Analytics")
    
    total = df['TotalStudents'].iloc[0]
    employed = df['EmployedStudents'].iloc[0]
    
    col1, col2 = st.columns(2)
    
    card_style = """
        <div style="background-color:{bg_color}; padding:20px; border-radius:10px; color:white; 
             height: 180px; display: flex; flex-direction: column; align-items: center; justify-content: center;
             box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); margin: 10px 0;">
            <h3 style="text-align:center; margin:0; color:white;">{title}</h3>
            <p style="text-align:center; font-size:36px; margin:10px 0;">{value}</p>
            <p style="text-align:center; font-size:16px; margin:0; color:white;">{rate}</p>
        </div>
    """
    
    with col1:
        st.markdown(card_style.format(
            bg_color='#900d06', 
            title="Total Students", 
            value=total, 
            rate=""
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(card_style.format(
            bg_color='#808080',
            title="Employed Students", 
            value=employed, 
            rate=f"{employed/total:.1%} employment rate"
        ), unsafe_allow_html=True)
    
    fig1 = px.pie(
        names=['Employed', 'Not Employed'],
        values=[employed, total-employed],
        title='Employment Distribution',
        color=['Employed', 'Not Employed'],
        color_discrete_map={'Employed': '#808080', 'Not Employed': '#900d06'}
    )
    st.plotly_chart(fig1, use_container_width=True)

# لوحة التنبؤ
def prediction_dashboard(df):
    st.subheader("🔮 Career Prediction")
    
    # تصميم البطاقة الشخصية للطالب
    student_card = """
    <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; 
         box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1); margin-bottom:20px;">
        <h3 style="color:#333; border-bottom:1px solid #ddd; padding-bottom:10px;">📋 Student Profile</h3>
        {content}
    </div>
    """
    
    selected_student_key, selected_student_name = student_selectbox(df)
    student_data = df[df['StudentKey'] == selected_student_key].iloc[0]
    
    profile_content = f"""
    <p><strong>Name:</strong> {student_data['StudentName']}</p>
    <p><strong>Age:</strong> {student_data['StudentAge']}</p>
    <p><strong>Branch:</strong> {student_data['Branch']}</p>
    <p><strong>Certificate:</strong> {student_data['Certificate']}</p>
    <p><strong>Current Status:</strong> {'<span style="color:green;">✅ Employed</span>' if student_data['IsEmployed'] else '<span style="color:red;">❌ Not Employed</span>'}
    """
    
    if student_data['IsEmployed']:
        profile_content += f"""<p><strong>Current Salary:</strong> ${student_data['Salary']:,.0f}</p>"""
    
    st.markdown(student_card.format(content=profile_content), unsafe_allow_html=True)

    # نموذج التنبؤ
    # st.markdown("""
    # <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; 
    #      box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1); margin-bottom:20px;">
    #     <h3 style="color:#333; border-bottom:1px solid #ddd; padding-bottom:10px;">🎯 Employment Prediction Model</h3>
    # </div>
    # """, unsafe_allow_html=True)
    
    
    # تحضير البيانات
    features = df[['Branch', 'Certificate', 'StudentAge', 'JobType']]
    features = pd.get_dummies(features, columns=['Branch', 'Certificate', 'JobType'])
    target = df['IsEmployed']
    
    # تقسيم البيانات
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    # تدريب النموذج
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # تقييم النموذج
    accuracy = accuracy_score(y_test, model.predict(X_test))
    # st.write(f"Model Accuracy: {accuracy:.1%}")
    
    # تحضير بيانات الطالب للتنبؤ
    student_features = pd.DataFrame({
        'StudentAge': [student_data['StudentAge']],
        **{col: 0 for col in features.columns if col != 'StudentAge' }
    })
    
    for col in ['Branch_'+student_data['Branch'], 
                'Certificate_'+student_data['Certificate'],
                'JobType_'+student_data['JobType']]:
        if col in student_features.columns:
            student_features[col] = 1
    
    # التنبؤ
    try:
        proba = model.predict_proba(student_features)[0][1]
        prediction = model.predict(student_features)[0]
        
        prediction_card = """
        <div style="background-color:{color}; padding:15px; border-radius:10px; color:white; 
             margin:10px 0; text-align:center;">
            <h4 style="margin:0;">Employment Probability</h4>
            <p style="font-size:24px; margin:5px 0;"><strong>{proba:.0%}</strong></p>
            <p style="margin:0;">{status}</p>
        </div>
        """
        
        if prediction:
            st.markdown(prediction_card.format(
                color='#28a745',
                proba=proba,
                status="Likely to be employed"
            ), unsafe_allow_html=True)
        else:
            st.markdown(prediction_card.format(
                color='#dc3545',
                proba=proba,
                status="Unlikely to be employed"
            ), unsafe_allow_html=True)
        
        # تفسير النتائج
        st.markdown("""
        <div style="background-color:#f8f9fa; padding:15px; border-radius:10px; 
             border-left:4px solid #6c757d; margin-top:20px;">
            <h4 style="color:#333; margin-top:0;">📌 Prediction Factors:</h4>
            <ul style="margin-bottom:0;">
                <li><strong>Branch:</strong> Different branches have varying employment rates</li>
                <li><strong>Certificate:</strong> Specialized certificates increase employability</li>
                <li><strong>Age:</strong> Younger students tend to have better employment chances</li>
                <li><strong>Job Type:</strong> Some job types are more in-demand than others</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# الواجهة الرئيسية
def main():
    st.set_page_config(
        page_title="ITI Student Analytics Dashboard", 
        layout="centered",  # لجعل المحتوى في منتصف الشاشة
        page_icon="🎓"
    )
    
    # رأس الصفحة مع اللوجو والعنوان جنباً إلى جنب
    header_col1, header_col2 = st.columns([1, 4])
    with header_col1:
        st.image("logo.jpg", width=60)  # تصغير حجم اللوجو
    with header_col2:
        st.title("ITI Student Analytics")
    
    df = load_data()
    if df.empty:
        st.warning("No data loaded. Please check database connection.")
        return
    
    # تصميم التبويبات
    tab1, tab2 = st.tabs([
        "📊 Analytics Dashboard", 
        "🔮 Prediction Dashboard"
    ])
    
    with tab1:
        analytics_dashboard(df)
    with tab2:
        prediction_dashboard(df)

if __name__ == "__main__":
    main()