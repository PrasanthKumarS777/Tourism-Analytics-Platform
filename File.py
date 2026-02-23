import streamlit as st, pandas as pd, numpy as np, os, warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
import plotly.express as px, plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Tourism Intelligence Hub", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>h1{background:linear-gradient(120deg,#1e3c72,#2a5298);color:white;padding:1.5rem;border-radius:15px;text-align:center;font-weight:800;letter-spacing:2px;box-shadow:0 10px 30px rgba(0,0,0,.2);margin-bottom:2rem}.stTabs [data-baseweb="tab"]{background:linear-gradient(135deg,#667eea,#764ba2);color:white;border-radius:10px;padding:12px 24px;font-weight:600}.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#f093fb,#f5576c);transform:scale(1.05)}.info-box{background:white;padding:25px;border-radius:15px;box-shadow:0 10px 25px rgba(0,0,0,.1);margin:15px 0;border-left:6px solid #667eea}div[data-testid="stMetric"]{background:white;padding:20px;border-radius:12px;box-shadow:0 5px 20px rgba(0,0,0,.1);border-left:5px solid #667eea}.stButton>button{background:linear-gradient(135deg,#667eea,#764ba2);color:white;border:none;border-radius:10px;padding:12px 24px;font-weight:600}section[data-testid="stSidebar"]{background:linear-gradient(180deg,#2b5876,#4e4376)}section[data-testid="stSidebar"] *{color:white!important}</style>""", unsafe_allow_html=True)
st.markdown("<h1>🌍 TOURISM INTELLIGENCE HUB</h1>", unsafe_allow_html=True)
st.markdown("""<div style="text-align:center;padding:10px;background:rgba(255,255,255,.9);border-radius:10px;margin-bottom:20px"><p style="font-size:18px;color:#64748b;margin:0"><strong>Advanced Analytics & ML-Powered Insights for Tourism Industry</strong></p><p style="font-size:14px;color:#94a3b8;margin:5px 0 0 0">Real-time predictions • Personalized recommendations • Business intelligence</p></div>""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    try:
        d = {n: pd.read_excel(f"Dataset/{n}.xlsx") for n in ["City","User","Transaction","Item","Type"] if os.path.exists(f"Dataset/{n}.xlsx")}
        if len(d) < 5: st.warning("⚠️ Some files missing"); return pd.DataFrame()
        return d["Transaction"].merge(d["User"],on="UserId",how="left").merge(d["Item"],on="AttractionId",how="left").merge(d["Type"],on="AttractionTypeId",how="left").merge(d["City"].add_prefix("User_"),left_on="CityId",right_on="User_CityId",how="left").dropna()
    except Exception as e: st.error(f"❌ {e}"); return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def train_models(_df):
    if _df.empty: return [None]*8
    Xr,yr = _df[['VisitYear','VisitMonth','AttractionTypeId','ContinentId','CountryId']],_df['Rating']
    Xc,le = _df[['VisitYear','VisitMonth','ContinentId','CountryId']],LabelEncoder()
    yc = le.fit_transform(_df['VisitMode'])
    Xtr,Xte,ytr,yte = train_test_split(Xr,yr,test_size=0.25,random_state=42)
    Xtc,Xtec,ytc,ytec = train_test_split(Xc,yc,test_size=0.25,random_state=42)
    reg = RandomForestRegressor(random_state=42,n_estimators=100,max_depth=15,n_jobs=-1).fit(Xtr,ytr)
    clf = XGBClassifier(eval_metric='logloss',random_state=42,verbosity=0,n_estimators=100,max_depth=8,learning_rate=0.1).fit(Xtc,ytc)
    r2,mae,acc = r2_score(yte,reg.predict(Xte)),mean_absolute_error(yte,reg.predict(Xte)),accuracy_score(ytec,clf.predict(Xtec))
    fi_r = pd.DataFrame({'feature':Xr.columns,'importance':reg.feature_importances_}).sort_values('importance',ascending=False)
    fi_c = pd.DataFrame({'feature':Xc.columns,'importance':clf.feature_importances_}).sort_values('importance',ascending=False)
    return reg,clf,le,r2,mae,acc,fi_r,fi_c

@st.cache_data(show_spinner=False)
def get_recommendations(_df, uid, top_n=10):
    if _df.empty or 'Attraction' not in _df.columns: return pd.DataFrame({'Attraction':['No Data'],'Rating':[0],'Confidence':[0]})
    pivot = _df.pivot_table(index='UserId',columns='Attraction',values='Rating').fillna(0)
    if uid not in pivot.index:
        p = _df.groupby('Attraction').agg(Rating=('Rating','mean'),Visits=('Rating','count')).reset_index()
        p['Confidence'] = p['Rating']*np.log1p(p['Visits'])/10; return p.nlargest(top_n,'Confidence')[['Attraction','Rating','Confidence']]
    sim = pd.DataFrame(cosine_similarity(pivot),index=pivot.index,columns=pivot.index)
    sim_users = sim[uid].sort_values(ascending=False)[1:11].index
    r = _df[_df['UserId'].isin(sim_users)].groupby('Attraction').agg(Rating=('Rating','mean'),Count=('Rating','count')).reset_index()
    r['Confidence'] = r['Rating']*np.sqrt(r['Count'])/5
    visited = _df[_df['UserId']==uid]['Attraction'].unique()
    return r[~r['Attraction'].isin(visited)].nlargest(top_n,'Confidence')[['Attraction','Rating','Confidence']]

df = load_data()
if df.empty: st.error("❌ Data Loading Failed"); st.stop()
reg,clf,le,r2,mae,acc,fi_r,fi_c = train_models(df)

with st.sidebar:
    st.markdown("## ⚙️ CONTROL PANEL\n---\n### 📅 TIME PERIOD")
    yr_range = st.slider("Year Range",int(df['VisitYear'].min()),int(df['VisitYear'].max()),(int(df['VisitYear'].min()),int(df['VisitYear'].max())))
    dff = df[(df['VisitYear']>=yr_range[0])&(df['VisitYear']<=yr_range[1])]
    st.markdown("### 🎯 PREDICTION INPUTS")
    year=st.selectbox("📅 Visit Year",sorted(df['VisitYear'].unique()),index=len(df['VisitYear'].unique())-1)
    month=st.selectbox("📆 Month",sorted(df['VisitMonth'].unique()))
    atype=st.selectbox("🎭 Attraction Type",sorted(df['AttractionTypeId'].unique()))
    cont=st.selectbox("🌍 Continent",sorted(df['ContinentId'].unique()))
    ctry=st.selectbox("🗺️ Country",sorted(df['CountryId'].unique()))
    st.markdown("### 👤 USER SELECTION")
    uid=st.selectbox("User ID",sorted(df['UserId'].unique())[:100])
    st.markdown("---\n### 📊 MODEL PERFORMANCE")
    st.metric("Rating Model R²",f"{r2:.3f}"); st.metric("Mode Model Accuracy",f"{acc:.1%}")

kpi = lambda t,v,i: st.markdown(f'<div class="info-box"><h4 style="margin:0;color:#64748b;font-size:14px">{i} {t}</h4><h2 style="margin:10px 0 0 0;color:#1e293b;font-size:36px;font-weight:800">{v}</h2></div>',unsafe_allow_html=True)
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["📊 Executive Dashboard","🎯 ML Predictions","⭐ Smart Recommendations","📈 Deep Analytics","🔍 Trend Analysis","📋 Data Explorer"])

with tab1:
    st.markdown("## 📊 EXECUTIVE DASHBOARD\nReal-time KPIs and business metrics")
    c=st.columns(5)
    for col,(title,val,icon) in zip(c,[("Total Visits",f"{len(dff):,}","📋"),("Unique Users",f"{dff['UserId'].nunique():,}","👥"),("Attractions",f"{dff['Attraction'].nunique():,}","🏛️"),("Avg Rating",f"{dff['Rating'].mean():.2f}/5.0","⭐"),("Continents",f"{dff['ContinentId'].nunique()}","🌍")]):
        with col: kpi(title,val,icon)
    c1,c2=st.columns(2)
    with c1:
        yt=dff.groupby('VisitYear').size().reset_index(name='Visits'); f=go.Figure(go.Scatter(x=yt['VisitYear'],y=yt['Visits'],mode='lines+markers',line=dict(color='#667eea',width=4),fill='tozeroy',fillcolor='rgba(102,126,234,.2)')); f.update_layout(title="📈 Visit Trends",height=350,plot_bgcolor='rgba(255,255,255,.9)',paper_bgcolor='rgba(255,255,255,.9)'); st.plotly_chart(f,use_container_width=True)
    with c2:
        cd=dff['ContinentId'].value_counts().reset_index(); cd.columns=['Continent','Count']; f=px.pie(cd,values='Count',names='Continent',title="🌍 Geographic Distribution",hole=0.4,color_discrete_sequence=px.colors.sequential.RdBu); f.update_layout(height=350); st.plotly_chart(f,use_container_width=True)
    c1,c2=st.columns(2)
    mn=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    with c1:
        mo=dff.groupby('VisitMonth').size().reset_index(name='Visits'); mo['Month']=mo['VisitMonth'].apply(lambda x:mn[int(x)-1]); f=px.bar(mo,x='Month',y='Visits',title="📆 Seasonal Patterns",color='Visits',color_continuous_scale='Viridis'); f.update_layout(height=350); st.plotly_chart(f,use_container_width=True)
    with c2:
        f=px.histogram(dff,x='Rating',title="⭐ Rating Distribution",nbins=20,color_discrete_sequence=['#667eea']); f.update_layout(height=350); st.plotly_chart(f,use_container_width=True)

with tab2:
    st.markdown("## 🎯 MACHINE LEARNING PREDICTIONS")
    if reg and clf and le:
        c1,c2=st.columns(2)
        with c1:
            pr=max(1,min(5,reg.predict([[year,month,atype,cont,ctry]])[0]))
            st.markdown(f'<div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:3rem;border-radius:20px;text-align:center;box-shadow:0 15px 35px rgba(0,0,0,.3)"><h3 style="color:white;margin:0">⭐ PREDICTED RATING</h3><h1 style="color:white;font-size:5rem;margin:1.5rem 0;font-weight:900">{pr:.2f}</h1><p style="color:white;opacity:.95;font-size:18px;margin:0">out of 5.0 stars</p></div>',unsafe_allow_html=True)
            cc1,cc2=st.columns(2); cc1.metric("🎯 R² Score",f"{r2:.4f}"); cc2.metric("📊 MAE",f"{mae:.4f}")
            f=px.bar(fi_r,x='importance',y='feature',orientation='h',color='importance',color_continuous_scale='Blues',title="Feature Importance (Rating)"); f.update_layout(height=300,showlegend=False,yaxis={'categoryorder':'total ascending'}); st.plotly_chart(f,use_container_width=True)
        with c2:
            pm=le.inverse_transform(clf.predict([[year,month,cont,ctry]]))[0]; pp=clf.predict_proba([[year,month,cont,ctry]])[0]; conf=max(pp)*100
            st.markdown(f'<div style="background:linear-gradient(135deg,#f093fb,#f5576c);padding:3rem;border-radius:20px;text-align:center;box-shadow:0 15px 35px rgba(0,0,0,.3)"><h3 style="color:white;margin:0">🧭 RECOMMENDED MODE</h3><h1 style="color:white;font-size:3.5rem;margin:1.5rem 0;font-weight:900">{pm}</h1><p style="color:white;opacity:.95;font-size:18px;margin:0">Best travel mode</p><p style="color:white;margin:1rem 0 0 0">Confidence: {conf:.1f}%</p></div>',unsafe_allow_html=True)
            st.metric("🎯 Model Accuracy",f"{acc:.2%}")
            md=pd.DataFrame({'Mode':le.classes_,'Probability':pp*100}).sort_values('Probability',ascending=False)
            f=px.bar(md,x='Probability',y='Mode',orientation='h',color='Probability',color_continuous_scale='Reds',text='Probability'); f.update_traces(texttemplate='%{text:.1f}%',textposition='outside'); f.update_layout(height=300,showlegend=False,yaxis={'categoryorder':'total ascending'}); st.plotly_chart(f,use_container_width=True)

with tab3:
    st.markdown(f"## ⭐ PERSONALIZED RECOMMENDATIONS FOR USER #{uid}")
    recs=get_recommendations(df,uid,top_n=15)
    if not recs.empty and recs.iloc[0]['Rating']>0:
        c1,c2,c3=st.columns(3); c1.metric("🎫 User Visits",len(df[df['UserId']==uid])); c2.metric("⭐ Avg Rating Given",f"{df[df['UserId']==uid]['Rating'].mean():.2f}"); c3.metric("🎯 Recommendations",len(recs))
        c1,c2=st.columns([3,2])
        with c1:
            sr=recs.head(10).copy(); sr['Rating']=sr['Rating'].round(2); sr['Confidence']=sr['Confidence'].round(3); sr.index=range(1,len(sr)+1)
            st.dataframe(sr,height=400,use_container_width=True); st.download_button("📥 Download Recommendations",sr.to_csv().encode(),'recommendations.csv','text/csv')
        with c2:
            t6=recs.head(6); f=px.bar(t6,x='Confidence',y='Attraction',orientation='h',color='Rating',color_continuous_scale='RdYlGn',text='Rating'); f.update_traces(texttemplate='%{text:.2f}⭐',textposition='outside'); f.update_layout(height=400,yaxis={'categoryorder':'total ascending'},showlegend=False); st.plotly_chart(f,use_container_width=True)
        uh=df[df['UserId']==uid][['Attraction','Rating','VisitYear','VisitMonth']].sort_values(['VisitYear','VisitMonth'],ascending=False).head(10)
        if not uh.empty:
            c1,c2=st.columns([2,1]);
            with c1: st.dataframe(uh,hide_index=True,use_container_width=True)
            with c2: f=px.line(uh.iloc[::-1],y='Rating',title="Rating Trend",markers=True); f.update_layout(height=250,showlegend=False); st.plotly_chart(f,use_container_width=True)

with tab4:
    st.markdown("## 📈 DEEP ANALYTICS & INSIGHTS")
    c1,c2=st.columns(2)
    with c1:
        ta=dff['Attraction'].value_counts().head(15).reset_index(); ta.columns=['Attraction','Visits']; f=px.bar(ta,x='Visits',y='Attraction',orientation='h',color='Visits',color_continuous_scale='Blues',text='Visits',title="🏆 Top 15 Attractions"); f.update_traces(textposition='outside'); f.update_layout(height=500,yaxis={'categoryorder':'total ascending'},showlegend=False); st.plotly_chart(f,use_container_width=True)
    with c2:
        if 'User_CityName' in dff.columns:
            tc=dff['User_CityName'].value_counts().head(15).reset_index(); tc.columns=['City','Users']; f=px.bar(tc,x='Users',y='City',orientation='h',color='Users',color_continuous_scale='Reds',text='Users',title="🏙️ Top 15 User Cities"); f.update_traces(textposition='outside'); f.update_layout(height=500,yaxis={'categoryorder':'total ascending'},showlegend=False); st.plotly_chart(f,use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        md=dff['VisitMode'].value_counts().reset_index(); md.columns=['Mode','Count']; f=px.pie(md,values='Count',names='Mode',hole=0.5,title="🚗 Visit Mode Distribution",color_discrete_sequence=px.colors.qualitative.Set3); f.update_traces(textposition='inside',textinfo='percent+label'); f.update_layout(height=400); st.plotly_chart(f,use_container_width=True)
    with c2:
        tp=dff.groupby('AttractionTypeId').agg(Avg_Rating=('Rating','mean'),Count=('UserId','count')).reset_index(); tp.columns=['Type','Avg_Rating','Count']; f=px.scatter(tp,x='Count',y='Avg_Rating',size='Count',color='Avg_Rating',color_continuous_scale='RdYlGn',hover_data=['Type'],text='Type',title="🎭 Attraction Type Performance"); f.update_traces(textposition='top center'); f.update_layout(height=400); st.plotly_chart(f,use_container_width=True)

with tab5:
    st.markdown("## 🔍 TREND ANALYSIS")
    ys=df.groupby('VisitYear').agg(Visits=('UserId','count'),Avg_Rating=('Rating','mean')).reset_index(); ys['Growth_%']=ys['Visits'].pct_change()*100
    fig=make_subplots(rows=1,cols=2,subplot_titles=('Visit Volume','Average Rating')); fig.add_trace(go.Bar(x=ys['VisitYear'],y=ys['Visits'],name='Visits',marker_color='#667eea'),row=1,col=1); fig.add_trace(go.Scatter(x=ys['VisitYear'],y=ys['Avg_Rating'],mode='lines+markers',name='Avg Rating',line=dict(color='#f5576c',width=3)),row=1,col=2); fig.update_layout(height=400); st.plotly_chart(fig,use_container_width=True)
    c1,c2=st.columns([2,1])
    with c1: st.dataframe(ys.style.format({'Visits':'{:,.0f}','Avg_Rating':'{:.2f}','Growth_%':'{:.2f}%'}),hide_index=True,use_container_width=True)
    with c2: tg=((ys['Visits'].iloc[-1]/ys['Visits'].iloc[0])-1)*100; st.metric("Total Growth",f"{tg:.1f}%",delta=f"{ys['Growth_%'].iloc[-1]:.1f}% YoY")
    mp=df.groupby(['VisitYear','VisitMonth']).size().reset_index(name='Visits'); f=px.line(mp,x='VisitMonth',y='Visits',color='VisitYear',markers=True,title="Monthly Visit Patterns by Year"); f.update_layout(height=400); st.plotly_chart(f,use_container_width=True)
    hm=df.pivot_table(index='VisitMonth',columns='VisitYear',values='UserId',aggfunc='count').fillna(0); f=px.imshow(hm,labels=dict(x="Year",y="Month",color="Visits"),aspect="auto",color_continuous_scale='YlOrRd'); f.update_layout(height=400,title="Visit Heatmap: Month vs Year"); st.plotly_chart(f,use_container_width=True)

with tab6:
    st.markdown("## 📋 DATA EXPLORER")
    c1,c2,c3=st.columns(3)
    fc=c1.multiselect("Filter by Continent",sorted(df['ContinentId'].unique()),default=[])
    fm=c2.multiselect("Filter by Visit Mode",sorted(df['VisitMode'].unique()),default=[])
    rr=c3.slider("Filter by Rating",float(df['Rating'].min()),float(df['Rating'].max()),(float(df['Rating'].min()),float(df['Rating'].max())))
    de=dff.copy()
    if fc: de=de[de['ContinentId'].isin(fc)]
    if fm: de=de[de['VisitMode'].isin(fm)]
    de=de[(de['Rating']>=rr[0])&(de['Rating']<=rr[1])]
    c1,c2,c3,c4=st.columns(4); c1.metric("Filtered Records",f"{len(de):,}"); c2.metric("Avg Rating",f"{de['Rating'].mean():.2f}"); c3.metric("Unique Attractions",f"{de['Attraction'].nunique()}"); c4.metric("Unique Users",f"{de['UserId'].nunique()}")
    dc=['Attraction','Rating','VisitYear','VisitMonth','VisitMode','ContinentId','CountryId']
    st.dataframe(de[dc].head(100),hide_index=True,use_container_width=True,height=400)
    c1,c2=st.columns(2)
    with c1: st.download_button("📥 Download Filtered Data (CSV)",de[dc].to_csv(index=False).encode(),'tourism_filtered_data.csv','text/csv')
    with c2:
        if st.button("📊 Generate Summary Report"):
            st.download_button("📥 Download Summary Statistics",de.describe().to_csv().encode(),'summary_statistics.csv','text/csv')

st.markdown('---\n<div style="text-align:center;padding:25px;background:linear-gradient(135deg,#667eea,#764ba2);border-radius:15px"><h3 style="color:white;margin:0">🌍 Tourism Intelligence Hub</h3><p style="color:white;opacity:.9;margin:10px 0 0 0">© 2024 | Powered by Advanced Machine Learning & AI</p></div>',unsafe_allow_html=True)
