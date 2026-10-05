import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sympy as sp
from scipy.integrate import solve_ivp
import streamlit as st

# =====================================================================
# 1. إعدادات الواجهة المتقدمة
# =====================================================================
st.set_page_config(
    page_title="AI Physics Lab & Discoverer 3D",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #c9d1d9; }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .analyzer-box {
        background-color: #1c2128;
        border-left: 4px solid #58a6ff;
        padding: 15px;
        border-radius: 4px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚛️ مختبر الفيزياء الاصطناعي المتقدم | AI Theoretical Physics Discovery Engine")
st.caption("منصة استكشاف وتحليل النظريات الديناميكية غير الخطية، تقييم الاتزان، والمحاكاة ثلاثية الأبعاد")

# =====================================================================
# 2. المحرك الرمزي وتوليد المعادلات
# =====================================================================
t = sp.Symbol('t', real=True, positive=True)
x = sp.Symbol('x', real=True)
v = sp.Symbol('v', real=True)

def generate_truly_novel_expression(max_depth=3):
    term_pool = [x, v, t, sp.sin(x), sp.cos(v), sp.exp(-0.05*t), x**2, v**2, sp.sin(t), sp.cos(x*v)]
    
    def build_tree(depth):
        if depth == 0 or random.random() < 0.25:
            return random.choice(term_pool)
        
        op = random.choice(['add', 'sub', 'mul', 'sin', 'cos', 'nonlin'])
        if op == 'add':
            return build_tree(depth - 1) + build_tree(depth - 1)
        elif op == 'sub':
            return build_tree(depth - 1) - build_tree(depth - 1)
        elif op == 'mul':
            return build_tree(depth - 1) * build_tree(depth - 1)
        elif op == 'sin':
            return sp.sin(build_tree(depth - 1))
        elif op == 'cos':
            return sp.cos(build_tree(depth - 1))
        else:
            return build_tree(depth - 1) * sp.cos(x)
            
    try:
        expr = build_tree(max_depth)
        if not (expr.has(x) or expr.has(v)):
            expr = expr - 0.4 * v - 1.2 * sp.sin(x)
        return sp.simplify(expr)
    except Exception:
        return -1.0 * sp.sin(x) - 0.2 * v

def simulate_and_evaluate(expr, t_span=(0, 30)):
    try:
        acc_lambda = sp.lambdify((t, x, v), expr, 'numpy')

        def system_odes(t_curr, state):
            x_curr, v_curr = state
            try:
                res = acc_lambda(t_curr, x_curr, v_curr)
                if isinstance(res, complex):
                    res = res.real
                a_curr = float(res)
                a_curr = np.clip(a_curr, -500, 500)
            except Exception:
                a_curr = 0.0
            return [v_curr, a_curr]

        t_eval = np.linspace(t_span[0], t_span[1], 1200)
        sol = solve_ivp(system_odes, t_span, [1.0, 0.0], t_eval=t_eval, method='RK45')
        
        x_pts, v_pts = sol.y[0], sol.y[1]
        if np.isnan(x_pts).any() or np.isinf(x_pts).any():
            return -1.0, None
            
        std_x = np.std(x_pts)
        max_x = np.max(np.abs(x_pts))
        
        if std_x < 0.001 or max_x > 1000:
            return 0.1, None

        fitness = float(std_x * 2.5 + len(str(expr)) * 0.005)
        return fitness, (sol.t, x_pts, v_pts)
    except Exception:
        return -1.0, None

# =====================================================================
# 3. محرك الذكاء الاصطناعي التحليلي (AI Physics Analyzer)
# =====================================================================
def analyze_theory_physics(expr, t_pts, x_pts, v_pts):
    """تحليل ذكي معمق للمعادلة والسلوك الفيزيائي ومخطط الطور"""
    analysis = {}
    
    # 1. تحليل مكونات الرياضية
    has_damping = expr.has(v)
    has_trig = expr.has(sp.sin) or expr.has(sp.cos)
    has_time_drive = expr.has(t)
    has_nonlinearity = any(expr.has(term) for term in [x**2, v**2, x*v])
    
    # 2. حساب مؤشرات الاستقرار والطاقة
    kinetic_energy = 0.5 * (v_pts ** 2)
    energy_trend = np.polyfit(t_pts, kinetic_energy, 1)[0]
    std_v = np.std(v_pts)
    
    # 3. تصنيف النظام
    if has_time_drive:
        system_type = "نظام ديناميكي محفّز زمنياً (Non-autonomous Driven System)"
    elif has_damping and energy_trend < -0.01:
        system_type = "مهتز خامد للطاقة (Dissipative Dynamical System)"
    elif has_nonlinearity:
        system_type = "نظام اهتزازي غير خطي معقد (Nonlinear Coupled Oscillator)"
    else:
        system_type = "مهتز دوري محفاظ نسبياً (Quasi-Conservative Oscillator)"

    # 4. تقييم الصحة الفيزيائية (Physical Validity Criteria)
    validity_score = 100
    flags = []
    
    if np.max(np.abs(x_pts)) > 100:
        validity_score -= 30
        flags.append("انفجار الموضع (Position Divergence) قد يشير لعدم واقعية الفيزيائية بدون حوادث إرجاع.")
    if energy_trend > 0.5:
        validity_score -= 25
        flags.append("توليد طاقة غير محدود (Infinite Energy Gain) يخالف قانون حفظ الطاقة.")
    if std_v < 0.05:
        validity_score -= 40
        flags.append("النظام يؤول لسكون تام سريعا (Trivial Equilibrium).")
        
    if validity_score >= 80:
        validity_label = "عالية جداً (نظرية فيزيائية متماسكة)"
    elif validity_score >= 50:
        validity_label = "متوسطة (تتطلب شروط حدودية إضافية)"
    else:
        validity_label = "منخفضة (نموذج رياضي نظري غير مستقر)"

    # 5. التفسير الفيزيائي المكتوب
    desc = f"تُمثل هذه المعادلة **{system_type}**."
    if has_trig:
        desc += " تعتمد قوة الإرجاع فيها على دوال دائرية مما يسبب استجابة اهتزازية متعددة الاستقرار (Multistable)."
    if has_damping:
        desc += " تحتوي المعادلة على حدود مرتبطة بالسرعة $v$ تؤدي لتبديد أو ضخ الطاقة في فضاء الطور."
    if has_time_drive:
        desc += " وجود المتغير الزمني $t$ يجعل النظام يمتلك اضطرابات خارجية قد تسبب سلوكاً شبه فوضوي (Chaotic Behavior)."

    analysis['type'] = system_type
    analysis['validity_score'] = validity_score
    analysis['validity_label'] = validity_label
    analysis['flags'] = flags
    analysis['description'] = desc
    analysis['energy_trend'] = energy_trend
    
    return analysis

# =====================================================================
# 4. الشريط الجانبي والتحكم
# =====================================================================
st.sidebar.header("⚙️️ إعدادات المكتشف والتطور")
generations = st.sidebar.slider("عدد الأجيال التطورية:", 2, 20, 6)
pop_size = st.sidebar.slider("حجم مجتمع النظريات:", 4, 25, 10)
simulation_time = st.sidebar.slider("زمن المحاكاة (t_max):", 10, 100, 30)

start_button = st.sidebar.button("🚀 توليد وتحليل نظريات جديدة", use_container_width=True)

if "discoveries" not in st.session_state:
    st.session_state.discoveries = []

if start_button:
    st.session_state.discoveries = []
    population = [generate_truly_novel_expression() for _ in range(pop_size)]

    progress_bar = st.progress(0)
    status_text = st.empty()

    for gen in range(1, generations + 1):
        status_text.text(f"🧬 جاري توليد وتحليل الجيل {gen}...")
        scores = []
        for ind in population:
            score, sim_data = simulate_and_evaluate(ind, t_span=(0, simulation_time))
            scores.append((score, ind, sim_data))

        scores.sort(key=lambda item: item[0], reverse=True)
        best_score, best_expr, sim_data = scores[0]

        if best_score > 0 and sim_data is not None:
            st.session_state.discoveries.append({
                "gen": gen,
                "score": best_score,
                "expr": best_expr,
                "expr_str": str(best_expr),
                "sim_data": sim_data
            })

        progress_bar.progress(gen / generations)
        population = [generate_truly_novel_expression() for _ in range(pop_size)]

    status_text.success("✅ تم توليد النظريات وتحليلها بنجاح!")

# =====================================================================
# 5. عرض النتائج والواجهة العلمية
# =====================================================================
if st.session_state.discoveries:
    options = [f"الجيل {d['gen']} | L-Score: {d['score']:.2f} | a = {d['expr_str']}" for d in st.session_state.discoveries]
    selected_option = st.selectbox("🔬 اختر النظرية المكتشفة للدراسة والتحليل العميق:", options)
    
    selected_idx = options.index(selected_option)
    selected_data = st.session_state.discoveries[selected_idx]
    
    t_pts, x_pts, v_pts = selected_data["sim_data"]
    expr = selected_data["expr"]

    # إجراء التحليل الفيزيائي بالذكاء الاصطناعي
    ai_analysis = analyze_theory_physics(expr, t_pts, x_pts, v_pts)

    st.markdown("---")
    
    # عرض المعادلة ولوحات القياس
    col_eq, col_m1, col_m2, col_m3 = st.columns([2, 1, 1, 1])
    with col_eq:
        st.markdown("**الصياغة الرياضية للنظرية:**")
        st.latex(f"a = \\frac{{d^2 x}}{{dt^2}} = {sp.latex(expr)}")
    with col_m1:
        st.metric("درجة الصحة الفيزيائية", f"{ai_analysis['validity_score']}%")
    with col_m2:
        st.metric("أقصى إزاحة (Max X)", f"{np.max(np.abs(x_pts)):.2f}")
    with col_m3:
        st.metric("معدل تغير الطاقة", f"{ai_analysis['energy_trend']:.4f}")

    # صندوق التحليل الذكي
    st.markdown(f"""
    <div class="analyzer-box">
        <h4>🤖 التحليل الفيزيائي المستنتج بالذكاء الاصطناعي:</h4>
        <p><b>تصنيف النظام:</b> {ai_analysis['type']}</p>
        <p><b>تقييم الموثوقية:</b> {ai_analysis['validity_label']}</p>
        <p>{ai_analysis['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if ai_analysis['flags']:
        with st.expander("⚠️ ملاحظات الاستقرار الفيزيائي"):
            for flag in ai_analysis['flags']:
                st.write(f"- {flag}")

    st.markdown("### 📊 الرسوم والتحليلات البيانية المتقدمة")
    tab1, tab2, tab3 = st.tabs(["📈 التحليل الزمني وفضاء الطور", "🧊 المحاكاة ثلاثية الأبعاد (3D View)", "⚡ تحليل الطاقة والحركة"])

    with tab1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        
        ax1.plot(t_pts, x_pts, label="الموضع x(t)", color="#58a6ff", linewidth=1.5)
        ax1.plot(t_pts, v_pts, label="السرعة v(t)", color="#ffa657", linestyle="--", linewidth=1.2)
        ax1.set_title("تطور المتغيرات عبر الزمن")
        ax1.set_xlabel("الزمن t")
        ax1.grid(True, alpha=0.2)
        ax1.legend()

        ax2.plot(x_pts, v_pts, color="#3fb950", linewidth=1.0)
        ax2.set_title("مسار فضاء الطور (Phase Space Portrait: v vs x)")
        ax2.set_xlabel("الموضع x")
        ax2.set_ylabel("السرعة v")
        ax2.grid(True, alpha=0.2)
        
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.subheader("🧊 المحاكاة التفاعلية ثلاثية الأبعاد (3D Phase-Time Portrait)")
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=x_pts,
            y=v_pts,
            z=t_pts,
            mode='lines',
            line=dict(
                color=t_pts,
                colorscale='Viridis',
                width=4
            )
        )])
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='الموضع (X)',
                yaxis_title='السرعة (V)',
                zaxis_title='الزمن (T)'
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=500
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab3:
        kinetic_energy = 0.5 * (v_pts ** 2)
        fig_e, ax_e = plt.subplots(figsize=(10, 3))
        ax_e.plot(t_pts, kinetic_energy, color="#d2a8ff", linewidth=1.5)
        ax_e.set_title("مخطط الطاقة الحركية التقديرية عبر الزمن E_k(t)")
        ax_e.set_xlabel("الزمن t")
        ax_e.set_ylabel("الطاقة الحركية")
        ax_e.grid(True, alpha=0.2)
        st.pyplot(fig_e)

    # تصدير البيانات
    st.markdown("---")
    st.subheader("💾 تصدير التقرير العلمي")
    df_export = pd.DataFrame([{
        "Generation": selected_data["gen"],
        "Equation": selected_data["expr_str"],
        "Validity_Score": ai_analysis['validity_score'],
        "System_Type": ai_analysis['type']
    }])
    st.download_button("📊 تنزيل تقرير النظرية (CSV)", df_export.to_csv(index=False), "theory_report.csv", "text/csv")

else:
    st.info("👈 اضغط على زر **'توليد وتحليل نظريات جديدة'** لبدء تشغيل محرك الاستكشاف الفيزيائي المتقدم.")
    
