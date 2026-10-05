```python
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import solve_ivp
import streamlit as st

# =====================================================================
# 1. إعدادات الصفحة والهيكل العام للواجهة
# =====================================================================
st.set_page_config(
    page_title="مختبر اكتشاف النظريات الفيزيائية | AI Physics Lab",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 مولد ومكتشف النظريات الفيزيائية المستقل")
st.caption("نظام مبتكر لتوليد معادلات تفاضلية غير سابقة واختبار استقرارها وتطورها في فضاء الطور")

# =====================================================================
# 2. إعداد الرموز والخوارزميات الجينية وتوليد المعادلات غير المسبوقة
# =====================================================================
t = sp.Symbol('t', real=True, positive=True)
x = sp.Symbol('x', real=True)
v = sp.Symbol('v', real=True)

def generate_truly_novel_expression(depth=0, max_depth=2):
    """توليد دالة رياضية مبتكرة كلياً عبر تركيب أشجار التعابير الرمزية العشوائية"""
    terminal_nodes = [x, v, sp.sin(x), sp.cos(v), sp.exp(-0.1 * x**2), x*v, sp.tanh(v)]
    binary_ops = [
        lambda p, q: p + q,
        lambda p, q: p - q,
        lambda p, q: p * q,
        lambda p, q: p * sp.sin(q),
        lambda p, q: p / (1 + q**2)
    ]
    
    if depth >= max_depth or (depth > 0 and random.random() < 0.4):
        coeff = round(random.uniform(-3.0, 3.0), 2)
        return coeff * random.choice(terminal_nodes)
    
    op = random.choice(binary_ops)
    left = generate_truly_novel_expression(depth + 1, max_depth)
    right = generate_truly_novel_expression(depth + 1, max_depth)
    return op(left, right)

def mutate_expression(expr):
    """تطبيق طفرة جينية عشوائية على بنية المعادلة"""
    mutations = [
        lambda e: e + round(random.uniform(-1.5, 1.5), 2) * sp.sin(x * v),
        lambda e: e * (1 + 0.2 * sp.cos(t)),
        lambda e: e.subs(x, x**2 / (1 + sp.Abs(x))),
        lambda e: e + round(random.uniform(-2, 2), 2) * sp.tanh(v)
    ]
    try:
        mutated = random.choice(mutations)(expr)
        return sp.simplify(mutated)
    except Exception:
        return expr

def simulate_and_evaluate(expr, t_span=(0, 25)):
    """محاكاة النظرية المبتكرة واستبعاد الاهتزازات التافهة أو المتباعدة"""
    try:
        acc_lambda = sp.lambdify((t, x, v), expr, modules=['numpy', 'sympy'])

        def system_odes(t_curr, state):
            x_curr, v_curr = state
            try:
                res = acc_lambda(t_curr, x_curr, v_curr)
                a_curr = float(sp.re(res)) if hasattr(res, 'evalf') else float(np.real(res))
                a_curr = np.clip(a_curr, -500.0, 500.0)
            except Exception:
                a_curr = 0.0
            return [v_curr, a_curr]

        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        sol = solve_ivp(system_odes, t_span, [1.5, 0.1], t_eval=t_eval, method='RK45')
        
        x_pts, v_pts = sol.y[0], sol.y[1]
        if np.isnan(x_pts).any() or np.isinf(x_pts).any():
            return -1.0, None
            
        std_x = np.std(x_pts)
        std_v = np.std(v_pts)
        max_x = np.max(np.abs(x_pts))
        
        # استبعاد الحالات الثابتة تماماً أو التي تتمدد بشكل انفجاري
        if std_x < 0.05 or max_x > 300:
            return 0.0, None

        # تقييم التعقيد الديناميكي وفضاء الطور
        fitness = float(std_x * 1.5 + std_v * 1.2 + (len(str(expr)) % 15) * 0.05)
        return fitness, (sol.t, x_pts, v_pts)
    except Exception:
        return -1.0, None

# =====================================================================
# 3. واجهة التحكم والتفاعل
# =====================================================================
st.sidebar.header("⚙️ محرك الابتكار الرياضي")
generations = st.sidebar.slider("عدد أجيال التطور:", min_value=3, max_value=20, value=6)
pop_size = st.sidebar.slider("عدد النظريات في الجيل:", min_value=4, max_value=25, value=10)
complexity = st.sidebar.slider("مستوى تعقيد المعادلات:", min_value=1, max_value=4, value=2)

start_button = st.sidebar.button("✨ توليد نظريات غير مسبوقة", use_container_width=True)

if "discoveries" not in st.session_state:
    st.session_state.discoveries = []

if start_button:
    st.session_state.discoveries = []
    
    # بناء مجتمع ابتدائي يعتمد كلياً على أشجار رموز عشوائية
    population = [generate_truly_novel_expression(max_depth=complexity) for _ in range(pop_size)]

    progress_bar = st.progress(0)
    status_text = st.empty()

    for gen in range(1, generations + 1):
        status_text.text(f"🧬 جاري تخليق ومحاكاة الجيل {gen} من النظريات...")
        scores = []
        for ind in population:
            score, sim_data = simulate_and_evaluate(ind)
            scores.append((score, ind, sim_data))

        scores.sort(key=lambda item: item[0], reverse=True)
        best_score, best_expr, sim_data = scores[0]

        if best_score > 0:
            st.session_state.discoveries.append({
                "gen": gen,
                "score": best_score,
                "expr": best_expr,
                "expr_str": str(best_expr),
                "sim_data": sim_data
            })

        survivors = [item[1] for item in scores[:max(2, pop_size // 2)]]
        next_pop = list(survivors)
        while len(next_pop) < pop_size:
            next_pop.append(mutate_expression(random.choice(survivors)))
                
        population = next_pop
        progress_bar.progress(gen / generations)

    status_text.success("✅ تم اكتشاف وتوليد نظريات ديناميكية جديدة!")

# =====================================================================
# 4. عرض الرسومات والتصدير
# =====================================================================
if st.session_state.discoveries:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📜 النظريات المبتكرة المكتشفة")
        options = [f"الجيل {d['gen']} | درجة الابتكار: {d['score']:.2f} | a = {d['expr_str']}" for d in st.session_state.discoveries]
        selected_option = st.selectbox("اختر نظرية لعرض تفاصيلها:", options)
        
        selected_idx = options.index(selected_option)
        selected_data = st.session_state.discoveries[selected_idx]

        st.markdown("**الصياغة الرياضية الرمزية (LaTeX):**")
        st.latex(f"a(t, x, v) = {sp.latex(selected_data['expr'])}")

    with col2:
        st.subheader("📊 الاستجابة ومخطط الطور")
        if selected_data["sim_data"]:
            t_pts, x_pts, v_pts = selected_data["sim_data"]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5))

            ax1.plot(t_pts, x_pts, label="الموضع x(t)", color="#0284c7", linewidth=1.5)
            ax1.plot(t_pts, v_pts, label="السرعة v(t)", color="#ea580c", linestyle="--", linewidth=1.2)
            ax1.set_title("السلوك الزمني للنظرية المخلقة")
            ax1.set_grid(True, alpha=0.3)
            ax1.legend()

            ax2.plot(x_pts, v_pts, color="#16a34a", linewidth=1.0)
            ax2.set_title("مسار فضاء الطور الفريد (Phase Portrait)")
            ax2.set_xlabel("الموضع x")
            ax2.set_ylabel("السرعة v")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

    st.markdown("---")
    st.subheader("💾 تصدير الكتالوج والبيانات")
    
    export_list = [{
        "generation": item["gen"],
        "score": item["score"],
        "equation_text": item["expr_str"],
        "equation_latex": str(sp.latex(item["expr"]))
    } for item in st.session_state.discoveries]
    
    json_string = json.dumps(export_list, ensure_ascii=False, indent=4)
    csv_data = pd.DataFrame(export_list).to_csv(index=False, encoding='utf-8-sig')

    c_json, c_csv = st.columns(2)
    c_json.download_button("📥 تنزيل الكتالوج (JSON)", json_string, "new_physics.json", "application/json")
    c_csv.download_button("📊 تنزيل المعادلات (CSV)", csv_data, "new_physics.csv", "text/csv")
```
