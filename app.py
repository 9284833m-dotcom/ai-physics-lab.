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
    page_title="مختبر الفيزياء التطورية | AI Physics Lab",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 لوحة تحكم مكتشف الفيزياء الاصطناعي (AI Science Discoverer)")
st.caption("نظام تطوري ذكي لاكتشاف القوانين الفيزيائية ومحاكاتها افتراضياً مع إمكانية تصدير البيانات")

# =====================================================================
# 2. إعداد الرموز والتوليد الرمزي العشوائي
# =====================================================================
t = sp.Symbol('t', real=True, positive=True)
x = sp.Symbol('x', real=True)
v = sp.Symbol('v', real=True)

def generate_truly_novel_expression(max_depth=3):
    """توليد معادلة تفاضلية جديدة كلياً من الصفر باستخدام شجرة العمليات الرمزية"""
    term_pool = [x, v, t, sp.sin(x), sp.cos(v), sp.exp(-0.1*t), x**2, v**2, sp.sin(t)]
    
    def build_tree(depth):
        if depth == 0 or random.random() < 0.3:
            return random.choice(term_pool)
        
        op = random.choice(['add', 'sub', 'mul', 'sin', 'nonlin'])
        if op == 'add':
            return build_tree(depth - 1) + build_tree(depth - 1)
        elif op == 'sub':
            return build_tree(depth - 1) - build_tree(depth - 1)
        elif op == 'mul':
            return build_tree(depth - 1) * build_tree(depth - 1)
        elif op == 'sin':
            return sp.sin(build_tree(depth - 1))
        else:
            return build_tree(depth - 1) * sp.cos(x)
            
    try:
        expr = build_tree(max_depth)
        # ضمان وجود متغيرات الحركة لمنع معادلات ثابتة
        if not (expr.has(x) or expr.has(v)):
            expr = expr - 0.5 * v - 1.5 * sp.sin(x)
        return sp.simplify(expr)
    except Exception:
        return -1.0 * sp.sin(x) - 0.2 * v

def mutate_expression(expr):
    """إجراء طفرة جينية رمزية على المعادلة لتوليد صيغ جديدة"""
    mutations = [
        lambda e: e + random.choice([0.5*sp.cos(x), -0.2*v, 0.1*sp.sin(2*t)]),
        lambda e: e * random.choice([0.8, 1.2, sp.cos(x)]),
        lambda e: e.subs(v, v**2 if not e.has(v**2) else v*sp.sin(x)),
        lambda e: e.subs(x, sp.sin(x) if not e.has(sp.sin(x)) else x**2)
    ]
    try:
        mutated = random.choice(mutations)(expr)
        return sp.simplify(mutated)
    except Exception:
        return expr

def crossover_expressions(parent1, parent2):
    """تزاوج جيني بين معادلتين ناجحتين"""
    try:
        return sp.simplify((parent1 / 2) + (parent2 / 2))
    except Exception:
        return parent1

def simulate_and_evaluate(expr, t_span=(0, 20)):
    """محاكاة الحركة وتقييم درجة لياقة المعادلة (Fitness Score)"""
    try:
        acc_lambda = sp.lambdify((t, x, v), expr, 'numpy')

        def system_odes(t_curr, state):
            x_curr, v_curr = state
            try:
                res = acc_lambda(t_curr, x_curr, v_curr)
                # معالجة القيم المركبة
                if isinstance(res, complex):
                    res = res.real
                a_curr = float(res)
                a_curr = np.clip(a_curr, -1e3, 1e3)
            except Exception:
                a_curr = 0.0
            return [v_curr, a_curr]

        t_eval = np.linspace(t_span[0], t_span[1], 800)
        sol = solve_ivp(system_odes, t_span, [1.0, 0.0], t_eval=t_eval, method='RK45')
        
        x_pts, v_pts = sol.y[0], sol.y[1]
        if np.isnan(x_pts).any() or np.isinf(x_pts).any():
            return -1.0, None
            
        std_x = np.std(x_pts)
        max_x = np.max(np.abs(x_pts))
        
        if std_x < 0.01 or max_x > 500:
            return 0.1, None

        fitness = float(std_x * 2.0 + len(str(expr)) * 0.01)
        return fitness, (sol.t, x_pts, v_pts)
    except Exception:
        return -1.0, None

# =====================================================================
# 3. الشريط الجانبي (Sidebar) لمعاملات التحكم
# =====================================================================
st.sidebar.header("⚙️ إعدادات التجربة التطورية")
generations = st.sidebar.slider("عدد الأجيال (Generations):", min_value=2, max_value=15, value=5)
pop_size = st.sidebar.slider("حجم المجتمع (Population Size):", min_value=4, max_value=20, value=8)

start_button = st.sidebar.button("🚀 بدء اكتشاف نظريات جديدة", use_container_width=True)

# =====================================================================
# 4. دورة المحاكاة والتطور وتخزين النتائج
# =====================================================================
if "discoveries" not in st.session_state:
    st.session_state.discoveries = []

if start_button:
    st.session_state.discoveries = []
    # توليد مجتمع كلياً من معادلات جديدة غير مسبوقة
    population = [generate_truly_novel_expression() for _ in range(pop_size)]

    progress_bar = st.progress(0)
    status_text = st.empty()

    for gen in range(1, generations + 1):
        status_text.text(f"🧬 جاري تقييم وتطوير الجيل {gen} من {generations}...")
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
            if random.random() < 0.7:
                next_pop.append(mutate_expression(random.choice(survivors)))
            else:
                p1, p2 = random.sample(survivors, min(2, len(survivors))) if len(survivors) >= 2 else (survivors[0], survivors[0])
                next_pop.append(crossover_expressions(p1, p2))
                
        population = next_pop
        progress_bar.progress(gen / generations)

    status_text.success("✅ اكتملت دورتك التطورية واكتُشفت نظريات جديدة!")

# =====================================================================
# 5. عرض النتائج والرسوم البيانية التفاعلية
# =====================================================================
if st.session_state.discoveries:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📜 قائمة النظريات والمعادلات المكتشفة")
        options = [f"الجيل {d['gen']} | الدرجة: {d['score']:.2f} | a = {d['expr_str']}" for d in st.session_state.discoveries]
        selected_option = st.selectbox("اختر معادلة لدراسة سلوكها الفيزيائي:", options)
        
        selected_idx = options.index(selected_option)
        selected_data = st.session_state.discoveries[selected_idx]

        st.markdown("**الصيغة الرياضية المنمقة (LaTeX):**")
        st.latex(f"a = {sp.latex(selected_data['expr'])}")

    with col2:
        st.subheader("📊 المحاكاة الفيزيائية التفاعلية")
        if selected_data["sim_data"]:
            t_pts, x_pts, v_pts = selected_data["sim_data"]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

            ax1.plot(t_pts, x_pts, label="الموضع x(t)", color="#1f77b4", linewidth=1.5)
            ax1.plot(t_pts, v_pts, label="السرعة v(t)", color="#ff7f0e", linestyle="--", linewidth=1.2)
            ax1.set_title("تطور الحركة عبر الزمن")
            ax1.set_xlabel("الزمن t")
            ax1.set_ylabel("القيم الفيزيائية")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            ax2.plot(x_pts, v_pts, color="#2ca02c", linewidth=1.0)
            ax2.set_title("مخطط الطور (Phase Space Portrait: v vs x)")
            ax2.set_xlabel("الموضع x")
            ax2.set_ylabel("السرعة v")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

    # =====================================================================
    # 6. قسم تصدير وحفظ البيانات (JSON و CSV)
    # =====================================================================
    st.markdown("---")
    st.subheader("💾 تصدير وحفظ النظريات المكتشفة")
    
    export_data_list = []
    for item in st.session_state.discoveries:
        export_data_list.append({
            "generation": item["gen"],
            "fitness_score": item["score"],
            "equation_text": item["expr_str"],
            "equation_latex": str(sp.latex(item["expr"]))
        })
    
    json_string = json.dumps(export_data_list, ensure_ascii=False, indent=4)
    df_export = pd.DataFrame(export_data_list)
    csv_data = df_export.to_csv(index=False, encoding='utf-8-sig')

    col_json, col_csv = st.columns(2)

    with col_json:
        st.download_button(
            label="📥 تنزيل الكتالوج بصيغة JSON",
            data=json_string,
            file_name="physics_discoveries.json",
            mime="application/json",
            use_container_width=True
        )

    with col_csv:
        st.download_button(
            label="📊 تنزيل جدول المعادلات بصيغة CSV (Excel)",
            data=csv_data,
            file_name="physics_discoveries.csv",
            mime="text/csv",
            use_container_width=True
        )
else:
    st.info("👈 اضغط على زر **'بدء اكتشاف نظريات جديدة'** في الشريط الجانبي لتوليد واختبار معادلات فريدة.")
    
