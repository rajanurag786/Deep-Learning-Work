import streamlit as st
from Prompt_Parser import parse_prompt
from Ansys_Geometry_Creator_with_Prompt import run_static_analysis
from Report_Generator import generate_pdf_report
import plotly.graph_objects as go
import numpy as np
import os

st.set_page_config(page_title="FEA AI Agent", layout="wide")

st.title("🛠️ AI‐Driven FEA Agent")

prompt = st.text_area(
    "Enter your analysis prompt:",
    height=150,
    placeholder="e.g. Analyze a cylinder r=0.3m, h=1.2m, concrete, under 10 kN axial load."
)

def plot_geometry_preview(geom: dict) -> go.Figure:
    """
    Given geom = {'type': str, 'parameters': {...}},
    return a Plotly Mesh3d figure for interactive rotation.
    """
    typ = geom["type"]
    params = geom["parameters"]

    if typ == "cylinder":
        r = float(params["radius"])
        h = float(params["height"])
        # discretize circle
        n = 50
        theta = np.linspace(0, 2 * np.pi, n)
        x_bot = r * np.cos(theta)
        y_bot = r * np.sin(theta)
        z_bot = np.zeros(n)
        x_top = x_bot.copy()
        y_top = y_bot.copy()
        z_top = np.full(n, h)

        # vertices
        x = np.concatenate([x_bot, x_top])
        y = np.concatenate([y_bot, y_top])
        z = np.concatenate([z_bot, z_top])

        # build faces
        i, j, k = [], [], []
        for k_idx in range(n - 1):
            # side quad → two triangles
            i += [k_idx, k_idx]
            j += [k_idx + 1, k_idx + n + 1]
            k += [k_idx + n, k_idx + n + 1]
        # close the loop
        i += [n - 1, n - 1]
        j += [0,     n]
        k += [2*n - 1, 2*n - 1]

    elif typ == "box":
        w = float(params["width"])
        d = float(params["depth"])
        h = float(params["height"])
        # 8 corners
        verts = np.array([
            [0, 0, 0],
            [w, 0, 0],
            [w, d, 0],
            [0, d, 0],
            [0, 0, h],
            [w, 0, h],
            [w, d, h],
            [0, d, h],
        ])
        x, y, z = verts.T
        # 12 triangles
        faces = [
            [0,1,2],[0,2,3],  # bottom
            [4,5,6],[4,6,7],  # top
            [0,1,5],[0,5,4],  # front
            [1,2,6],[1,6,5],  # right
            [2,3,7],[2,7,6],  # back
            [3,0,4],[3,4,7],  # left
        ]
        i, j, k = zip(*faces)
    else:
        raise ValueError(f"Cannot preview shape '{typ}'")

    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        opacity=0.5,
        colorscale="Viridis",
        intensity=z  # colored by height
    )
    fig = go.Figure(mesh)
    fig.update_layout(
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig


if st.button("Run Analysis"):
    if not prompt.strip():
        st.error("Please enter a prompt first.")
    else:
        try:
            with st.spinner("Parsing prompt…"):
                parsed = parse_prompt(prompt)

            st.json(parsed)

            # 2) Show interactive preview
            try:
                geom = parsed.get("geometry", parsed)
                # handle list vs dict
                if isinstance(geom, list): geom = geom[0]
                fig = plot_geometry_preview(geom)
                st.subheader("🔧 Geometry Preview (3D)")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not preview geometry: {e}")

            with st.spinner("Running Ansys analysis (this may take a minute)…"):
                results = run_static_analysis(parsed)

            st.success("Analysis complete!")
            st.write("**Results:**")
            st.write(f"- Max displacement (Z): {results['max_displacement_z']:.6f} m")
            st.write(f"- Max von Mises stress: {results['max_stress']:.2f} Pa")

            # Show contour images if present
            for key in ("stress_img", "disp_img", "stress_side", "disp_side"):
                if key in results and os.path.exists(results[key]):
                    st.image(results[key], caption=key.replace("_", " ").title(), use_column_width=True)

            # Generate PDF
            pdf_path = "AI_FEA_Report.pdf"
            generate_pdf_report(results, parsed, output_path=pdf_path)

            # Provide download link
            with open(pdf_path, "rb") as f:
                btn = st.download_button(
                    label="📄 Download PDF Report",
                    data=f,
                    file_name=pdf_path,
                    mime="application/pdf"
                )

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)