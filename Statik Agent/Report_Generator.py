# Report_Generator.py
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.platypus import Image

def generate_pdf_report(results: dict, parsed: dict, output_path: str = "Static_Analysis_Report.pdf"):
    # unwrap geometry if needed
    geo = parsed.get("geometry", parsed)
    if isinstance(geo, list):
        geo = geo[0]
    params = geo["parameters"]

    c = canvas.Canvas(output_path, pagesize=A4)
    w, h = A4

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, h-50, "Static Analysis Report")

    # Geometry
    c.setFont("Helvetica", 12)
    c.drawString(50, h-100, "Geometry:")
    if geo["type"] == "cylinder":
        c.drawString(70, h-120, f"Type: Cylinder")
        c.drawString(70, h-140, f"Radius: {params['radius']} m")
        c.drawString(70, h-160, f"Height: {params['height']} m")
    else:
        c.drawString(70, h-120, f"Type: Box")
        c.drawString(70, h-140, f"W: {params['width']} m, D: {params['depth']} m")
        c.drawString(70, h-160, f"Height: {params['height']} m")

    # Material & Load
    c.drawString(50, h-200, f"Material: {parsed.get('material', params.get('material','steel'))}")
    ld = parsed.get("load", {})
    c.drawString(50, h-220, f"Load: {ld.get('magnitude_N', 'N/A')} N (axial)")

    # Results Summary
    c.drawString(50, h-260, "Results Summary:")
    c.drawString(70, h-280, f"Max Displacement (Z): {results['max_displacement_z']:.6f} m")
    c.drawString(70, h-300, f"Max von Mises Stress: {results['max_stress']:.2f} Pa")

    # Embed ANSYS contour images
    y = h - 360
    c.drawString(50, y, "ANSYS Stress Contour:")
    c.drawImage(results["stress_img"], 50, y - 240, width=500, height=200)

    y = y - 280
    c.drawString(50, y, "ANSYS Displacement Contour (Z):")
    c.drawImage(results["disp_img"], 50, y - 240, width=500, height=200)

    # Optional deformed overlay
    y = y - 280
    c.drawString(50, y, "Deformed Shape:")
    c.drawImage(results["def_img"], 50, y - 240, width=500, height=200)

    # c.drawString(50, 50, "Top-Down von Mises Stress:")
    # c.drawImage(results["stress_img"], 50, 100, width=600, height=300)
    # c.drawString(50, 420, "Top-Down Displacement (Z):")
    # c.drawImage(results["disp_img"], 50, 470, width=600, height=300)

    # New page for side view
    c.showPage()
    c.drawString(50, 50, "Side-On von Mises Stress:")
    c.drawImage(results["stress_side"], 50, 100, width=600, height=300)
    c.drawString(50, 420, "Side-On Displacement (Z):")
    c.drawImage(results["disp_side"], 50, 470, width=600, height=300)

    c.showPage()
    c.save()
    print(f"✅ Report saved to: {output_path}")
