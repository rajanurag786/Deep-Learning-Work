from flask_wtf import FlaskForm
from wtforms import (
    FormField,
    SubmitField,
    FloatField,
    IntegerField,
    BooleanField,
)
from wtforms.validators import DataRequired, NumberRange, InputRequired


class HL:
    """Unescaped HTML label"""

    def __init__(self, html_text):
        self.html_text = html_text

    def __html__(self):
        return self.html_text


class HLF(HL):
    """Unescaped HTML label with bold heading"""

    def __init__(self, html_text, heading):
        super().__init__(html_text)
        self.heading = heading

    def __html__(self):
        return f'<b style="font-size:1.2em">{self.heading}</b> {self.html_text}'


class LoadForm(FlaskForm):
    gk = FloatField(
        "Eigengewicht Solarmodule [kN/m²]",
        validators=[InputRequired(), NumberRange(0, 1)],
        default=0.60,
    )
    sk = FloatField(
        "Schneelast auf Module [kN/m²]",
        validators=[InputRequired(), NumberRange(0, 5)],
        default=0.5,
    )
    wab = FloatField(
        "Winddruck [kN/m²]", validators=[InputRequired(), NumberRange(0, 5)], default=1.0
    )
    wauf = FloatField(
        "Windsog [kN/m²]", validators=[InputRequired(), NumberRange(-5, 0)], default=-1.0
    )


class TwoPostForm(FlaskForm):
    e = FloatField(
        HLF("Einflussbreite (Stützenabstand in y-Richtung) [m]", "e:"),
        validators=[DataRequired(), NumberRange(0.5, 20.0)],
        default=4,
    )
    deg_earth = IntegerField(
        HLF("Neigungswinkel Gelände [°]", "α:"), validators=[NumberRange(0, 30)], default=0
    )
    deg_binder = IntegerField(
        HLF("Neigungswinkel Panelebene [°]", "β:"), validators=[NumberRange(0, 50)], default=0
    )
    t = FloatField(
        HLF("Tiefe Verankerungs-/Einspannpunkt [m]", "t:"),
        validators=[InputRequired(), NumberRange(0, 2)],
        default=0.5,
    )
    rigid_supports = BooleanField("Eingespannte Auflager", default=True)
    h = FloatField(
        HLF("Stützenhöhe [m]", "h:"), validators=[DataRequired(), NumberRange(0.5, 10)], default=2
    )
    a = FloatField(
        HLF("Stützweite (horizontaler Pfostenabstand) [m]", "a:"),
        validators=[DataRequired(), NumberRange(0.5, 10)],
        default=3,
    )
    u = FloatField(
        HLF("Horizontaler Überstand links [m]", "u<sub>L</sub>:"),
        validators=[DataRequired(), NumberRange(0.5, 10)],
        default=2,
    )
    u_r = FloatField(
        HLF("Horizontaler Überstand rechts [m]", "u<sub>R</sub>:"),
        validators=[DataRequired(), NumberRange(0.5, 10)],
        default=2,
    )


class TwoPostWithTrussForm(TwoPostForm):
    s = FloatField(
        HLF("Schenkellänge Abstrebung [m]", "s:"),
        validators=[DataRequired(), NumberRange(0.2, 5)],
        default=1.5,
    )


class TwoPostWithTrussAndLoadForm(FlaskForm):
    load = FormField(LoadForm)
    geo = FormField(TwoPostWithTrussForm)
    submit = SubmitField("Berechnen")
