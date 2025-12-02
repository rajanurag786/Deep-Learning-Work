from math import degrees
import secrets

from flask import Flask, redirect, render_template, request, session
from flask_bootstrap import Bootstrap5
from flask_wtf import CSRFProtect, FlaskForm
from werkzeug.datastructures import MultiDict


from .forms import TwoPostWithTrussAndLoadForm
from .geometries.solar_ground_mounted import TwoPileWithTwoStrutsParams, TwoPileWithTwoStrutsSystem, TwoPileWithoutStrutsSystem
from .geometries.base import BeamProfileParams, CS_SB_150_62_18, CS_W146, MAT_S235
from .geometries.solar_ground_mounted import LoadParams, calculate_system

app = Flask(__name__)
app.secret_key = secrets.token_urlsafe(16)

bootstrap = Bootstrap5(app)
csrf = CSRFProtect(app)


@app.context_processor
def utility_processor():
    return dict(deg=degrees)


@app.route("/", methods=["GET", "POST"])
def index():
    formdata = session.pop("formdata", None)
    if formdata:
        form = TwoPostWithTrussAndLoadForm(MultiDict(formdata))
    else:
        form = TwoPostWithTrussAndLoadForm()
    if form.validate_on_submit():
        session["formdata"] = request.form
        return redirect("/results")
    return render_template("index.html", form=form)


@app.route("/results", methods=["GET"])
def results():
    formdata = session.get("formdata", None)
    if formdata:
        form = TwoPostWithTrussAndLoadForm(MultiDict(formdata))
        bp_pile = BeamProfileParams(MAT_S235, CS_W146)
        bp_girder = BeamProfileParams(MAT_S235, CS_SB_150_62_18)
        params = TwoPileWithTwoStrutsParams(
            form.geo.rigid_supports.data,
            form.geo.deg_earth.data,
            form.geo.deg_binder.data,
            form.geo.t.data,
            form.geo.h.data,
            form.geo.u.data,
            form.geo.u_r.data,
            bp_pile,
            bp_girder,
            form.geo.a.data,
            form.geo.s.data,
            form.geo.s.data,
            bp_girder,
        )
        system = TwoPileWithTwoStrutsSystem(params, nelem=40)
        load = LoadParams(
            form.geo.e.data,
            form.load.gk.data,
            form.load.sk.data,
            form.load.wab.data,
            form.load.wauf.data,
        )
        results = calculate_system(system, load)
        try:
            pass
        except:
            return render_template(
                "results.html", load=load, geo=params, e=form.geo.e.data, error=True
            )
        beams = {
            "Riegel": {"profil": bp_pile, "nodes": ["1", "2", "3", "4", "5", "6", "7"]},
            "Strebe": {"profil": bp_girder, "nodes": ["8", "9"]},
            "Stütze": {"profil": bp_girder, "nodes": ["10", "11"]},
        }
        return render_template(
            "results.html",
            load=load,
            geo=params,
            e=form.geo.e.data,
            beams=beams,
            supports=["A1", "A2"],
            gzt=results["gzt"],
            gzg=results["gzg"],
        )
    else:
        return redirect("/")
