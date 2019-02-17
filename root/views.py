from django.shortcuts import render
from django.http import HttpResponse, HttpRequest, JsonResponse
from django.template import loader
from . import solver
import sympy
from sympy import Symbol, latex

# Create your views here.


def index(request):
    template = loader.get_template('index.html')
    context = {
        'haha': 'hahahahhhahhahahh'
    }
    return HttpResponse(template.render(context, request))


def procedure_to_dict(procedure: solver.Procedure):
    return {
        procedure[i][0]: [latex(a) for a in procedure[i][1]] for i in range(len(procedure))
    }


def ode(request: HttpRequest):
    if request.method == "POST":

        p: str = request.POST["p"]
        q: str = request.POST["q"]
        r: str = request.POST["r"]
        g: str = request.POST["g"]

        if p == "0":
            # first order
            if "y" not in q and "y" not in r and "y" not in g:
                # linear
                pass
            pass
        else:
            try:
                # second order const coeff
                a = float(p)
                b = float(q)
                c = float(r)

                if a.is_integer():
                    a = int(a)

                if b.is_integer():
                    b = int(b)

                if c.is_integer():
                    c = int(c)

                y1, y2, procedure = solver.sec_order_const_coeff(a, b, c)

                return JsonResponse({
                    "result": {
                        "y1": latex(y1),
                        "y2": latex(y2),
                    },
                    "procedure": procedure_to_dict(procedure),
                })
            except ValueError:

                pass
            pass
            # second order

        return JsonResponse({"p": request.POST["p"]})
    else:
        return HttpResponse("""<div style="color:red">invalid request</div>""")
