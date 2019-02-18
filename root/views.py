from django.shortcuts import render
from django.http import HttpResponse, HttpRequest, JsonResponse
from django.template import loader
from . import solver
import sympy
from sympy import Symbol, latex
from typing import List, Union, Tuple

# Create your views here.


def index(request):
    template = loader.get_template('index.html')
    context = {}
    return HttpResponse(template.render(context, request))


def procedure_to_dict(procedure: solver.Procedure):
    return {
        procedure[i][0]: [latex(a) for a in procedure[i][1]] for i in range(len(procedure))
    }


def parseNum(args: List[str]) -> Tuple[bool, List[Union[solver.Number, Symbol]]]:
    parsed = []
    const_coeff = True
    for arg in args:
        try:
            p = float(arg)
            if p.is_integer():
                parsed.append(int(arg))
            else:
                parsed.append(p)
        except:
            # parse as symbol
            const_coeff = False

    return const_coeff, parsed


def ode(request: HttpRequest):
    if request.method == "POST":

        p: str = request.POST["p"]
        q: str = request.POST["q"]
        r: str = request.POST["r"]
        g: str = request.POST["g"]

        if p == "0" or p == "":
            # first order
            if "y" not in q and "y" not in r and "y" not in g:
                # linear
                pass
            pass
        else:
            if g == "" or g == "0":  # sec order homogeneous
                try:
                    # second order const

                    is_const, coeffs = parseNum([p, q, r])

                    if is_const:
                        y1, y2, procedure = solver.sec_order_const_coeff(
                            *coeffs
                        )

                        return JsonResponse({
                            "result": {
                                "y1": latex(y1),
                                "y2": latex(y2),
                            },
                            "procedure": procedure_to_dict(procedure),
                        })
                    else:
                        # non constant coefficient
                        pass

                except ValueError:
                    pass
                pass
            # second order

        return JsonResponse({"p": request.POST["p"]})
    else:
        return HttpResponse("""<div style="color:red">invalid request</div>""")
