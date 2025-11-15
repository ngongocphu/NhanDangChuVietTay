# coding=utf-8
def outer_factory():

    def inner_factory(ag__):
        tf__lam = lambda y_true, y_pred: ag__.with_function_scope(lambda lscope: y_pred, 'lscope', ag__.STD)
        return tf__lam
    return inner_factory