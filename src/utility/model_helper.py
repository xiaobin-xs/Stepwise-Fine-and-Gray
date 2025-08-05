def map_model_name(model_name):
    if model_name == "coxph":
        model_name = "CoxPH"
    elif model_name == "dsm":
        model_name = "DSM"
    elif model_name == "deephit":
        model_name = "DeepHit"
    elif model_name == "deepsurv":
        model_name = "DeepSurv"
    else:
        pass
    return model_name