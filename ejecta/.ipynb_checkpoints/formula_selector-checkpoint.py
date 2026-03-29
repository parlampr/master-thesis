# formula_selector.py
from gemma.ejecta import bns, nsbh

# Dictionary of implemented BNS and NSBH formulae
IMPLEMENTED_NSNS_FORMULAE = {
    "bns_disk_mass": ["kruger", "radice", "coughlin", "barbieri"],
    "bns_dynamical_mass": ["kruger", "radice", "coughlin"],
    "bns_dynamical_velocity": ["kruger", "radice", "coughlin", "barbieri"],
}

IMPLEMENTED_NSBH_FORMULAE = {
    "bhns_dynamical_velocity": ["foucart", "kawaguchi"],
    "bhns_dynamical_mass" : ["kruger", "kawaguchi"],
    "bhns_disk_mass": ["foucart"]
}

def select_bns_dynamical_mass_function(formula_name):
    """
    Select the appropriate BNS dynamical mass formula function based on the formula_name provided.

    Parameters:
        formula_name (str): Name of the desired formula to use for calculating BNS dynamical mass.

    Returns:
        function: A function reference for calculating BNS dynamical mass.
    """
    if formula_name in IMPLEMENTED_NSNS_FORMULAE["bns_dynamical_mass"]:
        if formula_name == "kruger":
            return bns.calculate_bns_dynamical_mass_kruger
        elif formula_name == "radice":
            return bns.calculate_bns_dynamical_mass_radice
        elif formula_name == "coughlin":
            return bns.calculate_bns_dynamical_mass_coughlin
        # Add more formulas if necessary
    else:
        raise ValueError(f"Formula '{formula_name}' is not implemented for BNS dynamical mass.")

def select_bns_dynamical_velocity_function(formula_name):
    """
    Select the appropriate BNS dynamical velocity formula function based on the formula_name provided.

    Parameters:
        formula_name (str): Name of the desired formula to use for calculating BNS dynamical velocity.

    Returns:
        function: A function reference for calculating BNS dynamical velocity.
    """
    if formula_name in IMPLEMENTED_NSNS_FORMULAE["bns_dynamical_velocity"]:
        if formula_name == "kruger":
            return bns.calculate_bns_dynamical_velocity_kruger
        elif formula_name == "radice":
            return bns.calculate_bns_dynamical_velocity_radice
        elif formula_name == "coughlin":
            return bns.calculate_bns_dynamical_velocity_coughlin
        # elif formula_name == "coughlinv2":
        #     return bns.calculate_bns_dynamical_velocity_coughlin_v2
        # Add more formulas if necessary
    else:
        raise ValueError(f"Formula '{formula_name}' is not implemented for BNS dynamical velocity.")

def select_bns_disk_mass_function(formula_name):
    """
    Select the appropriate BNS disk mass formula function based on the formula_name provided.

    Parameters:
        formula_name (str): Name of the desired formula to use for calculating BNS disk mass.

    Returns:
        function: A function reference for calculating BNS disk mass.
    """
    if formula_name in IMPLEMENTED_NSNS_FORMULAE["bns_disk_mass"]:
        if formula_name == "kruger":
            return bns.calculate_bns_disk_mass_kruger
        elif formula_name == "radice":
            return bns.calculate_bns_disk_mass_radice
        elif formula_name == "barbieri":
            return bns.calculate_bns_disk_mass_barbieri
        elif formula_name == "coughlin":
            return bns.calculate_bns_disk_mass_coughlin
        # elif formula_name == "coughlinv2":
        #     return bns.calculate_bns_disk_mass_coughlin_v2
        # Add more formulas if necessary
    else:
        raise ValueError(f"Formula '{formula_name}' is not implemented for BNS disk mass.")

# Similarly, you can add selectors for NSBH models

def select_nsbh_dynamical_mass_function(formula_name):
    """
    Select the appropriate NSBH dynamical mass formula function based on the formula_name provided.
    """
    if formula_name in IMPLEMENTED_NSBH_FORMULAE["bhns_dynamical_mass"]:
        if formula_name == "kruger":
            return nsbh.calculate_bhns_dynamical_mass_kruger
        elif formula_name == "kawaguchi":
            return nsbh.calculate_bhns_dynamical_mass_kawaguchi
        # Add more formulas if necessary
    else:
        raise ValueError(f"Formula '{formula_name}' is not implemented for NSBH dynamical mass.")

def select_nsbh_dynamical_velocity_function(formula_name):
    """
    Select the appropriate NSBH dynamical velocity formula function based on the formula_name provided.
    """
    if formula_name in IMPLEMENTED_NSBH_FORMULAE["bhns_dynamical_velocity"]:
        if formula_name == "foucart":
            return nsbh.calculate_bhns_dynamical_velocity_foucart
        elif formula_name == "kawaguchi":
            return nsbh.calculate_bhns_dynamical_velocity_kawaguchi
        # Add more formulas if necessary
    else:
        raise ValueError(f"Formula '{formula_name}' is not implemented for NSBH dynamical velocity.")

def select_nsbh_disk_mass_function(formula_name):
    """
    Select the appropriate NSBH disk mass formula function based on the formula_name provided.
    """
    if formula_name in IMPLEMENTED_NSBH_FORMULAE["bhns_disk_mass"]:
        if formula_name == "foucart":
            return nsbh.calculate_bhns_disk_mass_foucart
        # Add more formulas if necessary
    else:
        raise ValueError(f"Formula '{formula_name}' is not implemented for NSBH disk mass.")
