import numpy as np
def apply_quadratic_volatility_model(strikes, spot, atm_vol, slope, quadratic_term, texp_years):
    """
    Apply the quadratic volatility model to new data points.
    
    Parameters:
        strikes (array-like): Array of strike prices.
        spot (float): Spot price.
        atm_vol (float): At-the-money volatility.
        slope (float): Slope of the linear term.
        quadratic_term (float): Coefficient of the quadratic term.
        texp_years (float): Time to expiration in years.
    
    Returns:
        array-like: Fitted volatilities for the given strikes.
    """
    log_strikes = np.log(strikes) - np.log(spot)
    #fitted_vols = atm_vol + (slope / np.sqrt(texp_years)) * log_strikes + quadratic_term * log_strikes**2
    fitted_vols = atm_vol + slope * log_strikes + quadratic_term * log_strikes**2
    fitted_vols= np.clip(fitted_vols, .05,.4)
    return fitted_vols