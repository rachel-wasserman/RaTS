import numpy as np

class broken_power_law:
    """
    Sharply broken power law lightcurve class.

    The light curve shape in burst-relative time s = t - tcrit:
        F(s) = F0 * (s / tau)^alpha_rise   for 0 <= s < tau   (rising to peak)
        F(s) = F0 * (s / tau)^alpha_decay  for s >= tau        (decaying from peak)

    where:
        - tau        = time from burst onset (tcrit) to peak
        - F0         = peak flux at s = tau
        - alpha_rise > 0  (default +2.0)
        - alpha_decay < 0 (default -1.5)

    The burst has a definite start but no strict end, so:
        edges = [1, 0]   (same convention as fred.py)
    """

    def __init__(self, alpha_rise=2.0, alpha_decay=-1.5):
        self.edges = [1, 0]
        self.alpha_rise  = float(alpha_rise)
        self.alpha_decay = float(alpha_decay)

    # ******************************************************************
    # Helpers
    # ******************************************************************

    def earliest_crit_time(self, start_survey, tau):
        return start_survey - tau

    def latest_crit_time(self, end_survey, tau):
        return end_survey

    # ******************************************************************
    # Integrated flux over one observation window
    # ******************************************************************

    def fluxint(self, F0, tcrit, tau, end_obs, start_obs):
        """
        Time-averaged flux of the broken power-law burst over [start_obs, end_obs].

        Parameters
        **********
        - F0        : float or array  – peak flux (Jy)
        - tcrit     : float or array  – burst start time (days)
        - tau       : float or array  – time from burst start to peak (days)
        - end_obs   : float           – end of observation window (days)
        - start_obs : float           – start of observation window (days)

        Returns
        *******
        mean_flux : same broadcast shape as (F0, tcrit, tau)
        """
        obs_dur = end_obs - start_obs

        alpha_r = self.alpha_rise
        alpha_d = self.alpha_decay
        exp_r   = alpha_r + 1.0
        exp_d   = alpha_d + 1.0

        # Burst-relative times; clamp to >= 0 (burst hasn't started before tcrit)
        s_obs_start = np.maximum(start_obs - tcrit, 0.0)
        s_obs_end   = np.maximum(end_obs   - tcrit, 0.0)

        # * Rising branch: s in [s_obs_start, min(s_obs_end, tau)] *
        s_rise_lo = s_obs_start
        s_rise_hi = np.minimum(s_obs_end, tau)

        # * Decaying branch: s in [max(s_obs_start, tau), s_obs_end] *
        s_decay_lo = np.maximum(s_obs_start, tau)
        s_decay_hi = s_obs_end

        with np.errstate(divide='ignore', invalid='ignore'):
            rise_int = np.where(
                s_rise_hi > s_rise_lo,
                (s_rise_hi ** exp_r - np.maximum(s_rise_lo, 0.0) ** exp_r) / (exp_r * tau ** alpha_r),
                0.0
            )
            decay_int = np.where(
                s_decay_hi > s_decay_lo,
                (s_decay_hi ** exp_d - s_decay_lo ** exp_d) / (exp_d * tau ** alpha_d),
                0.0
            )

        integral  = np.nan_to_num(rise_int) + np.nan_to_num(decay_int)
        mean_flux = np.multiply(F0, integral / obs_dur)
        return np.nan_to_num(mean_flux)

    # ******************************************************************
    # Boundary lines for the probability-contour plot
    # ******************************************************************

    def lines(self, xs, ys, durmax, max_distance, flux_err, obs):
        """
        Compute the red boundary lines for the probability-contour plot.

        For the boundary curves, we want the same physical meaning as in fred.py:
            both constraints are based on the burst being detected on the *decaying side*
            relative to the peak, not relative to the burst start.
            
            If u = time since peak, then for the broken power law decay branch:
                F(u) = F0 * (1 + u/tau)^alpha_decay,   u >= 0

        where tau is still the rise-to-peak timescale from the simulator.
        """

        gaps = np.zeros(len(obs) - 1, dtype=np.float64)
        for i in range(len(obs) - 1):
            gaps[i] = obs['start'][i + 1] - obs['start'][i] + obs['duration'][i]

        sens_last = obs['sens'][-1]
        lastdayobs = obs['duration'][-1]

        gap_idx = np.where(gaps == np.max(gaps))[0] + 1
        sens_maxgap = obs['sens'][gap_idx][0]
        duration_maxgap = obs['duration'][gap_idx][0]

        alpha_d = float(self.alpha_decay)
        exp_d = alpha_d + 1.0

        durmax_y = np.full(xs.shape, np.inf, dtype=np.float64)
        maxdist_y = np.full(xs.shape, np.inf, dtype=np.float64)

        def decay_integral_peak_relative(u_lo, u_hi, tau):
            """
            Integral of (1 + u/tau)^alpha_d du from u_lo to u_hi, with u >= 0.
            """
            if u_hi <= u_lo or tau <= 0:
                return 0.0
            
            a = 1.0 + u_lo / tau
            b = 1.0 + u_hi / tau

            if np.isclose(exp_d, 0.0):
                # alpha_d = -1 case
                return tau * np.log(b / a)

            return tau * (b**exp_d - a**exp_d) / exp_d

        for i, x in enumerate(xs):
            tau = 10.0**x

            # 1) durmax boundary:
            #    last observation catches the decaying tail, measured relative to PEAK
            try:
                u_lo = durmax - lastdayobs
                u_hi = durmax
                integral = decay_integral_peak_relative(u_lo, u_hi, tau)

                if integral > 0 and np.isfinite(integral):
                    durmax_y[i] = (1.0 + flux_err) * sens_last * lastdayobs / integral
            except Exception:
                durmax_y[i] = np.inf

            # 2) maxgap boundary:
            #    observation begins max_distance after peak, also peak-relative
            try:
                u_lo = max_distance
                u_hi = max_distance + duration_maxgap
                integral = decay_integral_peak_relative(u_lo, u_hi, tau)

                if integral > 0 and np.isfinite(integral):
                    maxdist_y[i] = (1.0 + flux_err) * sens_maxgap * duration_maxgap / integral
            except Exception:
                maxdist_y[i] = np.inf

        durmax_x = ' '
        maxdist_x = ' '

        y_min = np.amin(10**ys)
        y_max = np.amax(10**ys)

        durmax_y_indices = np.where((durmax_y < y_max) & (durmax_y > y_min))[0]
        maxdist_y_indices = np.where((maxdist_y < y_max) & (maxdist_y > y_min))[0]
        
        return durmax_x, maxdist_x, durmax_y, maxdist_y, durmax_y_indices, maxdist_y_indices
