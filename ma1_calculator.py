"""A teaching calculator for the theoretical MA(1) lag-1 autocorrelation."""

from dataclasses import dataclass
import math
from typing import Optional


@dataclass(frozen=True)
class MA1Result:
    theta: Optional[float]
    other_theta: Optional[float]
    status: str


def calculate_ma1_theta(rho: float) -> MA1Result:
    """Solve rho = theta / (1 + theta**2), using the plus-sign convention.

    States are invalid, zero, boundary, and invertible. Boundary roots are
    repeated and are not strictly invertible. At zero there is one finite root.
    """
    if not math.isfinite(rho) or abs(rho) > 0.5:
        return MA1Result(None, None, "invalid")
    if rho == 0:
        return MA1Result(0.0, None, "zero")
    if abs(rho) == 0.5:
        return MA1Result(math.copysign(1.0, rho), None, "boundary")

    # Rationalizing the small root avoids cancellation close to rho = 0.
    discriminant = (1.0 - 2.0 * rho) * (1.0 + 2.0 * rho)
    theta = (2.0 * rho) / (1.0 + math.sqrt(discriminant))
    other_theta = 1.0 / theta
    return MA1Result(theta, other_theta, "invertible")


def format_coefficient(value: float) -> str:
    """Keep small nonzero values and near-boundary values distinguishable."""
    return format(value, ".17g")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="MA(1) Invertibility Calculator", layout="centered")
    st.title("MA(1) Invertibility Calculator")
    st.markdown(
        r"""
Find the MA(1) coefficient **θ** from a **theoretical lag-1 autocorrelation ρ₁**.
We use the **plus-sign convention**:
"""
    )
    st.latex(r"X_t = \mu + \varepsilon_t + \theta\varepsilon_{t-1}")
    st.markdown(
        r"""
Here, $\varepsilon_t$ is white noise with mean zero and finite, positive variance
$\sigma^2$. Under this convention:
"""
    )
    st.latex(r"\rho_1 = \frac{\theta}{1+\theta^2}, \qquad \rho_k=0\quad(k\geq2)")
    st.caption(
        "If your course uses a minus sign before θ, the lag-1 formula changes sign. "
        "A compatible lag-1 value alone does not establish that a process is MA(1)."
    )

    rho = st.number_input(
        "Theoretical lag-1 autocorrelation, ρ₁",
        min_value=-0.5,
        max_value=0.5,
        value=0.3,
        step=0.01,
        format="%.10f",
        help=(
            "Theoretical MA(1) values lie between −0.5 and 0.5. "
            "Strict invertibility requires values strictly inside that interval. "
            "You can type a value directly."
        ),
    )
    st.caption(
        "This is a theoretical calculation, not a fitted model. A sample "
        "autocorrelation can lie outside this range because of sampling variation."
    )
    st.markdown("---")
    st.subheader("Calculation results")
    result = calculate_ma1_theta(rho)

    if result.status == "invalid":
        st.error("Enter a finite theoretical autocorrelation between −0.5 and 0.5.")
    elif result.status == "boundary":
        st.metric("Repeated MA(1) root, θ", format_coefficient(result.theta))
        st.warning(
            r"No strictly invertible solution: $|\rho_1|=0.5$ gives "
            r"$|\theta|=1$. This is the non-invertible boundary."
        )
        st.write("The quadratic has one repeated root, rather than two distinct roots.")
    elif result.status == "zero":
        st.metric("Invertible MA(1) coefficient, θ", "0")
        st.info(
            r"When $\rho_1=0$, the equation reduces to $-\theta=0$. "
            r"The only finite solution is $\theta=0$: white noise around the mean. "
            r"There is no second finite root."
        )
    else:
        st.metric("Invertible MA(1) coefficient, θ", format_coefficient(result.theta))
        st.markdown(r"This root satisfies the strict condition $|\theta|<1$.")
        if math.isfinite(result.other_theta):
            st.write(
                "Other root (non-invertible): θ = "
                + format_coefficient(result.other_theta)
            )
        else:
            st.info("The other root is too large in magnitude to display numerically.")
        st.markdown(
            r"The other root is $1/\theta$, with magnitude greater than 1. "
            r"Both roots give the same theoretical autocorrelation function."
        )
        reconstructed = result.theta / (1.0 + result.theta**2)
        st.caption("Check: θ / (1 + θ²) = " + format_coefficient(reconstructed))

    with st.expander("How the calculation works"):
        st.latex(r"\rho_1\theta^2-\theta+\rho_1=0")
        st.markdown(
            r"For $0<|\rho_1|<0.5$, there are two distinct reciprocal roots. "
            r"The invertible root can be written as:"
        )
        st.latex(r"\theta=\frac{2\rho_1}{1+\sqrt{1-4\rho_1^2}}")
        st.write(
            "This form avoids subtracting nearly equal numbers when the "
            "autocorrelation is close to zero. It also gives θ = 0 at ρ₁ = 0."
        )

    st.markdown("---")
    st.subheader("About invertibility and stationarity")
    st.markdown(
        r"""
**Invertibility** means that the innovations can be recovered from current and
past observations through a convergent filter with absolutely summable coefficients:
"""
    )
    st.latex(r"\varepsilon_t=\sum_{j=0}^{\infty}(-\theta)^j(X_{t-j}-\mu),\qquad |\theta|<1")
    st.markdown(
        r"""
**Stationarity is different.** With finite-variance white noise, an MA(1) process
is weakly stationary for every finite $\theta$, including non-invertible values.

**Why choose the invertible root?** Nonzero reciprocal coefficients give the same
ACF; invertibility selects a unique representation when $|\rho_1|<0.5$.
Matching autocovariances also requires changing the innovation variance: replacing
$\theta$ by $1/\theta$ requires replacing $\sigma^2$ by $\theta^2\sigma^2$.
"""
    )


if __name__ == "__main__":
    main()
