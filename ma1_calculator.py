import streamlit as st
import numpy as np

# Set up the page configuration for better aesthetics
st.set_page_config(
    page_title="MA(1) Invertibility Calculator",
    layout="centered",
    initial_sidebar_state="expanded"
)

def calculate_ma1_theta(rho):
    """
    Calculates the two possible theta coefficients for an MA(1) process
    given the autocorrelation at lag 1 (rho), and selects the invertible one.

    The relationship is: rho = theta / (1 + theta^2), which is a quadratic:
    (rho) * theta^2 - (1) * theta + (rho) = 0
    """
    # Check for the stationarity/invertibility boundary constraint: |rho| <= 0.5
    if abs(rho) > 0.5:
        return None, "Error: The absolute value of the Autocorrelation (rho) for an MA(1) process must be less than or equal to 0.5 to have real solutions for theta."

    # Coefficients for the quadratic equation A*theta^2 + B*theta + C = 0
    A = rho
    B = -1.0
    C = rho

    # Discriminant: B^2 - 4AC
    discriminant = B**2 - 4 * A * C

    # Since we already checked abs(rho) <= 0.5, the discriminant should be non-negative.
    # However, we handle the edge case near 0 just in case.
    if discriminant < 0:
        return None, "Error: Mathematical error (discriminant < 0), check input data."

    # Handle the boundary case where rho = 0.5 or rho = -0.5
    if abs(rho) == 0.5:
        # If rho = 0.5, theta = 1. If rho = -0.5, theta = -1.
        # This is the non-invertible boundary case, but it's the only real root.
        theta_1 = -B / (2 * A) # Should be 1 or -1
        return theta_1, f"Boundary Case: $|\rho| = 0.5$, which means $|\theta| = 1$. The process is on the boundary of invertibility."

    # Quadratic formula: (-B +/- sqrt(discriminant)) / (2A)
    # Note: A=rho is guaranteed to be non-zero here because we handled rho=0 separately below.
    sqrt_discriminant = np.sqrt(discriminant)
    theta_1 = (1 + sqrt_discriminant) / (2 * rho)
    theta_2 = (1 - sqrt_discriminant) / (2 * rho)

    # Invertibility condition: we must choose the root where |theta| < 1
    if abs(theta_1) < 1:
        return theta_1, "Success"
    elif abs(theta_2) < 1:
        return theta_2, "Success"
    else:
        # This case should ideally not happen if |rho| < 0.5
        return None, "Error: Neither root satisfied the invertibility condition ( $|\theta| < 1$ )."


# --- Streamlit UI Components ---

st.title("MA(1) Invertibility Coefficient $\\theta$ Calculator")
st.markdown("""
This app calculates the $\\theta$ coefficient for an **Invertible Moving Average process of order 1 (MA(1))**
based on its first-order autocorrelation, $\\rho$.

The relationship is given by:
$$\\rho = \\frac{\\theta}{1 + \\theta^2}$$

We solve the resulting quadratic equation $\\rho \\theta^2 - \\theta + \\rho = 0$ for $\\theta$, selecting the root
that satisfies the invertibility condition: $|\theta| < 1$.
""")

# Input field for ACF(1) value
rho_input = st.number_input(
    label="Enter the Autocorrelation at Lag 1 ($\\rho$):",
    min_value=-0.5,
    max_value=0.5,
    value=0.3,
    step=0.01,
    format="%.4f",
    help="The value of |rho| must be less than or equal to 0.5 for an MA(1) process."
)

st.markdown("---")

# Calculation and Display
if rho_input != 0:
    # Perform the calculation
    theta_result, status_message = calculate_ma1_theta(rho_input)

    if theta_result is not None:
        # Calculate the other, non-invertible root (which is 1/theta) for context
        # Handle the case where theta_result is 0 (which happens if rho is 0, but we checked rho!=0)
        # and the boundary cases (theta=1 or -1)
        if abs(theta_result) == 1:
            # Boundary case |rho|=0.5
            theta_other = theta_result
        else:
            theta_other = 1 / theta_result

        # Display results
        st.subheader("Calculation Results")

        st.metric(
            label="Invertible MA(1) Coefficient ($\\theta$)",
            value=f"{theta_result:.6f}",
            # Reverted to single backslash for help text
            help="This root satisfies the invertibility condition $|\\theta| < 1$."
        )

        # Reverted to single backslash within the f-string for better markdown parsing
        st.info(
            f"""
            The two roots of the quadratic equation are:
            * **Invertible Root ($\\theta$):** `{theta_result:.6f}` (where $|\\theta| \\le 1$)
            * **Non-invertible Root ($1/\\theta$):** `{theta_other:.6f}` (where $|1/\\theta| \\ge 1$)
            """
        )

        if "Boundary Case" in status_message:
             st.warning(status_message)

    else:
        # Display error message
        st.error(status_message)

else:
    # Handle the trivial case where rho = 0
    st.info("If the Autocorrelation at Lag 1 ($\\rho$) is 0, the MA(1) coefficient ($\theta$) is also 0.")
    st.metric(
        label="Invertible MA(1) Coefficient ($\\theta$)",
        value="0.0"
    )

st.markdown("---")
st.markdown("#### About Invertibility")
st.markdown(
    """
    For an MA(1) process to be **invertible**, meaning it can be represented as an
    infinite order autoregressive (AR($\infty$)) process, the coefficient $\\theta$ must satisfy
    $|\\theta| < 1$.

    If you select the root $|\\theta| > 1$, the resulting MA(1) process is *not* invertible.
    However, it has the *exact same* autocorrelation function as the invertible process with the coefficient $1/\\theta$.
    """
)
