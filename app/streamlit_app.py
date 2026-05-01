"""Planificador de rutas multi-sucursal — aplicación Streamlit.

Permite cargar un CSV de clientes y obtener un plan diario coordinado entre
sucursales, con visualización del mapa, kilometraje total y comparativa
antes/después. Si no hay CSV cargado, usa el dataset sintético del repo.
"""

from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="Gestión de rutas multi-sucursal · Leanmaster Pymes",
    page_icon=":truck:",
    layout="wide",
)


def main() -> None:
    st.title("Gestión de rutas multi-sucursal")
    st.caption("Machine Learning + Programación Dinámica · Leanmaster Pymes")
    st.info("Aplicación en construcción. La versión funcional cargará en breve.")

    with st.sidebar:
        st.markdown("---")
        st.markdown(
            "**Código abierto · Licencia MIT** · "
            "[github.com/leanmasterpymes/gestion_ruta]"
            "(https://github.com/leanmasterpymes/gestion_ruta)"
        )


if __name__ == "__main__":
    main()
