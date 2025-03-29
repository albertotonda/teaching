# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 21:55:24 2024

@author: Alberto
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sympy


# function that adds a number to the list
def add_number():
    if st.session_state['number'] is not None :
        if st.session_state["number"] not in st.session_state["points_explored"] :
            st.session_state['points_explored'].append(st.session_state['number'])
            st.session_state['number'] = None  # Reset the input

# functions
functions_names = ["Function 1", "Function 2"]
functions_list = ["x * sin(x) - 2 * sin(x)", "x^6 - 15*x^4 + 27*x^2"]
functions_x_boundaries = [[-12, 12], [-10, 10]]
functions_y_boundaries = [[-14, 11], [-100000, 100000]]


# check the state variables
if 'points_explored' not in st.session_state :
    st.session_state['points_explored'] = []
if 'best_value' not in st.session_state :
    st.session_state["best_value"] = -np.inf
if 'function_index' not in st.session_state :
    st.session_state["function_index"] = 0
if 'display_function' not in st.session_state :
    st.session_state["display_function"] = False

function = sympy.sympify(functions_list[st.session_state["function_index"]])
x = sympy.Symbol("x")
f = sympy.lambdify(x, function, "numpy") 
function_x_boundaries = functions_x_boundaries[st.session_state["function_index"]]
function_y_boundaries = functions_y_boundaries[st.session_state["function_index"]]

col1, col2, col3 = st.columns(3)

with col1 :
    st.number_input('Enter a number:', key='number', format='%.4f')

with col2 :
    st.button('Add number', on_click=add_number)

with col3 :
    # check which option is currently selected
    option = st.selectbox("Select a function", functions_names)
    option_index = functions_names.index(option)
    
    # if there is a change, we need to reset everything!
    if option_index != st.session_state["function_index"] :
        st.session_state["function_index"] = option_index
        st.session_state["points_explored"] = []
        
        function = sympy.sympify(functions_list[st.session_state["function_index"]])
        f = sympy.lambdify(x, function, "numpy") 
        function_x_boundaries = functions_x_boundaries[st.session_state["function_index"]]
        function_y_boundaries = functions_y_boundaries[st.session_state["function_index"]]
    
    st.session_state["display_function"] = st.checkbox("Display real function", value=False)

# part that plots the figure
fig, ax = plt.subplots()

if len(st.session_state["points_explored"]) > 0 :
    y_values = f(st.session_state["points_explored"])
    ax.scatter(st.session_state["points_explored"], y_values, alpha=0.7, zorder=2, label="Points guessed so far")
    st.session_state["best_value"] = max(y_values)

# also plot the full line
if st.session_state["display_function"] :
    x_function = np.linspace(function_x_boundaries[0], function_x_boundaries[1], num=1000)
    y_function = f(x_function)
    ax.plot(x_function, y_function, color='red', alpha=0.7, zorder=1, label="Real function")

ax.set_xlabel("$x$")
ax.set_ylabel("$y = f(x)$")
ax.set_title("%d points explored: best value found, x=%.4f" % (len(st.session_state["points_explored"]), 
                                                               st.session_state["best_value"]))

# set limits for the plot
ax.set_xlim(function_x_boundaries)
ax.set_ylim(function_y_boundaries)

ax.legend(loc='best')
ax.grid(True)

# this visualizes in streamlit (maybe)
st.pyplot(fig)

st.write(str(st.session_state["points_explored"]))
