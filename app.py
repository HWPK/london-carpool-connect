"""
London Carpool Connect - Streamlit Prototype

This app was developed for my final year project. It uses synthetic London
commuter data to test passenger-driver matching based on journey time,
location similarity, accessibility requirements, seat availability and vehicle type.

The prototype includes:
- A baseline matching model using straight-line distance
- An API-based matching model using OpenRouteService for realistic route distance
- A dashboard for evaluating Good and Excellent matches
- A map view to visualise passenger and driver routes
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
import openrouteservice
import time


# Page setup
st.set_page_config(
    page_title="London Carpool Connect",
    layout="wide"
)


# Load data
# I cached the CSV loading so the app does not reload the files every time
# the user changes page or selects a different passenger.
@st.cache_data
def load_data():
    passengers = pd.read_csv("Data/passengers.csv")
    drivers = pd.read_csv("Data/drivers.csv")
    locations = pd.read_csv("Data/LondonLocationsDataset.csv")
    return passengers, drivers, locations


passengers, drivers, locations = load_data()


# Helper functions used by both matching models
def safe_bool(value):
    # Some boolean values from the CSV may be read as text, so I normalise them here.
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() == "true"

    return bool(value)


def haversine_distance(lat1, lon1, lat2, lon2):
    # Straight-line distance is used in the baseline model because it is fast
    # and does not require an external API.
    R = 6371

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def time_difference_minutes(time1, time2):
    # Calculates the difference between two HH:MM times in minutes.
    t1 = datetime.strptime(time1, "%H:%M")
    t2 = datetime.strptime(time2, "%H:%M")
    difference = abs((t1 - t2).total_seconds() / 60)
    return difference


def classify_match(score):
    # These categories are used throughout the app to make the score easier to understand.
    if score >= 80:
        return "Excellent match"
    elif score >= 60:
        return "Good match"
    elif score >= 40:
        return "Weak match"
    else:
        return "Not recommended"


# Baseline model: straight-line distance and weighted scoring
def calculate_match_score(passenger, driver):
    """
    Baseline matching score using time, location similarity, zones and vehicle type.
    The final score is between 0 and 100.
    """

    # A driver cannot be recommended if they have no seats available.
    if driver["seats_available"] <= 0:
        return 0

    # Accessibility is treated as a hard requirement because it is not optional for the passenger.
    if safe_bool(passenger["requires_accessibility"]) and not safe_bool(driver["accepts_accessibility"]):
        return 0

    time_diff = time_difference_minutes(
        passenger["preferred_time"],
        driver["departure_time"]
    )

    allowed_time_diff = passenger["time_flexibility_mins"] + driver["time_flexibility_mins"]

    if allowed_time_diff == 0:
        time_score = 0
    elif time_diff > allowed_time_diff:
        time_score = 0
    else:
        time_score = 100 - ((time_diff / allowed_time_diff) * 100)

    origin_distance = haversine_distance(
        passenger["origin_lat"],
        passenger["origin_lon"],
        driver["origin_lat"],
        driver["origin_lon"]
    )

    if origin_distance > passenger["max_pickup_distance_km"] + 5:
        origin_score = 0
    else:
        origin_score = max(0, 100 - (origin_distance * 10))

    destination_distance = haversine_distance(
        passenger["destination_lat"],
        passenger["destination_lon"],
        driver["destination_lat"],
        driver["destination_lon"]
    )

    if destination_distance > 10:
        destination_score = 0
    else:
        destination_score = max(0, 100 - (destination_distance * 10))

    zone_score = 0

    if passenger["origin_zone"] == driver["origin_zone"]:
        zone_score += 50

    if passenger["destination_zone"] == driver["destination_zone"]:
        zone_score += 50

    if driver["vehicle_type"] == "Electric":
        sustainability_score = 100
    elif driver["vehicle_type"] == "Hybrid":
        sustainability_score = 80
    elif driver["vehicle_type"] == "Petrol":
        sustainability_score = 50
    else:
        sustainability_score = 40

    # I gave the highest weights to time, origin and destination because these
    # are the main factors that decide whether a carpool journey is practical.
    # Zone similarity and vehicle type still matter, but they are secondary.
    final_score = (
        0.30 * time_score +
        0.25 * origin_score +
        0.25 * destination_score +
        0.10 * zone_score +
        0.10 * sustainability_score
    )

    return round(final_score, 2)


def find_matches_for_passenger(passenger, drivers_df, top_n=5):
    # This calculates every possible driver match for one selected passenger.
    match_results = []

    for _, driver in drivers_df.iterrows():
        score = calculate_match_score(passenger, driver)

        result = {
            "passenger_id": passenger["passenger_id"],
            "passenger_name": passenger["passenger_name"],
            "passenger_origin": passenger["origin"],
            "passenger_destination": passenger["destination"],
            "passenger_time": passenger["preferred_time"],
            "driver_id": driver["driver_id"],
            "driver_name": driver["driver_name"],
            "driver_origin": driver["origin"],
            "driver_destination": driver["destination"],
            "driver_time": driver["departure_time"],
            "driver_vehicle_type": driver["vehicle_type"],
            "driver_seats_available": driver["seats_available"],
            "match_score": score,
            "match_category": classify_match(score),
            "passenger_origin_lat": passenger["origin_lat"],
            "passenger_origin_lon": passenger["origin_lon"],
            "passenger_destination_lat": passenger["destination_lat"],
            "passenger_destination_lon": passenger["destination_lon"],
            "driver_origin_lat": driver["origin_lat"],
            "driver_origin_lon": driver["origin_lon"],
            "driver_destination_lat": driver["destination_lat"],
            "driver_destination_lon": driver["destination_lon"]
        }

        match_results.append(result)

    matches_df = pd.DataFrame(match_results)

    matches_df = matches_df.sort_values(
        by="match_score",
        ascending=False
    )

    return matches_df.head(top_n)


def find_good_excellent_matches_for_passenger(passenger, drivers_df, top_n=5):
    # I use 60 as the final recommendation threshold so weak matches are not
    # shown to the user. Weak matches can still be used separately for evaluation.
    matches_df = find_matches_for_passenger(
        passenger,
        drivers_df,
        top_n=len(drivers_df)
    )

    matches_df = matches_df[matches_df["match_score"] >= 60]

    return matches_df.head(top_n)


def find_best_matches_all(passengers_df, drivers_df):
    # This is used in the dashboard to find the best available driver for every passenger.
    all_best = []

    for _, passenger in passengers_df.iterrows():
        top_match = find_matches_for_passenger(
            passenger,
            drivers_df,
            top_n=1
        )

        all_best.append(top_match)

    best_df = pd.concat(all_best, ignore_index=True)

    best_df["final_status"] = np.where(
        best_df["match_score"] >= 60,
        "Good/Excellent match",
        "No good/excellent match"
    )

    return best_df


# API model: real route distance and driver detour
def get_driving_route(client, origin_lat, origin_lon, destination_lat, destination_lon):
    """
    Gets real driving distance and duration from OpenRouteService.
    OpenRouteService expects coordinates in longitude, latitude order.
    """

    coords = (
        (origin_lon, origin_lat),
        (destination_lon, destination_lat)
    )

    try:
        route = client.directions(
            coordinates=coords,
            profile="driving-car",
            format="geojson"
        )

        summary = route["features"][0]["properties"]["summary"]

        distance_km = summary["distance"] / 1000
        duration_mins = summary["duration"] / 60

        return round(distance_km, 2), round(duration_mins, 2)

    except Exception as e:
        st.error(f"API error: {e}")
        return None, None


def get_cached_driving_route(client, origin_name, origin_lat, origin_lon,
                             destination_name, destination_lat, destination_lon):
    # I added this cache so the same route is not requested from the API again
    # during the same app session. This also helps reduce waiting time.
    if "route_cache" not in st.session_state:
        st.session_state.route_cache = {}

    route_key = f"{origin_name}__to__{destination_name}"

    if route_key in st.session_state.route_cache:
        cached_route = st.session_state.route_cache[route_key]
        return cached_route["distance_km"], cached_route["duration_mins"]

    distance_km, duration_mins = get_driving_route(
        client,
        origin_lat,
        origin_lon,
        destination_lat,
        destination_lon
    )

    st.session_state.route_cache[route_key] = {
        "origin": origin_name,
        "destination": destination_name,
        "distance_km": distance_km,
        "duration_mins": duration_mins
    }

    # Small delay to avoid sending route requests too quickly.
    time.sleep(1)

    return distance_km, duration_mins


def calculate_api_match_score(passenger, driver, client):
    """
    Calculates a match score using real driving distance and estimated detour.
    Returns the score, detour minutes, detour kilometres and carpool duration.
    """

    if driver["seats_available"] <= 0:
        return 0, None, None, None

    if safe_bool(passenger["requires_accessibility"]) and not safe_bool(driver["accepts_accessibility"]):
        return 0, None, None, None

    time_diff = time_difference_minutes(
        passenger["preferred_time"],
        driver["departure_time"]
    )

    allowed_time_diff = passenger["time_flexibility_mins"] + driver["time_flexibility_mins"]

    if allowed_time_diff == 0:
        time_score = 0
    elif time_diff > allowed_time_diff:
        time_score = 0
    else:
        time_score = 100 - ((time_diff / allowed_time_diff) * 100)

    normal_distance, normal_duration = get_cached_driving_route(
        client,
        driver["origin"],
        driver["origin_lat"],
        driver["origin_lon"],
        driver["destination"],
        driver["destination_lat"],
        driver["destination_lon"]
    )

    leg1_distance, leg1_duration = get_cached_driving_route(
        client,
        driver["origin"],
        driver["origin_lat"],
        driver["origin_lon"],
        passenger["origin"],
        passenger["origin_lat"],
        passenger["origin_lon"]
    )

    leg2_distance, leg2_duration = get_cached_driving_route(
        client,
        passenger["origin"],
        passenger["origin_lat"],
        passenger["origin_lon"],
        passenger["destination"],
        passenger["destination_lat"],
        passenger["destination_lon"]
    )

    leg3_distance, leg3_duration = get_cached_driving_route(
        client,
        passenger["destination"],
        passenger["destination_lat"],
        passenger["destination_lon"],
        driver["destination"],
        driver["destination_lat"],
        driver["destination_lon"]
    )

    route_values = [
        normal_distance, normal_duration,
        leg1_distance, leg1_duration,
        leg2_distance, leg2_duration,
        leg3_distance, leg3_duration
    ]

    if any(value is None for value in route_values):
        return 0, None, None, None

    carpool_distance = leg1_distance + leg2_distance + leg3_distance
    carpool_duration = leg1_duration + leg2_duration + leg3_duration

    detour_km = carpool_distance - normal_distance
    detour_mins = carpool_duration - normal_duration

    if detour_km < 0:
        detour_km = 0

    if detour_mins < 0:
        detour_mins = 0

    if driver["max_detour_mins"] == 0:
        detour_score = 0
    elif detour_mins > driver["max_detour_mins"]:
        detour_score = 0
    else:
        detour_score = 100 - ((detour_mins / driver["max_detour_mins"]) * 100)

    passenger_direct_duration = leg2_duration

    if passenger_direct_duration <= 30:
        passenger_journey_score = 100
    elif passenger_direct_duration <= 60:
        passenger_journey_score = 80
    elif passenger_direct_duration <= 90:
        passenger_journey_score = 60
    else:
        passenger_journey_score = 40

    zone_score = 0

    if passenger["origin_zone"] == driver["origin_zone"]:
        zone_score += 50

    if passenger["destination_zone"] == driver["destination_zone"]:
        zone_score += 50

    if driver["vehicle_type"] == "Electric":
        sustainability_score = 100
    elif driver["vehicle_type"] == "Hybrid":
        sustainability_score = 80
    elif driver["vehicle_type"] == "Petrol":
        sustainability_score = 50
    else:
        sustainability_score = 40

    # In the API model, detour has the highest weight because a driver is unlikely
    # to accept a passenger if the extra journey time is too high.
    final_score = (
        0.35 * detour_score +
        0.25 * time_score +
        0.15 * passenger_journey_score +
        0.15 * zone_score +
        0.10 * sustainability_score
    )

    return round(final_score, 2), round(detour_mins, 2), round(detour_km, 2), round(carpool_duration, 2)


def find_api_matches_for_passenger(passenger, drivers_df, client, top_n=5):
    # This version is slower because it uses API route requests for each passenger-driver pair.
    api_matches = []

    for _, driver in drivers_df.iterrows():
        score, detour_mins, detour_km, carpool_duration = calculate_api_match_score(
            passenger,
            driver,
            client
        )

        result = {
            "passenger_id": passenger["passenger_id"],
            "passenger_name": passenger["passenger_name"],
            "passenger_origin": passenger["origin"],
            "passenger_destination": passenger["destination"],
            "passenger_time": passenger["preferred_time"],
            "driver_id": driver["driver_id"],
            "driver_name": driver["driver_name"],
            "driver_origin": driver["origin"],
            "driver_destination": driver["destination"],
            "driver_time": driver["departure_time"],
            "driver_vehicle_type": driver["vehicle_type"],
            "driver_seats_available": driver["seats_available"],
            "api_match_score": score,
            "detour_mins": detour_mins,
            "detour_km": detour_km,
            "carpool_duration_mins": carpool_duration,
            "match_category": classify_match(score)
        }

        api_matches.append(result)

    api_matches_df = pd.DataFrame(api_matches)

    api_matches_df = api_matches_df.sort_values(
        by="api_match_score",
        ascending=False
    )

    return api_matches_df.head(top_n)


def find_good_excellent_api_matches_for_passenger(passenger, drivers_df, client, top_n=5):
    # The API model uses the same 60+ threshold as the baseline model.
    api_matches_df = find_api_matches_for_passenger(
        passenger,
        drivers_df,
        client,
        top_n=len(drivers_df)
    )

    api_matches_df = api_matches_df[api_matches_df["api_match_score"] >= 60]

    return api_matches_df.head(top_n)


# Sidebar navigation
st.sidebar.title("🚗 London Carpool Connect")

page = st.sidebar.radio(
    "Go to:",
    [
        "Home",
        "Passenger Match Finder",
        "API Match Finder",
        "Driver Data",
        "Map View",
        "Dashboard",
        "Methodology"
    ]
)


# -----------------------------
# Home page
# -----------------------------
if page == "Home":
    st.title("London Carpool Connect")
    st.subheader("A London-only carpool matching prototype")

    st.write("""
    This prototype uses synthetic passenger and driver data to demonstrate how a
    carpooling system for London could match users based on location, time,
    seat availability, accessibility needs, and vehicle type.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Maximum Passengers in Dataset", len(passengers))

    with col2:
        st.metric("Maximum Drivers in Dataset", len(drivers))

    with col3:
        st.metric("London Locations Used", len(locations))

    st.info("""
    The app only displays Good and Excellent matches as recommendations.
    Good matches require a score of at least 60, and Excellent matches require a score of at least 80.
    """)

    st.subheader("Passenger Data Preview")
    st.dataframe(passengers.head())

    st.subheader("Driver Data Preview")
    st.dataframe(drivers.head())


# Passenger Match Finder page
elif page == "Passenger Match Finder":
    st.title("Passenger Match Finder")

    st.write("""
    Select a synthetic passenger to view their Good and Excellent driver matches.
    Weak and Not Recommended matches are calculated internally but hidden from the interface.
    """)

    passenger_options = passengers["passenger_id"] + " - " + passengers["passenger_name"]

    selected_passenger_label = st.selectbox(
        "Select a passenger:",
        passenger_options
    )

    selected_passenger_id = selected_passenger_label.split(" - ")[0]

    selected_passenger = passengers[
        passengers["passenger_id"] == selected_passenger_id
    ].iloc[0]

    st.markdown("### Selected Passenger Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Passenger ID:**", selected_passenger["passenger_id"])
        st.write("**Name:**", selected_passenger["passenger_name"])

    with col2:
        st.write("**Origin:**", selected_passenger["origin"])
        st.write("**Destination:**", selected_passenger["destination"])

    with col3:
        st.write("**Preferred Time:**", selected_passenger["preferred_time"])
        st.write("**Time Flexibility:**", selected_passenger["time_flexibility_mins"], "mins")

    top_n = st.slider(
        "Maximum number of Good/Excellent driver matches to show:",
        min_value=1,
        max_value=len(drivers),
        value=min(10, len(drivers))
    )

    top_matches = find_good_excellent_matches_for_passenger(
        selected_passenger,
        drivers,
        top_n=top_n
    )

    if top_matches.empty:
        st.warning("No Good or Excellent matches were found for this passenger.")
        st.stop()

    display_columns = [
        "driver_id",
        "driver_name",
        "driver_origin",
        "driver_destination",
        "driver_time",
        "driver_vehicle_type",
        "driver_seats_available",
        "match_score",
        "match_category"
    ]

    st.markdown("### Good and Excellent Driver Matches")
    st.dataframe(top_matches[display_columns])

    csv_matches = top_matches.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download baseline good/excellent matches as CSV",
        data=csv_matches,
        file_name="baseline_good_excellent_matches.csv",
        mime="text/csv"
    )

    best_match = top_matches.iloc[0]

    st.markdown("### Best Match Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Best Driver", best_match["driver_id"])

    with col2:
        st.metric("Match Score", best_match["match_score"])

    with col3:
        st.metric("Category", best_match["match_category"])

    st.write("**Best driver route:**")
    st.write(
        best_match["driver_origin"],
        "to",
        best_match["driver_destination"]
    )


# API Match Finder page
elif page == "API Match Finder":
    st.title("API Match Finder")

    st.write("""
    This page uses OpenRouteService to calculate real road-based driving distance
    and journey duration. It only displays Good and Excellent API-based matches.
    """)

    st.warning("""
    API matching uses live route requests, so it is slower than the baseline model.
    Even though the slider can go up to the maximum number of drivers, it is better
    to test with a small number of drivers first.
    """)

    ors_api_key = st.text_input(
        "Enter your OpenRouteService API key:",
        type="password"
    )

    if ors_api_key == "":
        st.info("Enter your ORS API key to use API-based matching.")
        st.stop()

    client = openrouteservice.Client(key=ors_api_key)

    passenger_options = passengers["passenger_id"] + " - " + passengers["passenger_name"]

    selected_passenger_label = st.selectbox(
        "Select passenger for API matching:",
        passenger_options
    )

    selected_passenger_id = selected_passenger_label.split(" - ")[0]

    selected_passenger = passengers[
        passengers["passenger_id"] == selected_passenger_id
    ].iloc[0]

    st.markdown("### Selected Passenger")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Passenger ID:**", selected_passenger["passenger_id"])
        st.write("**Name:**", selected_passenger["passenger_name"])

    with col2:
        st.write("**Origin:**", selected_passenger["origin"])
        st.write("**Destination:**", selected_passenger["destination"])

    with col3:
        st.write("**Preferred Time:**", selected_passenger["preferred_time"])
        st.write("**Max Detour:**", selected_passenger["max_detour_mins"], "mins")

    number_of_drivers = st.slider(
        "Maximum number of drivers to test with API:",
        min_value=1,
        max_value=len(drivers),
        value=min(5, len(drivers))
    )

    top_n = st.slider(
        "Maximum number of Good/Excellent API matches to show:",
        min_value=1,
        max_value=min(number_of_drivers, len(drivers)),
        value=min(3, number_of_drivers)
    )

    st.write("""
    The app will test the selected passenger against the first selected number
    of drivers from the synthetic driver dataset.
    """)

    sample_drivers = drivers.head(number_of_drivers)

    if st.button("Run API Matching"):
        with st.spinner("Calculating API-based matches. This may take a moment..."):
            api_matches = find_good_excellent_api_matches_for_passenger(
                selected_passenger,
                sample_drivers,
                client,
                top_n=top_n
            )

        if api_matches.empty:
            st.warning("No Good or Excellent API-based matches were found for this passenger.")
            st.stop()

        st.success("API matching complete.")

        display_columns = [
            "driver_id",
            "driver_name",
            "driver_origin",
            "driver_destination",
            "driver_time",
            "driver_vehicle_type",
            "driver_seats_available",
            "api_match_score",
            "detour_mins",
            "detour_km",
            "carpool_duration_mins",
            "match_category"
        ]

        st.markdown("### Good and Excellent API-Based Driver Matches")
        st.dataframe(api_matches[display_columns])

        csv_api_matches = api_matches.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download API good/excellent match results as CSV",
            data=csv_api_matches,
            file_name="api_good_excellent_matches_from_app.csv",
            mime="text/csv"
        )

        best_api_match = api_matches.iloc[0]

        st.markdown("### Best API Match Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Best Driver", best_api_match["driver_id"])

        with col2:
            st.metric("API Match Score", best_api_match["api_match_score"])

        with col3:
            st.metric("Detour", f"{best_api_match['detour_mins']} mins")

        with col4:
            st.metric("Category", best_api_match["match_category"])

        st.write("**Best API driver route:**")
        st.write(
            best_api_match["driver_origin"],
            "to",
            best_api_match["driver_destination"]
        )

        st.markdown("### Route Cache")

        if "route_cache" in st.session_state:
            cache_df = pd.DataFrame(st.session_state.route_cache).T.reset_index(drop=True)
            st.write("Routes currently stored in cache:", len(cache_df))
            st.dataframe(cache_df)

            csv_cache = cache_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download route cache as CSV",
                data=csv_cache,
                file_name="route_cache_from_app.csv",
                mime="text/csv"
            )


# Driver Data page
elif page == "Driver Data":
    st.title("Driver Data")

    st.write("""
    This page displays the synthetic driver dataset used in the prototype.
    Each driver has a London origin, destination, departure time, vehicle type,
    available seats, maximum detour preference, and accessibility option.
    """)

    st.markdown("### Driver Dataset")
    st.dataframe(drivers)

    st.markdown("### Driver Vehicle Type Breakdown")

    vehicle_counts = drivers["vehicle_type"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    vehicle_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Vehicle Type")
    ax.set_ylabel("Number of Drivers")
    ax.set_title("Driver Vehicle Types")

    for container in ax.containers:
        ax.bar_label(container)

    st.pyplot(fig)

    st.markdown("### Seats Available Breakdown")

    seat_counts = drivers["seats_available"].value_counts().sort_index()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    seat_counts.plot(kind="bar", ax=ax2)
    ax2.set_xlabel("Seats Available")
    ax2.set_ylabel("Number of Drivers")
    ax2.set_title("Seats Available by Driver")

    for container in ax2.containers:
        ax2.bar_label(container)

    st.pyplot(fig2)

    st.markdown("### Driver Departure Times")

    departure_counts = drivers["departure_time"].value_counts().sort_index()

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    departure_counts.plot(kind="bar", ax=ax3)
    ax3.set_xlabel("Departure Time")
    ax3.set_ylabel("Number of Drivers")
    ax3.set_title("Driver Departure Time Distribution")
    plt.xticks(rotation=45)

    for container in ax3.containers:
        ax3.bar_label(container)

    st.pyplot(fig3)


# Map View page
elif page == "Map View":
    st.title("Map View")

    st.write("""
    This page visualises the selected passenger journey and their best Good/Excellent
    matched driver journey on a London map. This version uses straight-line connections
    for visualisation. The API version can later show realistic road routes.
    """)

    passenger_options = passengers["passenger_id"] + " - " + passengers["passenger_name"]

    selected_passenger_label = st.selectbox(
        "Select passenger for map:",
        passenger_options
    )

    selected_passenger_id = selected_passenger_label.split(" - ")[0]

    selected_passenger = passengers[
        passengers["passenger_id"] == selected_passenger_id
    ].iloc[0]

    top_matches = find_good_excellent_matches_for_passenger(
        selected_passenger,
        drivers,
        top_n=1
    )

    if top_matches.empty:
        st.warning("No Good or Excellent matches were found for this passenger, so no recommended driver route can be shown.")
        st.stop()

    best_match = top_matches.iloc[0]

    london_map = folium.Map(
        location=[51.5074, -0.1278],
        zoom_start=11
    )

    folium.Marker(
        [selected_passenger["origin_lat"], selected_passenger["origin_lon"]],
        popup=f"Passenger Origin: {selected_passenger['origin']}",
        tooltip="Passenger Origin",
        icon=folium.Icon(color="blue")
    ).add_to(london_map)

    folium.Marker(
        [selected_passenger["destination_lat"], selected_passenger["destination_lon"]],
        popup=f"Passenger Destination: {selected_passenger['destination']}",
        tooltip="Passenger Destination",
        icon=folium.Icon(color="green")
    ).add_to(london_map)

    folium.Marker(
        [best_match["driver_origin_lat"], best_match["driver_origin_lon"]],
        popup=f"Driver Origin: {best_match['driver_origin']}",
        tooltip="Driver Origin",
        icon=folium.Icon(color="red")
    ).add_to(london_map)

    folium.Marker(
        [best_match["driver_destination_lat"], best_match["driver_destination_lon"]],
        popup=f"Driver Destination: {best_match['driver_destination']}",
        tooltip="Driver Destination",
        icon=folium.Icon(color="purple")
    ).add_to(london_map)

    # I used straight lines for the main map because they are fast and clear for the prototype.
    # The separate API model is used when more realistic road-based route data is needed.
    folium.PolyLine(
        locations=[
            [selected_passenger["origin_lat"], selected_passenger["origin_lon"]],
            [selected_passenger["destination_lat"], selected_passenger["destination_lon"]]
        ],
        tooltip="Passenger journey"
    ).add_to(london_map)

    folium.PolyLine(
        locations=[
            [best_match["driver_origin_lat"], best_match["driver_origin_lon"]],
            [best_match["driver_destination_lat"], best_match["driver_destination_lon"]]
        ],
        tooltip="Driver journey"
    ).add_to(london_map)

    st_folium(london_map, width=1000, height=600)

    st.markdown("### Match shown on map")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Passenger route:**")
        st.write(selected_passenger["origin"], "to", selected_passenger["destination"])

    with col2:
        st.write("**Best driver route:**")
        st.write(best_match["driver_origin"], "to", best_match["driver_destination"])

    st.write("**Match Score:**", best_match["match_score"])
    st.write("**Match Category:**", best_match["match_category"])


# Dashboard page
elif page == "Dashboard":
    st.title("Evaluation Dashboard")

    st.write("""
    This dashboard evaluates the baseline matching model across all synthetic
    passengers and drivers. It focuses only on Good and Excellent matches.
    """)

    best_matches = find_best_matches_all(passengers, drivers)

    # These are the final recommended matches used in the prototype evaluation.
    recommended_matches = best_matches[best_matches["match_score"] >= 60]

    total_passengers = len(passengers)
    total_drivers = len(drivers)
    matched_passengers = len(recommended_matches)
    unmatched_passengers = total_passengers - matched_passengers
    match_rate = (matched_passengers / total_passengers) * 100

    if matched_passengers > 0:
        average_score = recommended_matches["match_score"].mean()
    else:
        average_score = 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Maximum Passengers", total_passengers)

    with col2:
        st.metric("Maximum Drivers", total_drivers)

    with col3:
        st.metric("Good/Excellent Matches", matched_passengers)

    with col4:
        st.metric("Good/Excellent Match Rate", f"{match_rate:.1f}%")

    st.metric("Average Good/Excellent Match Score", f"{average_score:.2f}")

    display_columns = [
        "passenger_id",
        "passenger_name",
        "passenger_origin",
        "passenger_destination",
        "passenger_time",
        "driver_id",
        "driver_name",
        "driver_origin",
        "driver_destination",
        "driver_time",
        "driver_vehicle_type",
        "match_score",
        "match_category",
        "final_status"
    ]

    st.markdown("### Good and Excellent Matches Only")

    if recommended_matches.empty:
        st.warning("No Good or Excellent matches were found in the baseline dashboard evaluation.")
    else:
        st.dataframe(recommended_matches[display_columns])

        csv_dashboard = recommended_matches.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download dashboard good/excellent matches as CSV",
            data=csv_dashboard,
            file_name="dashboard_good_excellent_matches.csv",
            mime="text/csv"
        )

    st.markdown("### Good/Excellent Matches vs No Good/Excellent Match")

    status_counts = pd.Series({
        "Good/Excellent Matches": matched_passengers,
        "No Good/Excellent Match": unmatched_passengers
    })

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    status_counts.plot(kind="bar", ax=ax1)
    ax1.set_xlabel("Status")
    ax1.set_ylabel("Number of Passengers")
    ax1.set_title("Good/Excellent Matches vs No Good/Excellent Match")

    for container in ax1.containers:
        ax1.bar_label(container)

    st.pyplot(fig1)

    st.markdown("### Match Score Distribution for Good/Excellent Matches")

    if not recommended_matches.empty:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        recommended_matches["match_score"].plot(kind="hist", bins=10, ax=ax2)
        ax2.set_xlabel("Match Score")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of Good/Excellent Match Scores")
        ax2.axvline(60, linestyle="--", linewidth=2, label="Good threshold")
        ax2.axvline(80, linestyle="--", linewidth=2, label="Excellent threshold")
        ax2.legend()
        st.pyplot(fig2)

    st.markdown("### Match Category Breakdown")

    if not recommended_matches.empty:
        category_counts = recommended_matches["match_category"].value_counts()

        fig3, ax3 = plt.subplots(figsize=(8, 5))
        category_counts.plot(kind="bar", ax=ax3)
        ax3.set_xlabel("Match Category")
        ax3.set_ylabel("Number of Passengers")
        ax3.set_title("Good vs Excellent Match Categories")

        for container in ax3.containers:
            ax3.bar_label(container)

        st.pyplot(fig3)

    st.markdown("### Estimated Sustainability Impact")

    estimated_cars_reduced = matched_passengers
    estimated_co2_saved_kg = matched_passengers * 2.3

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Estimated Cars Reduced", estimated_cars_reduced)

    with col2:
        st.metric("Estimated CO₂ Saved", f"{estimated_co2_saved_kg:.1f} kg")

    st.write("""
    The CO₂ saving value is an indicative prototype estimate. It is used to
    demonstrate how the system could communicate possible environmental
    benefits, rather than as a precise emissions calculation.
    """)


# Methodology page
elif page == "Methodology":
    st.title("Methodology")

    st.markdown("### 1. Project Overview")
    st.write("""
    London Carpool Connect is a prototype carpooling application designed for
    London commuters. The system uses synthetic passenger and driver data to
    test whether users can be matched based on journey similarity, timing,
    available seats, accessibility requirements, and vehicle type.
    """)

    st.markdown("### 2. Synthetic Data")
    st.write("""
    Synthetic data was used instead of real user data. This avoids collecting
    sensitive personal information such as home addresses, workplace locations,
    names, contact details, or live movement patterns. Each synthetic passenger
    and driver has an origin, destination, coordinates, journey time, flexibility,
    and carpooling constraints.
    """)

    st.markdown("### 3. Baseline Matching Algorithm")
    st.write("""
    The baseline model uses Haversine straight-line distance between coordinates
    to estimate journey similarity. Each passenger is compared against each
    driver and given a match score out of 100.
    """)

    st.markdown("### 4. Match Quality Thresholds")
    st.write("""
    The final prototype only displays Good and Excellent matches to avoid
    recommending low-quality carpool pairings. A Good match requires a score of
    at least 60, while an Excellent match requires a score of at least 80. Weak
    and Not Recommended matches are still calculated internally but hidden from
    the user interface.
    """)

    st.markdown("### 5. Matching Criteria")
    st.write("""
    The score is based on:
    - Time compatibility
    - Origin proximity
    - Destination proximity
    - London zone similarity
    - Seat availability
    - Accessibility support
    - Vehicle sustainability
    """)

    st.markdown("### 6. API-Based Improvement")
    st.write("""
    An OpenRouteService API version was also developed. This version calculates
    real road-based driving time and distance between London locations. It
    improves realism by estimating driver detour when picking up and dropping off
    passengers.
    """)

    st.markdown("### 7. Evaluation")
    st.write("""
    The dashboard focuses on the number of passengers who received a Good or
    Excellent match. This provides a stricter and more realistic evaluation than
    simply counting weak matches as successful.
    """)

    st.markdown("### 8. Ethical Considerations")
    st.write("""
    The project avoids real personal travel data by using synthetic passenger and
    driver records. If developed as a real-world system, it would need
    GDPR-compliant data handling, clear user consent, secure authentication,
    location privacy protection, verified profiles, safety reporting, and clear
    terms of use.
    """)

    st.markdown("### 9. Limitations")
    st.write("""
    The current prototype has several limitations:
    - The main app currently uses straight-line route visualisation.
    - Synthetic data may not fully represent real commuter behaviour.
    - The match score weights were manually designed and could be improved.
    - The API model can be slower because it relies on external route requests.
    - The system does not include real user accounts or live booking.
    - The sustainability estimate is indicative rather than exact.
    """)

    st.markdown("### 10. Future Work")
    st.write("""
    Future work could include:
    - Real road-route geometry on the map
    - Live traffic data
    - Secure passenger and driver accounts
    - Real-time ride booking
    - Driver/passenger verification
    - More advanced optimisation algorithms
    - Larger-scale testing with realistic demand patterns
    """)