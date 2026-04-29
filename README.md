# London Carpool Connect

London Carpool Connect is a data-driven carpooling prototype developed for my 6DATA007W Data Science and Analytics Final Project at the University of Westminster.

The project focuses on matching passengers and drivers in London using structured datasets, location coordinates, distance calculation, visualisation and a Streamlit web application.

## Project Aim

The aim of this project is to design and develop a working prototype that supports carpooling between passengers and drivers in London. The application uses passenger data, driver data and London location data to identify possible shared travel opportunities.

## Main Features

- Loads passenger, driver and London location datasets from CSV files
- Displays datasets in an interactive Streamlit interface
- Calculates distances between London locations
- Generates possible passenger-driver matches
- Shows London locations on an interactive Folium map
- Compares matching results using different threshold settings
- Provides visual outputs to support analysis and evaluation

## Project Structure

- `app.py` - Main Streamlit application file
- `requirements.txt` - List of Python libraries required to run the app
- `Data/` - Contains passenger, driver and London location CSV datasets
- `Notebooks/` - Contains development notebooks used for data generation, matching logic and API testing
- `outcomes/` - Contains matching outputs from the threshold 40 and threshold 60 experiments
- `README.md` - Project overview and running instructions

## Datasets

The project uses three main CSV datasets:

Dataset	Description
passengers.csv	Contains sample passenger journey information
drivers.csv	Contains sample driver journey information and available seats
LondonLocationsDataset.csv	Contains London location names with latitude and longitude coordinates

The datasets are sample/prototype datasets and do not contain live personal user data.

## Tools and Libraries:
- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Folium
- streamlit-folium
- OpenRouteService
- Visual Studio Code
- Jupyter Notebook

## Notebooks

The Notebooks/ folder contains the development notebooks used during the project:

- London_Carpool_Data_Generation (1).ipynb
  Used to generate and prepare the sample passenger, driver and London location datasets.
- London_Carpool_Matching_Algorithm (1).ipynb
  Used to develop and test the matching logic between passengers and drivers.
- London_Carpool_API_Matching (1).ipynb
  Used to explore route-related matching and API-supported functionality.

## Outputs

The outcomes/ folder contains results from two matching threshold experiments:

- Score_GoE_to_40
- Score_GoE_to_60

These outputs were used to compare how different threshold settings affect the number and quality of passenger-driver matches.

A lower threshold provides a wider set of possible matches, while a stricter threshold produces fewer but potentially stronger matches.

## Prototype Scope

This project is a prototype and focuses on demonstrating the core data-driven matching process. It is not a complete commercial carpooling platform.

The prototype includes:

- Dataset loading
- Location-based matching
- Distance calculation
- Match result generation
- Map visualisation
- Threshold comparison
- Basic data visualisation

The prototype does not include:

- Live user accounts
- Real-time GPS tracking
- Secure payments
- Live booking management
- Full route optimisation
- Identity verification
- User ratings
- Direct messaging between users

##Limitations

The project uses sample datasets rather than live user data. This means the results demonstrate the matching process but do not fully represent real London commuting behaviour.

The matching logic is rule-based and simplified. It mainly considers location, distance and available seats. A real-world carpooling platform would need to include additional factors such as pickup time, arrival time, route detour, user preferences, trust, safety and live traffic conditions.

The distance calculation also has limitations because straight-line distance does not always represent real driving distance. Future versions could use routing APIs more deeply to calculate actual road distance and estimated travel time.

## Future Improvements

Future development could include:

- Real-time route optimisation
- Larger and more realistic datasets
- User account creation and secure login
- Time-based passenger-driver matching
- Live booking and journey confirmation
- Driver and passenger ratings
- Identity verification and safety reporting
- Payment or cost-sharing features
- Mobile-friendly interface improvements
- More advanced route and detour calculations

## Author

Syed Hussnain Ali Wasti

BSc Data Science and Analytics

University of Westminster
