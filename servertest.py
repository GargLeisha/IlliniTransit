import requests

# Define the base URL of your OSRM server
base_url = "http://localhost:5000"

# Define the endpoint for the route service
endpoint = "/route/v1/driving"

# Define the coordinates (longitude, latitude) for the start and end points
coordinates = "-122.42,37.78;-122.45,37.91"

# Define additional parameters
params = {
    "steps": "true"
}

# Construct the full URL
url = f"{base_url}{endpoint}/{coordinates}"

# Make the request to the OSRM server
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()
    print(data)
else:
    print(f"Error: {response.status_code}")
